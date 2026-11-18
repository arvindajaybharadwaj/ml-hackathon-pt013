"""Five baselines, grouped CV and optional Bayesian tuning. Nothing runs on import."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .data import load_dataset, fingerprint
from .models import MODEL_NAMES, build_model, suggest_parameters
from .predictor import Predictor
from .metrics import evaluate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def fit(dataset, name, seed, params=None, options=None):
    rows = dataset.targets()
    if set(rows.y.unique()) != {0, 1}:
        raise ValueError('Training needs both match and non-match candidate examples')
    options = options or {}
    pipeline = build_model(name, seed, params, **options)
    start = perf_counter()
    with threadpool_limits(limits=options.get('threads', 1)):
        pipeline.fit(rows[dataset.schema['feature_names']].to_numpy(dtype=float), rows.y)
    estimator = pipeline.named_steps['model']
    predictor = Predictor(pipeline, dataset.schema, {'model_name': name, 'seed': seed,
                          'training_device': getattr(estimator, 'device_', 'cpu')})
    return predictor, perf_counter() - start


def measure(predictor, dataset):
    start = perf_counter()
    with threadpool_limits(limits=1):
        scores = predictor.score_candidates(dataset)
    scoring_seconds = perf_counter() - start
    metrics, predictions = evaluate(dataset, scores)
    metrics['inference_seconds'] = scoring_seconds
    return metrics, scores, predictions


def validate_model(train, folds, name, seed, params=None, options=None):
    metrics = []
    for fold in folds:
        started = perf_counter()
        predictor, seconds = fit(train.subset(fold['train_trips']), name, seed, params, options)
        result, _, _ = measure(predictor, train.subset(fold['validation_trips']))
        result.update(fit_seconds=seconds, wall_seconds=perf_counter()-started)
        metrics.append(result)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Teammate feature-table directory')
    parser.add_argument('--output', type=Path, default=Path('runs/baseline'))
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--trials', type=int, default=0, help='0 skips tuning; >=4 enables Bayesian TPE')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES + ['torch_mlp'], default=MODEL_NAMES)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto',
                        help='Device for torch_mlp only; sklearn models use CPU')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--benchmark', action='store_true', help='Time full-data fits, estimate CV/tuning runtime; no selection')
    parser.add_argument('--train-trip-limit', type=int, help='Smoke test only: use this many whole training trips')
    args = parser.parse_args()
    if args.threads < 1 or args.epochs < 1:
        parser.error('--threads and --epochs must be positive')
    if len(set(args.models)) != len(args.models):
        parser.error('--models cannot contain duplicates')
    if 'torch_mlp' in args.models:
        import torch
        from .neural import resolve_device
        args.device = resolve_device(args.device)
        torch.set_num_threads(args.threads)
        print(f'PyTorch training device: {args.device}', flush=True)
    options = dict(device=args.device, epochs=args.epochs, threads=args.threads)
    started = perf_counter()
    if args.trials != 0 and args.trials < 4:
        parser.error('--trials must be 0 or >=4 (two startup trials then Bayesian proposals)')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty; choose another --output directory')
    dataset = load_dataset(args.input)
    train = dataset.partition('train')
    student_test = dataset.partition('student_test')
    if args.train_trip_limit is not None:
        if args.train_trip_limit < args.folds:
            parser.error('--train-trip-limit must be >= --folds')
        # Seeded choice of WHOLE trips, never candidate-row sampling.
        ids = np.array(sorted(train.observations.trip_id.unique()))
        ids = np.random.default_rng(args.seed).permutation(ids)[:args.train_trip_limit]
        train = train.subset(ids)
    print(f'Real data: {train.observations.trip_id.nunique()} training trips, '
          f'{len(train.candidates):,} training candidates; '
          f'{student_test.observations.trip_id.nunique()} student-test trips', flush=True)
    if args.folds < 2 or args.folds > train.observations.trip_id.nunique():
        parser.error('--folds must be between 2 and the number of training trips')
    # Split original observations by complete trip, not candidate rows.
    observations = train.observations.reset_index(drop=True)
    folds = []
    for fit_ids, val_ids in GroupKFold(n_splits=args.folds).split(observations, groups=observations.trip_id):
        folds.append({
            'train_trips': sorted(observations.iloc[fit_ids].trip_id.unique().tolist()),
            'validation_trips': sorted(observations.iloc[val_ids].trip_id.unique().tolist()),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'running', 'data_fingerprint': fingerprint(dataset),
                'arguments': vars(args), 'train_trips': sorted(train.observations.trip_id.unique()),
                'student_test_trips': sorted(student_test.observations.trip_id.unique()),
                'selection_metric': 'trip-grouped CV binary candidate Macro F1',
                'smoke_test': args.train_trip_limit is not None}
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'folds.json', folds)
    results, fold_results, configs = [], {}, {}
    for name in args.models:
        model_started = perf_counter()
        print(f'Training {name}', flush=True)
        cv = [] if args.benchmark else validate_model(train, folds, name, args.seed, options=options)
        predictor, fit_seconds = fit(train, name, args.seed, options=options)
        metrics, candidate_scores, predictions = measure(predictor, student_test)
        train_metrics, _, _ = measure(predictor, train)
        results.append({
            'model': name,
            'cv_candidate_macro_f1': float(np.mean([x['candidate_macro_f1'] for x in cv])) if cv else None,
            'cv_candidate_macro_f1_std': float(np.std([x['candidate_macro_f1'] for x in cv])) if cv else None,
            'train_candidate_macro_f1': train_metrics['candidate_macro_f1'],
            'fit_seconds': fit_seconds, 'wall_seconds': perf_counter() - model_started,
            'training_device': predictor.metadata['training_device'],
            **{f'student_test_{k}': v for k, v in metrics.items()},
        })
        fold_results[name] = cv
        configs[name] = predictor.pipeline.named_steps['model'].get_params(deep=False)
        manifest.update(completed_models=[row['model'] for row in results])
        write_json(args.output / 'manifest.json', manifest)
        model_dir = args.output / name
        predictor.save(model_dir / 'model')
        write_json(model_dir / 'metrics.json', metrics)
        write_json(model_dir / 'cv_metrics.json', cv)
        estimator = predictor.pipeline.named_steps['model']
        if hasattr(estimator, 'loss_curve_'):
            write_json(model_dir / 'training_loss.json', estimator.loss_curve_)
        pd.DataFrame(results).to_csv(args.output / 'comparison.csv', index=False)
        print(f'  fit={fit_seconds:.1f}s total={results[-1]["wall_seconds"]:.1f}s '
              f'test_top1={metrics["top1_accuracy"]:.4f}', flush=True)
    if args.benchmark:
        # Full-data timing: CV fits use fewer rows, but leave headroom for tuning complexity.
        costs = [r['wall_seconds'] for r in results]
        estimate = (args.folds + 1) * sum(costs) + (2 * args.trials * args.folds + 2) * max(costs)
        write_json(args.output / 'runtime_estimate.json', {
            'estimated_full_run_seconds': estimate, 'threshold_seconds': 600,
            'recommend_modal': estimate > 600, 'benchmark_seconds': perf_counter()-started,
            'note': 'Conservative projection from full-data fits and evaluation; tuning cost varies.'})
        manifest.update(status='complete', wall_seconds=perf_counter()-started)
        write_json(args.output / 'manifest.json', manifest)
        print(f'Projected CV + {args.trials} tuning trials: {estimate / 60:.1f} minutes. '
              f'{"Use Modal" if estimate > 600 else "Local training is suitable"}.', flush=True)
        return
    best = max(results, key=lambda row: row['cv_candidate_macro_f1'])
    name, selected_params = best['model'], None
    if args.trials:
        import optuna  # Only needed when tuning is requested.
        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=2),
            storage=f'sqlite:///{(args.output / "tuning.sqlite3").resolve()}',
        )
        def objective(trial):
            params = suggest_parameters(trial, name)
            cv = validate_model(train, folds, name, args.seed, params, options)
            trial.set_user_attr('fold_metrics', cv)
            return float(np.mean([row['candidate_macro_f1'] for row in cv]))
        study.optimize(objective, n_trials=args.trials)
        study.trials_dataframe().to_csv(args.output / 'tuning_trials.csv', index=False)
        tuned, seconds = fit(train, name, args.seed, study.best_params, options)
        tuned_metrics, _, _ = measure(tuned, student_test)
        tuned_train, _, _ = measure(tuned, train)
        tuned_cv = validate_model(train, folds, name, args.seed, study.best_params, options)
        fold_results[name + '__tuned'] = tuned_cv
        configs[name + '__tuned'] = tuned.pipeline.named_steps['model'].get_params(deep=False)
        results.append({
            'model': name + '__tuned', 'cv_candidate_macro_f1': study.best_value,
            'cv_candidate_macro_f1_std': float(np.std([x['candidate_macro_f1'] for x in tuned_cv])),
            'train_candidate_macro_f1': tuned_train['candidate_macro_f1'], 'fit_seconds': seconds,
            **{f'student_test_{k}': v for k, v in tuned_metrics.items()},
        })
        if study.best_value > best['cv_candidate_macro_f1']:
            selected_params = study.best_params
    # Selection uses training CV only, never student-test performance.
    selected, _ = fit(train, name, args.seed, selected_params, options)
    selected.metadata.update(
        parameters=selected.pipeline.named_steps['model'].get_params(deep=False),
        tuned=selected_params is not None, input_directory=str(args.input.resolve()),
        versions={p: importlib.metadata.version(p) for p in ['scikit-learn', 'numpy', 'pandas', 'joblib']},
    )
    selected.save(args.output / 'model')
    from .predictor import load_model
    reloaded = load_model(args.output / 'model')
    sample = student_test.candidates.head(128)
    batch = {'candidates': sample, 'schema_version': dataset.schema['schema_version']}
    np.testing.assert_allclose(selected.score_candidates(batch).p_match,
                               reloaded.score_candidates(batch).p_match)
    metrics, scores, predictions = measure(selected, student_test)
    pd.DataFrame(results).to_csv(args.output / 'comparison.csv', index=False)
    write_json(args.output / 'cv_metrics.json', fold_results)
    write_json(args.output / 'configurations.json', configs)
    write_json(args.output / 'metrics.json', metrics)
    scores.to_csv(args.output / 'candidate_scores.csv', index=False)
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    manifest.update(status='complete', selected_model=name, selected_tuned=selected_params is not None,
                    wall_seconds=perf_counter()-started)
    write_json(args.output / 'manifest.json', manifest)
    print(f'Saved comparison and selected {name} model to {args.output}')


if __name__ == '__main__':
    main()
