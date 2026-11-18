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
from .data import load_dataset
from .models import MODEL_NAMES, build_model, suggest_parameters
from .predictor import Predictor
from .metrics import evaluate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def fit(dataset, name, seed, params=None):
    rows = dataset.targets()
    if set(rows.y.unique()) != {0, 1}:
        raise ValueError('Training needs both match and non-match candidate examples')
    pipeline = build_model(name, seed, params)
    start = perf_counter()
    with threadpool_limits(limits=1):
        pipeline.fit(rows[dataset.schema['feature_names']].to_numpy(dtype=float), rows.y)
    predictor = Predictor(pipeline, dataset.schema, {'model_name': name, 'seed': seed})
    return predictor, perf_counter() - start


def measure(predictor, dataset):
    start = perf_counter()
    with threadpool_limits(limits=1):
        scores = predictor.score_candidates(dataset)
    scoring_seconds = perf_counter() - start
    metrics, predictions = evaluate(dataset, scores)
    metrics['inference_seconds'] = scoring_seconds
    return metrics, scores, predictions


def validate_model(train, folds, name, seed, params=None):
    metrics = []
    for fold in folds:
        predictor, _ = fit(train.subset(fold['train_trips']), name, seed, params)
        result, _, _ = measure(predictor, train.subset(fold['validation_trips']))
        metrics.append(result)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Teammate feature-table directory')
    parser.add_argument('--output', type=Path, default=Path('runs/baseline'))
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--trials', type=int, default=0, help='0 skips tuning; >=4 enables Bayesian TPE')
    args = parser.parse_args()
    if args.trials != 0 and args.trials < 4:
        parser.error('--trials must be 0 or >=4 (two startup trials then Bayesian proposals)')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty; choose another --output directory')
    dataset = load_dataset(args.input)
    train = dataset.partition('train')
    student_test = dataset.partition('student_test')
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
    write_json(args.output / 'folds.json', folds)
    results, fold_results, configs = [], {}, {}
    for name in MODEL_NAMES:
        print(f'Training {name}', flush=True)
        cv = validate_model(train, folds, name, args.seed)
        predictor, fit_seconds = fit(train, name, args.seed)
        metrics, _, _ = measure(predictor, student_test)
        train_metrics, _, _ = measure(predictor, train)
        results.append({
            'model': name,
            'cv_candidate_macro_f1': float(np.mean([x['candidate_macro_f1'] for x in cv])),
            'cv_candidate_macro_f1_std': float(np.std([x['candidate_macro_f1'] for x in cv])),
            'train_candidate_macro_f1': train_metrics['candidate_macro_f1'],
            'fit_seconds': fit_seconds, **{f'student_test_{k}': v for k, v in metrics.items()},
        })
        fold_results[name] = cv
        configs[name] = predictor.pipeline.named_steps['model'].get_params(deep=False)
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
            cv = validate_model(train, folds, name, args.seed, params)
            trial.set_user_attr('fold_metrics', cv)
            return float(np.mean([row['candidate_macro_f1'] for row in cv]))
        study.optimize(objective, n_trials=args.trials)
        study.trials_dataframe().to_csv(args.output / 'tuning_trials.csv', index=False)
        tuned, seconds = fit(train, name, args.seed, study.best_params)
        tuned_metrics, _, _ = measure(tuned, student_test)
        tuned_train, _, _ = measure(tuned, train)
        tuned_cv = validate_model(train, folds, name, args.seed, study.best_params)
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
    selected, _ = fit(train, name, args.seed, selected_params)
    selected.metadata.update(
        parameters=selected.pipeline.named_steps['model'].get_params(deep=False),
        tuned=selected_params is not None, input_directory=str(args.input.resolve()),
        versions={p: importlib.metadata.version(p) for p in ['scikit-learn', 'numpy', 'pandas', 'joblib']},
    )
    selected.save(args.output / 'model')
    metrics, scores, predictions = measure(selected, student_test)
    pd.DataFrame(results).to_csv(args.output / 'comparison.csv', index=False)
    write_json(args.output / 'cv_metrics.json', fold_results)
    write_json(args.output / 'configurations.json', configs)
    write_json(args.output / 'metrics.json', metrics)
    scores.to_csv(args.output / 'candidate_scores.csv', index=False)
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    print(f'Saved comparison and selected {name} model to {args.output}')


if __name__ == '__main__':
    main()
