from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from experiment_utils import (ROOT, arc_set_scores, compact, create_services,
                              sequence_similarity, settings_from_args)
from backend.app.hmm import HMMConfig, HMMViterbiDecoder
from backend.app.network import RoadNetwork
from backend.app.data import ReferenceRouteRepository


BASE_SIGMA = [15.0, 30.0, 50.0, 75.0]
SIGMA_FRACTION = [0.25, 0.5, 0.75]
MISSING_PENALTY = [3.0, 6.0, 9.0]
UNREACHABLE_PENALTY = [8.0, 12.0, 16.0]


def main():
    parser = argparse.ArgumentParser(description="Grid-search HMM transition parameters on a permitted development partition.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--partition", default="student_test")
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke grid (not a final tuning run).")
    parser.add_argument('--compact-grid', action='store_true', help='Twelve justified settings; use all development trips')
    parser.add_argument("--trip-limit", type=int, default=None, help="Limit number of trips for a quick smoke test.")
    parser.add_argument("--output", default=str(ROOT / "results" / "hmm_experiments.csv"))
    args = parser.parse_args()

    settings = settings_from_args(args.data_root, args.partition)
    trips, provider, predictor, _, evaluation = create_services(settings)
    trip_inputs = []
    trip_ids = trips.list_trip_ids()
    if args.trip_limit is not None:
        trip_ids = trip_ids[:max(1, args.trip_limit)]
    for trip_id in trip_ids:
        trip = trips.load_trip(trip_id)
        network = RoadNetwork(trip)
        candidates = provider.generate(trip, network)
        candidates = candidates.copy()
        candidates["model_score"] = predictor.score_candidates(candidates)
        reference = trips.partition_dir / trip_id / f"{trip_id}.route"
        # Reference labels are loaded only in this experiment/evaluation script, never by inference.
        ref_values = pd.to_numeric(pd.read_csv(reference, sep=r"\s+", header=None).iloc[:, 0], errors="coerce").dropna().astype(int).tolist()
        labels = ReferenceRouteRepository(trips, settings.candidate_dir).load_aligned_labels(trip_id).sort_values('obs_id')
        mask = labels.alignment_valid & labels.reference_arc_id.notna()
        truth = labels.loc[mask, 'reference_arc_id'].astype(int).to_numpy()
        trip_inputs.append((trip_id, trip, network, candidates, compact(ref_values), mask.to_numpy(), truth))
        print(f"Prepared {trip_id} for decoder experiments")

    if not trip_inputs:
        raise SystemExit(f"No trips found in {trips.partition_dir}")

    values = ([30.0, 50.0], [0.25, 0.5], [6.0], [8.0, 12.0]) if args.quick else (
        BASE_SIGMA, SIGMA_FRACTION, MISSING_PENALTY, UNREACHABLE_PENALTY)
    if args.compact_grid:
        values = ([15., 30., 50.], [.25, .5], [6.], [8., 12.])
    configs = list(itertools.product(*values))
    rows = []
    for index, (base, fraction, missing, unreachable) in enumerate(configs, 1):
        config = HMMConfig(
            transition_sigma_base_m=base,
            transition_sigma_fraction=fraction,
            missing_state_penalty=missing,
            unreachable_penalty=unreachable,
        )
        decoder = HMMViterbiDecoder(config)
        per_trip = []
        correct, evaluated = 0, 0
        for trip_id, trip, network, candidates, reference, mask, truth in trip_inputs:
            decoded = decoder.decode(trip.track, candidates, network)
            predicted = compact(decoded.arc_ids)
            similarity = sequence_similarity(predicted, reference)
            precision, recall, f1 = arc_set_scores(predicted, reference)
            per_trip.append((similarity, f1, precision, recall, decoded.diagnostics))
            import numpy as np
            pointwise = np.array([-1 if x is None else x for x in decoded.arc_ids])[mask]
            correct += int((pointwise == truth).sum())
            evaluated += len(truth)
        rows.append({
            **asdict(config),
            "mean_sequence_similarity": sum(x[0] for x in per_trip) / len(per_trip),
            "mean_arc_set_f1": sum(x[1] for x in per_trip) / len(per_trip),
            "mean_arc_set_precision": sum(x[2] for x in per_trip) / len(per_trip),
            "mean_arc_set_recall": sum(x[3] for x in per_trip) / len(per_trip),
            "trip_count": len(per_trip),
            "predictor_mode": predictor.mode,
            "predictor": predictor.name,
            'aligned_top1_accuracy': correct / evaluated if evaluated else 0.,
            'evaluated_observations': evaluated,
        })
        print(f"[{index}/{len(configs)}] sequence similarity={rows[-1]['mean_sequence_similarity']:.4f}, F1={rows[-1]['mean_arc_set_f1']:.4f}")

    frame = pd.DataFrame(rows).sort_values(
        ['aligned_top1_accuracy', "mean_sequence_similarity", "mean_arc_set_f1"], ascending=False
    ).reset_index(drop=True)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    best = frame.iloc[0].to_dict()
    best_config = {k: float(best[k]) for k in (
        "transition_sigma_base_m", "transition_sigma_fraction", "missing_state_penalty", "unreachable_penalty")}
    summary = {
        "partition": args.partition,
        "grid_size": len(configs),
        "trip_count": len(trip_inputs),
        "predictor_mode": predictor.mode,
        "predictor": predictor.name,
        "best_config": best_config,
        'selection_objective': 'Pooled aligned observation Top-1; ties use observation-arc sequence similarity and set F1',
        'best_aligned_top1_accuracy': float(best['aligned_top1_accuracy']),
        'trip_ids': trip_ids,
        'grid_rationale': 'Vary transition tolerance and unreachable penalty; missing penalty held fixed because these student-test trips have no candidate-free observations.',
        "best_mean_sequence_similarity": float(best["mean_sequence_similarity"]),
        "best_mean_arc_set_f1": float(best["mean_arc_set_f1"]),
        "warning": "Exploratory tuning on this partition is not an independent final evaluation. Do not tune against evaluator-test labels.",
    }
    summary_path = output.with_name(output.stem + "_best.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved grid: {output}")
    print(f"Saved best configuration: {summary_path}")


if __name__ == "__main__":
    main()
