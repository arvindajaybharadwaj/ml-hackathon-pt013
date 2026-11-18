from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiment_utils import ROOT


def main():
    parser = argparse.ArgumentParser(description="Create classifier-only vs classifier + HMM comparison from evaluation results.")
    parser.add_argument("--input", default=str(ROOT / "results" / "decoder_evaluation.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "decoder_comparison.csv"))
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Evaluation CSV not found: {source}. Run evaluate_all_trips.py first.")
    frame = pd.read_csv(source)
    if "status" in frame.columns:
        frame = frame[frame.status == "ok"].copy()
    required = {"trip_id", "sequence_edit_similarity", "classifier_only_sequence_edit_similarity",
                "arc_set_f1", "classifier_only_arc_set_f1", "route_continuity",
                "classifier_only_route_continuity", "time_total_ms", "time_candidate_generation_ms",
                "time_model_scoring_ms", "time_hmm_decoding_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Evaluation CSV missing columns: {sorted(missing)}")

    rows = []
    for _, row in frame.iterrows():
        rows.extend([
            {"trip_id": row.trip_id, "decoder": "classifier_only",
             "sequence_similarity": row.classifier_only_sequence_edit_similarity,
             "arc_set_f1": row.classifier_only_arc_set_f1,
             "route_continuity": row.classifier_only_route_continuity,
             "inference_time_ms": row.time_candidate_generation_ms + row.time_model_scoring_ms},
            {"trip_id": row.trip_id, "decoder": "classifier_plus_hmm",
             "sequence_similarity": row.sequence_edit_similarity,
             "arc_set_f1": row.arc_set_f1,
             "route_continuity": row.route_continuity,
             "inference_time_ms": row.time_total_ms},
        ])
    comparison = pd.DataFrame(rows)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    metric_columns = [column for column in comparison.select_dtypes(include="number").columns
                      if column != "trip_id"]
    summary = comparison.groupby("decoder")[metric_columns].mean().reset_index()
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved comparison: {output}")
    print(f"Saved aggregate means: {summary_path}")
    print(summary.to_string(index=False))
    print("Classifier-only time excludes HMM decoding; HMM time includes candidate generation, model scoring and decoding.")


if __name__ == "__main__":
    main()
