from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiment_utils import ROOT, create_services, settings_from_args


def _evaluate_one(data_root: str | None, partition: str, trip_id: str) -> dict:
    settings = settings_from_args(data_root, partition)
    trips, _, predictor, _, evaluation = create_services(settings)
    result = evaluation.evaluate(trip_id)
    timing = result.pop("timing_breakdown_ms")
    diagnostics = result.pop("decoder_diagnostics")
    return {**result, "status": "ok", **{f"time_{k}_ms": v for k, v in timing.items()},
            "missing_observations": diagnostics.get("candidate_counts", []).count(0),
            "unreachable_transitions": diagnostics.get("unreachable_transitions", 0),
            "long_gaps": diagnostics.get("long_gaps", 0),
            "predictor": predictor.name, "prediction_mode": predictor.mode}


def main():
    parser = argparse.ArgumentParser(description="Evaluate every trip in the configured development partition.")
    parser.add_argument("--data-root", default=None, help="Folder containing train/ and student_test/.")
    parser.add_argument("--partition", default="student_test")
    parser.add_argument("--output", default=str(ROOT / "results" / "decoder_evaluation.csv"))
    parser.add_argument("--single-trip", help=argparse.SUPPRESS)
    parser.add_argument("--trip-ids", nargs="*", default=None, help="Optionally evaluate only these trip IDs.")
    parser.add_argument("--timeout-per-trip", type=int, default=90, help="Child-process timeout in seconds.")
    args = parser.parse_args()

    if args.single_trip:
        print(json.dumps(_evaluate_one(args.data_root, args.partition, args.single_trip)))
        return

    settings = settings_from_args(args.data_root, args.partition)
    trips, _, predictor, _, _ = create_services(settings)
    trip_ids = args.trip_ids if args.trip_ids else trips.list_trip_ids()
    if not trip_ids:
        raise SystemExit(f"No trips found in {trips.partition_dir}. Check PATHSNAP_DATA_ROOT.")

    rows = []
    # Each trip runs in a short-lived process. This keeps large Shapely/NetworkX
    # graph allocations isolated when evaluating the whole partition.
    for index, trip_id in enumerate(trip_ids, 1):
        command = [sys.executable, str(Path(__file__).resolve()), "--partition", args.partition,
                   "--single-trip", trip_id]
        if args.data_root:
            command.extend(["--data-root", args.data_root])
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True,
                                       timeout=args.timeout_per_trip)
            row = json.loads(completed.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(f"[{index}/{len(trip_ids)}] {trip_id}: HMM sequence similarity={row['sequence_edit_similarity']:.4f}")
        except subprocess.TimeoutExpired:
            rows.append({"trip_id": trip_id, "status": "timeout", "error": f"Exceeded {args.timeout_per_trip}s"})
            print(f"[{index}/{len(trip_ids)}] {trip_id}: timed out; recorded as incomplete")
        except subprocess.CalledProcessError as exc:
            rows.append({"trip_id": trip_id, "status": "error", "error": exc.stderr[-1000:] if exc.stderr else "Child process failed"})
            print(f"[{index}/{len(trip_ids)}] {trip_id}: failed; see CSV error field")

    frame = pd.DataFrame(rows)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    successful = frame[frame.get("status", "ok") == "ok"] if "status" in frame.columns else frame
    numeric = successful.select_dtypes(include="number").mean(numeric_only=True).to_dict()
    summary = {
        "partition": args.partition,
        "trip_count": len(rows),
        "successful_trip_count": len(successful),
        "failed_trip_count": len(rows) - len(successful),
        "predictor": predictor.name,
        "prediction_mode": predictor.mode,
        "mean_metrics": numeric,
        "note": "Actual run output. Reference labels are loaded only after prediction in the evaluation service. This is development-partition evidence, not evaluator-set performance.",
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved per-trip results: {output}")
    print(f"Saved aggregate summary: {summary_path}")
    print(f"Mode: {predictor.mode} | Predictor: {predictor.name}")


if __name__ == "__main__":
    main()
