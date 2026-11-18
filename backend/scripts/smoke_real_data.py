"""Run a local, real-data smoke test against a supplied dataset partition."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import Settings
from backend.app.main import build_services


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Folder containing train/ and student_test/")
    parser.add_argument("--partition", default="student_test")
    parser.add_argument("--trip-id", default="00000052")
    args = parser.parse_args()
    settings = Settings.from_env()
    settings = Settings(**{**settings.__dict__, "data_root": Path(args.data_root).resolve(), "partition": args.partition})
    trips, candidate_provider, predictor, prediction, evaluation = build_services(settings)
    result = prediction.predict(args.trip_id)
    print("PREDICTION SUMMARY")
    print(json.dumps({k: result[k] for k in (
        "trip_id", "prediction_mode", "gps_observations", "matched_observations",
        "route_continuity", "classifier_only_route_continuity", "processing_time_ms"
    )}, indent=2))
    try:
        print("EVALUATION SUMMARY")
        print(json.dumps(evaluation.evaluate(args.trip_id), indent=2))
    except FileNotFoundError as exc:
        print(f"Evaluation skipped: {exc}")


if __name__ == "__main__":
    main()
