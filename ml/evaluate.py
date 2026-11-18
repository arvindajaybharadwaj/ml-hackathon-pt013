"""Evaluate a saved model on all student-test trips without refitting."""
import argparse
from pathlib import Path
import json
from .data import load_dataset
from .predictor import load_model
from .train import measure, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('prepared_data'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory must be empty')
    dataset = load_dataset(args.input).partition('student_test')
    predictor = load_model(args.model)
    metrics, scores, predictions = measure(predictor, dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'metrics.json', metrics)
    scores.to_parquet(args.output / 'candidate_scores.parquet', index=False)
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
