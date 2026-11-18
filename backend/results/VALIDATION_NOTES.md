# Person 3 Validation Notes

Date: 2026-11-18

## Automated tests

Command: `pytest -q`

Result: **7 passed**.

The tests cover basic Viterbi transition coherence, missing observations and gap
handling, malformed candidate indices, FastAPI smoke behaviour, invalid/unknown
trip responses, health diagnostics and production-mode fallback rejection.

## Real supplied student-test data

The uploaded `Student folder.zip` was extracted and the backend was run in
`development_fallback` mode. Two actual trips were smoke-tested:

| Trip | GPS observations | HMM sequence edit similarity | Classifier-only sequence similarity | HMM arc-set F1 | Classifier-only arc-set F1 | HMM direct continuity | Classifier-only direct continuity |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00000052 | 262 | 0.873684 | 0.500000 | 0.932584 | 0.720812 | 0.965517 | 0.773946 |
| 00000056 | 3,958 | 0.801487 | 0.465387 | 0.929658 | 0.734479 | 0.964114 | 0.763457 |

Raw per-trip JSON outputs are stored in:

- `real_data_smoke_00000052.json`
- `real_data_smoke_00000056.json`

`api_smoke_00000052.txt` records prediction and separate evaluation output,
including measured timing breakdown and decoder diagnostics.

## HMM tuning smoke test

Command:

```bash
python scripts/tune_hmm.py --data-root "/path/to/Student folder" --partition student_test --quick --trip-limit 1
```

The quick smoke grid ran eight parameter configurations on one development
trip. Results are in `hmm_experiments.csv`; the top row is recorded in
`hmm_experiments_best.json`. This is only a script/experiment-pipeline smoke
test, not a selected final HMM configuration.

## Critical interpretation

All real-data numbers above were generated with `DistanceFallbackPredictor`
and `NearestArcCandidateProvider`. These are development-only substitutes,
not Person 2's trained ML predictor or Person 1's final candidate package.
Therefore the observed improvement demonstrates that the API/HMM experiment
path executes; it does **not** establish final project model performance.

The full ten-trip evaluation script is included as
`scripts/evaluate_all_trips.py`, but a complete ten-trip batch was not
successfully completed in this validation session. Run it again after the
teammate adapters are integrated and retain the resulting CSV/summary.

## Not yet verified

- Person 1's real candidate table/provider integration.
- Person 2's real trained model integration.
- Full evaluator partition run.
- Laptop 2 to Laptop 1 LAN acceptance test.
