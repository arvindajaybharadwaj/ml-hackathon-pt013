# Person 3 implementation status

## Implemented in this package

- Directed road-network representation with metric arc lengths.
- Candidate-state HMM and log-space Viterbi decoding.
- Emission scores from the candidate predictor.
- Directed transition distances with shortest-path caching, GPS displacement/time-gap handling, and penalties for unreachable transitions.
- Explicit missing-candidate states.
- Classifier-only baseline output for comparison with HMM output.
- Route continuity and sequence/set evaluation metrics.
- FastAPI health, trip listing, metadata, prediction, and separate evaluation endpoints.
- Separate inference and reference-label repositories/services.
- Person 1 candidate-table adapter and Person 2 joblib model adapter.
- Development-only nearest-arc and distance-score fallbacks.
- Setup instructions, Swagger API, integration guide, and tests.

## Validation performed in this environment

- Python compile check: passed.
- Automated tests: 3 passed.
- All 80 training trips: required files parsed and reference arc IDs validated against their arc tables.
- All 10 student-testing trips: required files parsed and reference arc IDs validated against their arc tables.
- Real-data smoke test: trip `00000052` in `student_test` completed prediction and separate evaluation using the development fallback.
- FastAPI TestClient check on the real uploaded trip: health, trip listing, trip metadata, prediction and evaluation endpoints returned HTTP 200.

## Still required from teammates / organisers

1. Person 1's final candidate and feature-generation output. The package defines the accepted CSV/Parquet schema; until it is connected, nearest-arc generation is only a development fallback.
2. Person 2's trained model, feature order and class/probability semantics. Until it is connected, the distance-only scorer is only a development fallback.
3. The reserved evaluator partition. It is not present in the uploaded dataset ZIP, which contains only `train/` and `student_test/`.
4. Final end-to-end test on both laptops after Persons 1, 2 and 4 connect their components.

## Important academic-integrity note

Do not present the distance fallback's outputs or metrics as the team's selected ML model results. Final results must be generated again with Person 1's actual prepared candidates and Person 2's actual trained predictor. The evaluator partition must remain untouched until final evaluation. The supplied `.route` file is read only by the separate evaluation service and never by prediction inference.
