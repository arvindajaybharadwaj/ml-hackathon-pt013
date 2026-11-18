# Person 3 Implementation Status

## Implemented

- HMM with variable candidate counts and explicit missing states.
- Log-space Viterbi decoding and backtracking.
- Emission-score normalisation and configurable transition parameters.
- Directed road-distance transitions using bounded A* with an admissible straight-line heuristic.
- Handling and diagnostics for missing candidates, invalid candidate indices, unreachable transitions and long GPS gaps.
- FastAPI endpoints for health, trip listing, trip metadata, prediction and separate evaluation.
- Per-stage timings for candidate generation, model scoring, HMM decoding and total processing.
- API request logging including the remote client IP.
- Development fallback clearly identified in health and prediction responses.
- Production startup safeguards that reject fallback mode and require prepared candidate files and a trained model.
- HMM grid-search, decoder comparison and all-trip evaluation scripts.
- Unit/API/production-configuration tests and teammate integration documentation.
- Two-laptop LAN deployment guide.

## Validation completed on 2026-11-18

- `pytest -q`: **7 passed**.
- `python -m compileall -q backend scripts`: passed.
- Real-data smoke prediction/evaluation completed for student-test trips `00000052` and `00000056`.
- A quick eight-configuration HMM tuning smoke grid completed on one student-test trip.
- Raw outputs and limitations are documented in `results/VALIDATION_NOTES.md`.

All measured real-data results currently use `NearestArcCandidateProvider` and
`DistanceFallbackPredictor`. These are development fallbacks, not the final
Person 1 candidate pipeline or Person 2 trained model. Do not present these
numbers as final ML model results.

## Still dependent on teammates / organisers

1. Person 1: final prepared candidates/provider and confirmation of observation/arc indexing.
2. Person 2: final trained model, preprocessing artifact, feature order and probability/class semantics.
3. Organisers: authorised evaluator partition for final demo testing.
4. Person 4: frontend integration, LAN connection and UI verification.

## Metric definition and label separation

The `.route` file is a trip-level arc sequence, not an observation-aligned
label sequence. Evaluation therefore reports sequence/set metrics, not GPS
pointwise accuracy. `route_continuity` measures direct arc adjacency (same arc
or previous arc end node equals current arc start node). HMM transitions can
still consider multi-hop directed paths.

Prediction reads only `.track`, `.nodes` and `.arcs`. The separate evaluation
service reads `.route` only after prediction completes. Reference labels are not
included in trip metadata or prediction responses.

## Not yet complete

- Integration with the actual Person 1 candidate output.
- Integration with Person 2's actual trained model.
- Full ten-trip final evaluation and evaluator partition testing.
- Laptop 2 to Laptop 1 LAN acceptance test.

The backend is ready for teammate handoff and further integration, but final
project acceptance must wait for these external artifacts and tests.
