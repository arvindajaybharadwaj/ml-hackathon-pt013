# EDA report handoff

Use data_notes.md and verified_statistics.json as the current evidence.

| Partition | Trips | GPS observations | Candidate rows | Valid aligned observations | Alignment coverage | Zero candidate observations |
|---|---:|---:|---:|---:|---:|---:|
| train | 80 | 208,417 | 1,474,434 | 177,355 | 85.0962% | 18,232 |
| student_test | 10 | 19,160 | 164,457 | 17,821 | 93.0115% | 0 |

## Raw data and sampling

The four headerless files per trip are track (longitude, latitude, seconds), nodes (longitude, latitude), arcs (directed node IDs), and route (chronological reference arc IDs). Node and arc identifiers are zero-based row positions local to a trip. Reference route rows are not observation labels.

Across all intervals, 96.6820% are approximately one second (numpy.isclose). The longest training gap is 153,064 seconds and the longest student-test gap is 210 seconds. Parsed raw GPS values are finite and exported numeric candidate features have zero missing values. Invalid reference alignments are intentionally nullable labels, not feature missing values. Trips span multiple regions; use the per-partition coordinate ranges in verified_statistics.json rather than assuming all trips are in Darmstadt.

## Candidate features

The pipeline uses a trip-local equirectangular metric approximation and a Shapely STRtree. It searches within 50 m and retains up to 15 nearby directed arcs per observation. There is no universal verified 0.01% distortion guarantee. Feature names, units and order are in feature_schema.json; nominal kinematic ranges are descriptive, not enforced clipping guarantees. The original training schema remains in evidence/models/records/feature_schema_at_training.json.

The 13 features are distance, projection fraction, segment length, heading difference, heading cosine, GPS speed, time gap, spatial gap, distance rank, distance margin, step-to-length ratio, zero-speed flag and reliability flag. The first observation copies the first available motion step. Stationary observations use a neutral heading; observations with no nearby road remain present.

## Reference alignment and leakage

Reference alignment is separate from inference candidate generation. It searches a forward route window, admits a one-step backward jitter, expands the window for time gaps and accepts associations within 35 m. Labels may be invalid when observations are far from the reference or beyond its extent. Treat these derived associations as uncertain reference information, not independently verified physical ground truth. Report actual alignment coverage beside every pointwise metric.

All training transformations fit on training trips only. Evaluator labels must never be used to tune the classifier or decoder. Live prediction and map rendering do not load reference routes. A separate offline student-data report map includes the reference layer.

## Reproduction and figures

Run scripts/verify_data_statistics.py to recompute raw sampling statistics and plots. Its JSON output is evidence/data/verified_statistics.json. Candidate count, distance, network-scale and speed plots are also retained in evidence/data. The report and its statistics are based on executed exports, not earlier narrative estimates.
