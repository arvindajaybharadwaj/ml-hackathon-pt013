# Backend Architecture – Person 3

## Request flow

`Laptop 2 UI -> FastAPI on Laptop 1 -> TripRepository -> CandidateProvider -> Person 2 Predictor -> HMM/Viterbi -> JSON response`.

The prediction service loads only `.track`, `.nodes`, and `.arcs`. It does not
read `.route`. The evaluation service calls prediction first and only then loads
the reference `.route` file to calculate metrics. Reference labels are never
returned by trip listing, trip metadata or prediction endpoints.

## Integration boundaries

### Person 1: prepared candidate provider

Preferred handoff is one CSV or Parquet file per trip in
`PATHSNAP_CANDIDATE_DIR`. Required columns are:

- `obs_index`: zero-based row in the chronologically sorted track.
- `arc_id`: zero-based row in the original `.arcs` file.
- `distance_m`: perpendicular GPS-to-segment distance in metres.
- `fraction`: projection fraction along the directed arc, in `[0, 1]`.

All extra feature columns are preserved for Person 2. Candidate rows are checked
for valid observation and arc indices.

### Person 2: trained predictor

The joblib adapter supports `binary`, `arc_multiclass`, and `direct` modes.
The model feature names/order must be provided through
`PATHSNAP_MODEL_FEATURES` or `model_features.json`. If Person 2 uses a custom
artifact format, implement a small adapter exposing
`score_candidates(candidates) -> one score per candidate row`.

## Runtime modes

- `development`: permits nearest-arc candidate generation and distance-only
  scoring for local API smoke tests. `/health` reports `degraded` and
  `development_fallback`.
- `production`: requires Person 1's prepared candidate directory, a real
  Person 2 model, an existing configured partition, and candidate files for
  every trip. It refuses to start with the distance fallback.

Do not present development fallback metrics as the final ML model results.
