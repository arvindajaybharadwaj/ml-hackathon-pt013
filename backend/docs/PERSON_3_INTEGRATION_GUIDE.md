# Person 3 – Integration Guide

## Person 1 → Person 3: prepared candidate table

Preferred file format: one CSV or Parquet file per trip named `<trip_id>.csv` or `<trip_id>.parquet`.

Required columns:

- `obs_index`: zero-based index into the chronologically ordered observations.
- `arc_id`: zero-based row index into that trip's `.arcs` file.
- `distance_m`: perpendicular GPS-to-segment distance in metres.
- `fraction`: position along the directed arc in `[0, 1]`.

Keep additional feature columns required by Person 2. Do not drop/reorder candidate rows independently of their observation/arc IDs. Provide the feature names and units. `obs_index` must match `TripRepository` ordering (timestamps are stable-sorted).

Set `PATHSNAP_CANDIDATE_DIR` to the folder containing those files. If you prefer to expose a Python class instead, implement `generate(trip, network) -> pandas.DataFrame` matching the same schema.

## Person 2 → Person 3: trained predictor

Required handoff:

- Saved predictor artifact.
- Exact ordered feature list and preprocessing requirements.
- Output/class semantics.
- Confirmation whether scores are binary match probabilities or probabilities over arc-ID classes.
- Positive-class label for a binary classifier.

The preferred adapter contract is:

```python
scores = predictor.score_candidates(candidate_table)
# one finite score per input candidate row, same order
```

If using scikit-learn/joblib, place `model_features.json` beside the artifact or set `PATHSNAP_MODEL_FEATURES`. Configure `PATHSNAP_MODEL_MODE` as `binary`, `arc_multiclass`, or adapt `JoblibCandidatePredictor` for the real model.

## Person 4 → Person 3: frontend contract

Base URL: `http://<Laptop-1-LAN-IP>:8000`

- `GET /health`
- `GET /api/trips`
- `GET /api/trips/{trip_id}`
- `POST /api/predict/{trip_id}`
- `GET /api/evaluate/{trip_id}`

Prediction geometry uses `[longitude, latitude]` coordinate pairs. `predicted_arc_ids` is a sequence with one entry per GPS observation; `null` means no candidate was available for that observation. `route_coordinates` is a road-edge polyline approximation built from those selected arcs.

The frontend must not request or display the `.route` ground-truth file. Evaluation output contains metrics only.

Swagger docs are available at `/docs` while the API is running.

## Final evaluator dataset

The supplied student ZIP contains `train/` and `student_test/`, but no `evaluator_test/`. At final evaluation, set `PATHSNAP_DATA_ROOT` to a data root containing the authorised evaluator partition and `PATHSNAP_PARTITION` to its exact directory name. Keep that partition untouched during development.
