# API examples for Person 4

## List trips

`GET /api/trips`

```json
{
  "partition": "student_test",
  "trip_ids": ["00000000", "00000007"]
}
```

## Trip metadata

`GET /api/trips/00000052`

```json
{
  "trip_id": "00000052",
  "partition": "student_test",
  "gps_observations": 262,
  "road_nodes": 1770,
  "road_arcs": 3577,
  "bounds": {
    "min_longitude": 54.0,
    "max_longitude": 54.1,
    "min_latitude": 56.0,
    "max_latitude": 56.1
  }
}
```

The numeric values above are illustrative schema examples; use the API response for actual trip values.

## Prediction

`POST /api/predict/00000052`

The response includes `predicted_arc_ids`, `classifier_only_arc_ids`, `route_coordinates`, `matches`, `route_continuity` and `processing_time_ms`. The example below is illustrative and is not a measured PathSnap result.

```json
{
  "trip_id": "00000052",
  "partition": "student_test",
  "prediction_mode": "development_fallback",
  "predictor": "distance-only-development-fallback",
  "gps_observations": 3,
  "matched_observations": 3,
  "missing_observations": 0,
  "predicted_arc_ids": [12, 12, 15],
  "classifier_only_arc_ids": [12, 14, 15],
  "route_continuity": 1.0,
  "classifier_only_route_continuity": 0.5,
  "route_coordinates": [[54.0, 56.0], [54.001, 56.001]],
  "matches": [
    {"obs_index": 0, "arc_id": 12, "candidate_score": 0.8, "distance_m": 4.2}
  ],
  "viterbi_log_probability": -1.2,
  "processing_time_ms": 25.0,
  "coordinate_order": "[longitude, latitude]"
}
```

## Evaluation

`GET /api/evaluate/00000052`

Returns sequence and arc-set metrics comparing the reconstructed route against the server-side reference `.route` sequence. It does not return reference labels.

## Updated health response

Development fallback mode is explicitly marked as degraded:

```json
{
  "status": "degraded",
  "partition": "student_test",
  "model_loaded": false,
  "predictor": "distance-only-development-fallback",
  "predictor_mode": "development_fallback",
  "candidate_provider": "NearestArcCandidateProvider",
  "data_root_exists": true,
  "dataset_available": true,
  "hmm_enabled": true,
  "prediction_mode": "development_fallback"
}
```

The values above are an illustrative response. Your actual health response
reflects the current configuration.

## Prediction diagnostics

Prediction responses include `decoder_diagnostics` and `timing_breakdown_ms`.
The latter separates candidate generation, model scoring, HMM decoding and total
processing time. All timings are measured at runtime; do not copy example
numbers into the report.
