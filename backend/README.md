# PathSnap – Person 3 Backend Package

This package implements the **Person 3** work allocation:

- HMM route reconstruction and Viterbi decoding.
- Adapter points for Person 1's prepared candidates/features and Person 2's trained model.
- FastAPI prediction backend for Laptop 1.
- Separate server-side evaluation against the reference `.route` file.
- Unit tests, integration notes and API documentation.

It does **not** retrain the five ML models, own Person 1's production candidate-generation pipeline, build Person 4's UI, or write the team's final report.

## Important status distinction

The package includes two development fallbacks so it can be run before the other teammates hand over their artifacts:

1. `NearestArcCandidateProvider` finds nearby road segments directly from the supplied road graph. This is a **temporary development adapter**, not a replacement for Person 1's final candidate/feature pipeline.
2. `DistanceFallbackPredictor` converts candidate-to-GPS distance into a score. This is **not a trained ML model** and is not a substitute for Person 2's selected model.

The API health response labels this mode `development_fallback`. Do not report its metrics as the team's final trained-model results. Switch to Person 1/2's real outputs before the final demonstration.

## 1. Dataset setup

Extract the supplied `Student folder.zip`. The data root must contain the partition directories:

```text
Student folder/
├── train/
│   └── 00000001/
│       ├── 00000001.track
│       ├── 00000001.nodes
│       ├── 00000001.arcs
│       └── 00000001.route
└── student_test/
    └── 00000000/
        ├── 00000000.track
        ├── 00000000.nodes
        ├── 00000000.arcs
        └── 00000000.route
```

The supplied ZIP includes 80 training trips and 10 student-testing trips. It does **not** include the reserved evaluator partition. The `.route` file is not loaded by the prediction service; it is opened only by the separate evaluation service.

## 2. Install and run

Python 3.10+ is recommended.

```bash
cd pathsnap_person3
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `PATHSNAP_DATA_ROOT` to the extracted `Student folder` directory. Example:

```env
PATHSNAP_DATA_ROOT="/Users/yourname/Downloads/Student folder"
PATHSNAP_PARTITION=student_test
```

Start the API:

```bash
set -a && source .env && set +a  # macOS/Linux
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Or use `./run_backend.sh` (it creates a virtual environment, installs dependencies, sources `.env`, then starts the server).

Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs` (interactive Swagger documentation)

## 3. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend, data, candidate-provider and predictor status |
| GET | `/api/trips` | Trip IDs in the configured partition |
| GET | `/api/trips/{trip_id}` | Trip counts and geographic bounds; no reference labels |
| POST | `/api/predict/{trip_id}` | Candidate scoring + HMM/Viterbi route prediction |
| GET | `/api/evaluate/{trip_id}` | Separately evaluates prediction using server-side `.route` |

Example:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/trips
curl -X POST http://127.0.0.1:8000/api/predict/00000052
curl http://127.0.0.1:8000/api/evaluate/00000052
```

The prediction response includes `predicted_arc_ids`, `classifier_only_arc_ids`, route coordinates in `[longitude, latitude]` order, route-continuity diagnostics and processing time.

### Metric limitation

The supplied `.route` is a trip-level sequence of arc IDs. It is not an observation-aligned label array. Therefore, the current evaluator reports sequence edit similarity, LCS recall and arc-set precision/recall/F1, plus route continuity. It does **not** call these GPS pointwise accuracy. If Person 1 supplies a verified observation-to-reference-arc alignment, a pointwise metric can be added using that documented mapping.

## 4. Integrate Person 1's candidates

Preferred handoff: Person 1 exports one file per trip to `PATHSNAP_CANDIDATE_DIR`:

```text
candidate_tables/
├── 00000052.csv
├── 00000000.csv
└── ...
```

CSV or Parquet is supported. Required columns:

| Column | Meaning |
|---|---|
| `obs_index` | Zero-based row index in the chronologically ordered `.track` loaded by this package |
| `arc_id` | Zero-based row index in `.arcs` |
| `distance_m` | GPS-to-candidate road-segment distance in metres |
| `fraction` | Projection position along the directed arc, from 0 to 1 |

Additional numeric feature columns from Person 1 are preserved and can be passed to Person 2's predictor. Candidate rows must retain candidate-to-arc identity. `obs_index` must refer to the same sorted observation sequence returned by `TripRepository`.

Set:

```env
PATHSNAP_CANDIDATE_DIR=/path/to/candidate_tables
```

When this variable is set, the API uses `PreparedCandidateProvider`. Otherwise it uses the development-only nearest-arc candidate fallback.

If Person 1 already provides a Python provider instead of CSV/Parquet, adapt `CandidateProvider.generate(trip, network)` in `backend/app/candidates.py`; keep the output schema above.

## 5. Integrate Person 2's trained model

Person 2 should provide:

- The final saved model (for example, a scikit-learn `.joblib` artifact).
- Ordered feature names used by that model.
- The meaning of the model's output/classes.
- Confirmation whether its prediction is binary candidate match probability or multiclass arc-ID probability.

Create `model_features.json` beside the model, for example:

```json
["distance_m", "fraction", "segment_length_m", "heading_difference_deg"]
```

Then configure `.env`:

```env
PATHSNAP_MODEL_PATH=/path/to/final_model.joblib
PATHSNAP_MODEL_FEATURES=distance_m,fraction,segment_length_m,heading_difference_deg
PATHSNAP_MODEL_MODE=binary
PATHSNAP_POSITIVE_CLASS=1
PATHSNAP_ALLOW_DEV_FALLBACK=false
```

Supported modes:

- `binary`: each row is a GPS-observation/candidate pair; the model's positive class is a candidate match. Uses `predict_proba` where available.
- `arc_multiclass`: model classes are arc IDs; score for a candidate is the probability of that row's `arc_id` class.
- `direct`: for models with one/two probability columns or a `decision_function`/`predict` score.

This adapter is a generic scikit-learn bridge. If Person 2's model has a different interface, implement a small adapter with `score_candidates(candidates) -> one score per candidate row`. Do not silently reorder feature columns or candidate rows.

## 6. HMM design summary

- **Hidden state:** a candidate directed road arc for one GPS observation.
- **Emission:** normalized candidate score from Person 2's model. The development fallback uses a Gaussian-shaped score based on candidate distance. It does not multiply a second distance likelihood into an already distance-informed model score, avoiding an unexamined double-counting assumption.
- **Transition:** compare GPS displacement to directed road-network distance between candidate projected positions. Directed route distance includes the remaining part of the previous arc, a shortest path from its end node to the next arc's start node, and the traversed part of the current arc.
- **Decoder:** Viterbi in log space, with variable candidate counts per observation and explicit missing states.
- **Gaps:** transition tolerance increases with elapsed time. Unreachable directed transitions receive a penalty rather than being treated as valid road continuity.

The parameters are configurable through `.env`. They should be tuned on training/student-testing trips only and supported by the team's experiments.

## 7. Two-laptop setup

Run Laptop 1 with:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's LAN IP and ask Person 4 to set their frontend API base URL to:

```env
VITE_API_BASE_URL=http://LAPTOP_1_LAN_IP:8000
```

Ensure both laptops are on the same network and Laptop 1's firewall allows inbound TCP port 8000. Add the frontend origin to `PATHSNAP_CORS_ORIGINS`. Do not expose the development server directly to the public internet.

## 8. Tests

```bash
pytest -q
```

The tests use synthetic miniature networks and temporary trip folders. They validate the decoder/API contract, not the final trained model. For a real-data smoke test after configuring `.env`, use:

```bash
curl -X POST http://127.0.0.1:8000/api/predict/00000052
```

## 9. Before final handoff

- Replace the development candidate provider with Person 1's actual prepared data/provider.
- Replace the distance-only predictor with Person 2's final trained predictor.
- Configure the authorised evaluator partition on Laptop 1 when it is supplied.
- Confirm whether reference labels are available server-side in that partition. Never expose the `.route` file or labels through trip metadata, prediction inputs, or frontend responses.
- Re-run unit, integration and end-to-end tests using the actual teammate artifacts.
- Keep actual run logs and metrics; do not report fallback results as final model results.
