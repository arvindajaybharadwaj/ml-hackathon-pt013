# PathSnap handoff — 18 November 2026

## Current status

Training and frontend/backend integration are complete. Implementation changes are uncommitted; preserve them. No commit/push was requested.

## Training results

- Detached Modal training completed successfully: app `ap-B97lbex6EUmp9AyL8DNlkH`, run `modal-final-20261118`, profile `arvindajaybharadwaj19`.
- All six families trained on 80 real training trips, 3 whole-trip folds; 20 Bayesian tuning trials; 10 separate student-test trips.
- Selected **tuned histogram gradient boosting**. CV candidate Macro F1 0.880768 (baseline 0.880544). Student-test candidate Macro F1 0.894405; observation Top-1 0.823859 on 17,821 valid aligned observations. Alignment coverage 0.930115 across 19,160 test observations.
- Actual training duration: 866 seconds / 14.4 minutes. Neural candidate model trained using CUDA on Modal; local MPS training was also tested successfully. Classical models use CPU.
- All experiment results downloaded to `runs/modal-final-20261118/`; independently evaluated with `ml.evaluate`, matching saved metrics.
- Selected portable bundle is copied to **`artifacts/model/`** with model.joblib, schema, metadata, training summary and README. This small bundle is sufficient for the backend; full runs/ is gitignored.
- Original archive remains in Modal Volume `pathsnap-training-results`, `modal-final-20261118.tar.gz`.
- Old attached apps were stopped to prevent duplicate charges. Do not restart them.

## Integration implemented

- ML loader supports real partitioned Parquet exports and feature_order schema; no synthetic training data.
- Backend `PreparedCandidateProvider` reads the combined partition table, preserving real ML features and adding decoder aliases (obs_index, distance_m, fraction). It validates schema version and trip identity.
- `BundleCandidatePredictor` uses the existing versioned ML loader and candidate inference interface.
- Backend defaults to production, real packaged model, actual prepared student-test data, and refuses development fallbacks.
- HMM/Viterbi decodes scored candidates. Last four predictions are cached for subsequent evaluation/map requests.
- New `/api/maps/{trip_id}` returns a Folium map with raw GPS, nearby road network, classifier selections and HMM matched road segments. It never loads reference routes and does not draw invented connections between disconnected road segments.
- React fetches health and actual ten trip IDs, submits predictions, shows loading/errors/results, embeds map, compares classifier/HMM route metrics. No placeholder evaluator IDs.
- Same-origin Vite proxy serves API/map requests; `VITE_BACKEND_TARGET` supports backend on another laptop. `VITE_API_URL` supports separately hosted builds with configured CORS.
- Root run_backend.sh uses the shared .venv and proper imports. Root pytest.ini discovers data/ML + backend + integration tests.
- Root requirements.txt includes backend + ML + Folium. Dependencies installed locally; frontend npm ci completed.
- Root/backend/frontend README and .env.example files updated.

## Verification

- 30 Python tests passed, including real packaged-model API predictions, metric responses, map response, schema mismatch rejection, score parity and proving inference/maps cannot access reference labels.
- Frontend `npm run lint` and `npm run build` passed.
- Live browser successfully selected trip 00000052, submitted prediction, rendered Folium layers and displayed real route metrics.
- **All ten student-test trips passed live HTTP checks through the frontend proxy**, including prediction, evaluation and map rendering. Evidence: `evidence/integration/api_checks.json` (complete=true). Individual checks took ~0.4–6.5 seconds.
- Example trip 00000052: direct continuity classifier 93.1%, HMM 96.6%; route sequence similarity classifier 71.6%, HMM 83.2%. These are route metrics, not pointwise accuracy.
- Existing third-party NumPy/joblib and pytest/TestClient deprecation warnings are non-failing.

## Run commands (repository root)

Install if setting up a new environment (Python 3.12):
```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm --prefix frontend ci
```

Backend terminal:
```bash
bash run_backend.sh
```

Frontend terminal:
```bash
npm --prefix frontend run dev -- --port 5173
```

Open http://localhost:5173 (API docs http://localhost:8000/docs).

Tests and frontend checks:
```bash
.venv/bin/python -m pytest -q
npm --prefix frontend run lint
npm --prefix frontend run build
```

Full live check (servers must be running):
```bash
.venv/bin/python scripts/check_integration.py
```

Saved-model evaluation (choose a new empty output directory):
```bash
.venv/bin/python -m ml.evaluate --input prepared_data --model artifacts/model --output runs/final-independent-test
```

For a fresh full training run (unnecessary for current integration):
```bash
.venv/bin/python -m modal run ml/modal_train.py --trials 20 --epochs 15
```

## Practical limits

- Only the supplied ten student-test trips are currently configured. Organiser evaluator inputs/candidates must be supplied and environment paths changed before final evaluator deployment.
- Folium browser assets and OpenStreetMap tiles require internet. Prediction/model loading work locally.
- Map shows actual selected road segments. The decoder's selected observation arcs are not expanded into every intermediate shortest-path road edge.
- Existing legacy backend docs/results describe earlier fallback-only smoke work; root README and evidence/integration are authoritative for current integration.
- Servers were started for verification on ports 8000 and 5173. Check for existing processes before starting duplicates.
