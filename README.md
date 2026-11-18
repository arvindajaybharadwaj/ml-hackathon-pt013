# PathSnap

Match noisy GPS observations to roads, then reconstruct the vehicle route. The merged data pipeline exports 13 features per candidate road, with labels stored separately. The ML package consumes the real prepared data: 80 training trips and 10 student-test trips. The trained classifier is integrated with the FastAPI backend, HMM/Viterbi decoder and React dashboard.

## Environment

Use Python 3.12, from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Start the integrated application

The selected Modal model is packaged in `artifacts/model/` (about 300 KB), alongside its schema, metadata and training summary. Full experiment outputs are in `runs/modal-final-20261118/`. The default backend loads it with the prepared student-test candidates, without a development fallback.

Terminal 1, from the repository root:

```bash
bash run_backend.sh
```

Terminal 2, from the repository root:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev -- --port 5173
```

Open [the dashboard](http://localhost:5173) and select one of the 10 real student-test trip IDs. The UI calls the backend, displays prediction coverage and timing, embeds the Folium map and compares classifier/HMM route metrics. [API documentation](http://localhost:8000/docs) is available while the backend is running. The map includes GPS observations, nearby roads, classifier choices and HMM matched road segments. Separate segments are drawn individually so disconnected roads are not joined with invented straight lines. Reference routes stay on the server; only evaluation metrics reach the UI.

Map tiles and Folium's browser assets require internet access. The model and inference run locally. The backend caches the four most recent trip predictions so evaluation and map requests reuse decoding work; restart the API after changing its model or data.

The packaged selected model is sufficient to run the API. To retrieve all original experiment results from Modal:

```bash
.venv/bin/python -m modal volume get pathsnap-training-results modal-final-20261118.tar.gz runs/modal-final-20261118.tar.gz
```

Create `runs/` first if necessary, then extract the trusted training archive into it:

```bash
tar -xzf runs/modal-final-20261118.tar.gz -C runs
```

For the two-laptop demo, run the API on the model laptop. On the UI laptop, set `VITE_BACKEND_TARGET=http://MODEL_LAPTOP_LAN_IP:8000` before starting Vite (or put it in `frontend/.env.local`). The frontend proxy forwards API and map requests; the model/data stay on the API laptop. To serve an authorised evaluator dataset later, configure `PATHSNAP_PARTITION`, `PATHSNAP_DATA_ROOT` and `PATHSNAP_CANDIDATE_DIR` with its real inference inputs and schema. There are no fabricated evaluator trip IDs in the UI.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The root test suite includes data/ML tests, backend decoder/API tests and real saved-model integration tests. It covers label isolation, full candidate scoring, schema mismatches, checkpoints, map responses and missing-prediction metrics. Data tests require the supplied `Student folder/` and `prepared_data/` directories. Saved-model API tests use the packaged selected model in `artifacts/model/` (and explicitly skip when it is absent).

## Local training and timing

The supplied `prepared_data/train` and `prepared_data/student_test` Parquet files work directly. The schema is read from `feature_schema.json` beside `prepared_data/` (or inside it). No synthetic data is used for training.

First run a small real-trip smoke test:

```bash
.venv/bin/python -m ml --input prepared_data --output runs/smoke --train-trip-limit 3 --folds 3 --trials 4 --models logistic_regression gaussian_nb decision_tree random_forest hist_gradient_boosting torch_mlp --device mps --epochs 2
```

This holds out complete trips, but is not a final challenge run. The student-test partition is always kept intact. Use a new output directory for each run.

Benchmark full-data fits before a lengthy search:

```bash
.venv/bin/python -m ml --input prepared_data --output runs/benchmark --benchmark --trials 20 --models logistic_regression gaussian_nb decision_tree random_forest hist_gradient_boosting torch_mlp --device mps
```

The benchmark is timing evidence, not cross-validated model selection. `runtime_estimate.json` contains a conservative upper planning estimate using the slowest measured family for tuning; the actual selected family may train much faster. It includes model fitting/evaluation and excludes environment setup, upload, and cloud startup.

For a full local experiment:

```bash
.venv/bin/python -m ml --input prepared_data --output runs/full-local --folds 3 --trials 20 --threads 4 --models logistic_regression gaussian_nb decision_tree random_forest hist_gradient_boosting torch_mlp --device mps --epochs 15
```

Five sklearn models use CPU. `torch_mlp` uses Apple MPS when requested, CUDA on supported Linux hosts, or CPU. `--device auto` selects an available accelerator for the neural model; an explicitly unavailable device raises an error. Neural checkpoints store CPU numpy weights and work on CPU backends.

## Modal training

The full 20-trial comparison can exceed ten minutes. Install the Modal client and authenticate once:

```bash
.venv/bin/python -m pip install 'modal>=1,<2'
.venv/bin/python -m modal setup
```

Run the same training code remotely:

```bash
.venv/bin/python -m modal run ml/modal_train.py --trials 20 --epochs 15
```

The job requests 8 CPU cores, 16 GiB memory and a T4 GPU. It ships only ML source, prepared feature/label tables and the schema. CPU models use CPU, the neural model uses CUDA. Results persist in the `pathsnap-training-results` Modal Volume and download into a timestamped `runs/modal-.../` directory after completion. Cloud compute is billed by Modal. If the client disconnects, results remain in the Volume; recover them with `modal volume get pathsnap-training-results RUN_ID runs/`.

## Outputs and independent evaluation

Every completed experiment saves `comparison.csv`, fold/configuration JSON, per-model checkpoints and metrics, tuning history, a data fingerprint, and the selected `model/` bundle. Selection and tuning use trip-grouped CV candidate Macro F1; student-test labels do not fit models or select hyperparameters. Missing candidates/predictions count as observation errors. Alignment coverage records that reference labels were derived from supplied routes, not independently verified GPS-level truth.

Evaluate a saved selected model without retraining (substitute your run folder):

```bash
.venv/bin/python -m ml.evaluate --input prepared_data --model runs/full-local/model --output runs/full-local-test
```

The UI/backend should load `ml.predictor.load_model`, then call `score_candidates` with the upstream `CandidateBatch` or a dictionary containing `candidates` and `schema_version`. Inference never consumes reference labels. Emission scores are candidate-normalised classifier probabilities, ready for backend sequence decoding.

## End-to-end verification

With both local servers running, exercise all ten trips through the frontend proxy:

```bash
.venv/bin/python scripts/check_integration.py
```

Results are written to `evidence/integration/api_checks.json`. This checks predictions, evaluation and map endpoints with the actual selected model, rather than a distance fallback.

Completed Modal training took 866 seconds (14.4 minutes). Trip-grouped candidate Macro F1 improved from 0.880544 to 0.880768 after tuning. Selected histogram gradient boosting student-test Top-1 accuracy is 0.823859 on 17,821 validly aligned observations, with alignment coverage 0.930115 across 19,160 observations. This is classifier accuracy; route/HMM metrics are reported separately by the backend.

## Completed comparison and decoder evidence

The dashboard now includes a Model Comparison section with the actual six-model comparison, tuned model row, CV/test metrics, log loss, scoring/training time and learning-history graphs. Reproducible source records, plots and confusion matrices are tracked in `evidence/models/`.

The selected HMM settings are stored in `artifacts/model/hmm_config.json`. They were tuned over 12 configurations on all ten permitted student-test trips. Pooled aligned observation Top-1 is 82.39% for the classifier and 85.55% after HMM decoding. Decoder results are development measurements because its tuning used the student-test labels. Full metrics and search records are in `evidence/hmm/`.

Route output includes `route_segments` and expanded `predicted_route_arc_ids`. Valid bounded directed paths fill intermediate road edges; missing or unreachable connections remain explicit breaks. The legacy `route_coordinates` field contains only the longest continuous piece. Direct continuity counts missing observation endpoints as breaks.

### Two laptop launcher

Start `bash run_backend.sh` on the API laptop. On the UI laptop, install the frontend and use the backend laptop's actual LAN IP:

```bash
npm --prefix frontend ci
node frontend/start-demo.mjs --backend http://192.168.1.10:8000
```

The UI laptop needs Node.js only, not Python/model/data files. The launcher checks the API before starting the proxy. Detailed instructions, Windows notes and reference-free reserved-partition preparation are in [docs/TWO_LAPTOP_DEMO.md](docs/TWO_LAPTOP_DEMO.md).

All ten student-test trips passed through a separate frontend process targeting the API's actual LAN address (`evidence/integration/lan_proxy_checks.json`). A two-physical-laptop rehearsal and organiser-reserved data validation remain external steps.

Report work is currently deferred at the user's request. Any local report draft is not part of the completed/pushed deliverables.
