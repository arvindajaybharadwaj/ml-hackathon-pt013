# PathSnap backend

The backend is now integrated with the real data pipeline, the selected Modal model bundle and the React dashboard. See the root README for the complete setup and two-laptop instructions.

From the repository root:

```bash
.venv/bin/python -m pip install -r requirements.txt
bash run_backend.sh
```

Defaults load:

- Raw inference inputs: `Student folder/student_test/`.
- Prepared candidates: `prepared_data/student_test/candidates.parquet`.
- Feature schema: root `feature_schema.json`.
- Selected trained bundle: `artifacts/model/`.

The server refuses to silently fall back to distance scoring. Missing models, candidate tables or inconsistent schema versions fail explicitly. Optional environment overrides are documented in `.env.example`; the launch script inherits exported variables and does not install packages or overwrite configuration.

Endpoints:

- `GET /health`: actual model/data status.
- `GET /api/trips`: real IDs in the configured partition.
- `GET /api/trips/{id}`: observation/network summary.
- `POST /api/predict/{id}`: trained candidate scores and HMM decoding.
- `GET /api/evaluate/{id}`: reference comparison, performed server-side after prediction.
- `GET /api/maps/{id}`: Folium HTML with inference inputs and matched roads; no reference route access.

Prediction and map construction never read `.route` labels. Evaluation returns aggregate metrics, not reference road IDs. Last four decoded trips are cached per process to avoid repeating HMM work for maps/evaluation. Changing model/data requires restarting the process.

For legacy experiments, the previous per-trip candidate files and joblib adapters remain available via explicit environment settings. Production defaults use the new versioned ML bundle and combined partition tables. Legacy documents in `docs/` and old distance-fallback smoke evidence in `results/` describe the earlier backend component; use the root README and `evidence/integration/` for current integration commands and verification.

Run the entire Python suite from the repository root:

```bash
.venv/bin/python -m pytest -q
```
