# Two laptop demonstration

## Laptop 1 model and API

Use Python 3.12. Clone the repository, supply the raw challenge `Student folder/` (not stored in Git), and install dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash run_backend.sh
```

The trained model and tuned HMM configuration are in `artifacts/model/`. Student-test prepared feature tables are included. The API listens on `0.0.0.0:8000`; find this laptop's LAN IPv4 address using network settings (for example `192.168.1.10`). Keep the API terminal open so receiving requests are visible.

## Laptop 2 user interface

Only Node.js 20.19+ or 22.12+ and the frontend source are required. No Python ML environment, model, reference labels or raw GPS files are required on this laptop.

```bash
npm --prefix frontend ci
node frontend/start-demo.mjs --backend http://192.168.1.10:8000 --port 5173
```

Substitute the actual Laptop 1 address. The launcher checks `/health` and `/api/trips`, then starts Vite with the API proxy. Open http://localhost:5173 on Laptop 2. Select a real trip, click Run Map Matching, inspect the route, pointwise/sequence metrics and model comparison graphs. All requests, including maps and plots, go through the UI proxy to Laptop 1. Reference labels remain server-side.

Connection check without starting the UI:

```bash
node frontend/start-demo.mjs --backend http://192.168.1.10:8000 --check
```

Both devices must be on a mutually reachable LAN. Allow the API process through the receiving laptop's firewall on port 8000; do not disable the firewall. A university/guest Wi-Fi network may isolate devices. Use a permitted shared network or hotspot if isolation prevents the connection. Map tiles and browser map assets need internet.

## Reserved evaluator trips when supplied

On Laptop 1 only, place organiser-supplied inference files under `Evaluator data/evaluator_test/00000000/` etc. Generate candidate features without reading `.route` labels:

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_partition.py --raw-root "Evaluator data" --partition evaluator_test --output prepared_data/evaluator_test
```

Then export runtime settings and restart the API:

```bash
export PATHSNAP_DATA_ROOT="$PWD/Evaluator data"
export PATHSNAP_PARTITION=evaluator_test
export PATHSNAP_CANDIDATE_DIR="$PWD/prepared_data/evaluator_test"
bash run_backend.sh
```

Keep evaluator `.route` references on Laptop 1 when authorised for server evaluation. They are loaded after prediction; if observation labels are absent, the evaluation service derives alignment using the same separate route alignment procedure. Evaluator labels never enter candidate generation or inference. Model/decoder tuning scripts reject evaluator partitions. Do not retune after seeing evaluator performance.

For Windows, use `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --app-dir backend --host 0.0.0.0 --port 8000` and the corresponding PowerShell environment variable syntax; the Node frontend launcher supports Windows npm.cmd.

## Verification and capture checklist

- Backend health reports trained model and data available.
- UI displays all ten organiser-supplied IDs (no fabricated IDs).
- Exercise every trip; inspect receiving API logs on Laptop 1.
- Check the Folium map and aligned evaluation coverage for each trip.
- Capture the UI on Laptop 2 and receiving request log on Laptop 1 for the final PDF.
- Export the edited Word report to PDF after filling names/SRNs.

## What has actually been tested

Separate API and frontend processes were tested through this workstation's LAN address, including prediction, evaluation and maps for all ten student-test trips. This validates the LAN-address proxy configuration but is **not a claim that two physical laptops or reserved evaluator data have been tested**. The final hardware rehearsal and reserved-data validation require those external resources.
