# PathSnap dashboard

React dashboard connected to the trained FastAPI/HMM backend. From the repository root:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev -- --port 5173
```

Open http://localhost:5173. Start `bash run_backend.sh` in another terminal first. The dashboard fetches health and real trip IDs, submits predictions, displays a Folium map and compares classifier/HMM route metrics. Errors and loading states are shown; no fake trips or fabricated scores are supplied.

API requests and the map iframe use the same-origin Vite proxy. For a backend on another laptop, set `VITE_BACKEND_TARGET=http://BACKEND_LAPTOP_IP:8000` in `.env.local` and restart Vite. For a separately hosted static build, set `VITE_API_URL` at build time and configure backend CORS for the UI origin.

```bash
npm --prefix frontend run lint
npm --prefix frontend run build
```

Map tiles and Folium browser assets require internet. Ground-truth routes are never sent to the browser.
