# Two-Laptop Deployment – Person 3

## Laptop 1: backend

1. Extract the dataset and the Person 3 package.
2. Create a virtual environment and install `requirements.txt`.
3. Copy `.env.example` to `.env` and set `PATHSNAP_DATA_ROOT`.
4. During development, use `PATHSNAP_ENV=development` and `PATHSNAP_PARTITION=student_test`.
5. After receiving Person 1/2 artifacts, configure candidate/model paths and set `PATHSNAP_ENV=production`.
6. Start with `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`.
7. Open `http://127.0.0.1:8000/docs` and check `/health`.

Find Laptop 1's LAN IP (for example, `ipconfig getifaddr en0` on macOS or
`ip addr` on Linux). Allow inbound TCP port 8000 on the private LAN firewall.

## Laptop 2: frontend

Set the frontend environment variable to:

```env
VITE_API_BASE_URL=http://LAPTOP_1_LAN_IP:8000
```

Add the frontend origin to `PATHSNAP_CORS_ORIGINS` on Laptop 1. Do not use
`localhost` in Laptop 2's frontend configuration: that would point to Laptop 2
itself. Both laptops should be on the same trusted network.

## Verify from Laptop 2

```bash
curl http://LAPTOP_1_LAN_IP:8000/health
curl http://LAPTOP_1_LAN_IP:8000/api/trips
curl -X POST http://LAPTOP_1_LAN_IP:8000/api/predict/TRIP_ID
```

Watch the Laptop 1 terminal for request logs including the remote client IP.
The prediction endpoint returns predicted arc IDs, `[longitude, latitude]`
route coordinates, selected candidate diagnostics and processing-time breakdown.
The evaluator labels are not included in prediction responses.
