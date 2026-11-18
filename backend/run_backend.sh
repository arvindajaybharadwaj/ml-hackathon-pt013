#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env. Edit PATHSNAP_DATA_ROOT before starting the API."
fi
set -a
source .env
set +a
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
