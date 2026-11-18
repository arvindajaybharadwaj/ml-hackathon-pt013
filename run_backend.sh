#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn backend.app.main:app --app-dir backend --host 0.0.0.0 --port 8000 "$@"
