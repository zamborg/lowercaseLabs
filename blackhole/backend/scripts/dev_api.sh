#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -x ".venv312/bin/uvicorn" ]]; then
  exec .venv312/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
fi

if [[ -x ".venv/bin/uvicorn" ]]; then
  exec .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
fi

exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

