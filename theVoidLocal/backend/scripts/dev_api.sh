#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt >/dev/null
alembic upgrade head >/dev/null

HOST="${1:-127.0.0.1}"
PORT="${2:-8080}"

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
