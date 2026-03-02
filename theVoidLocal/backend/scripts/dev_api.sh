#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
POETRY="./scripts/poetry_cmd.sh"

"$POETRY" install --no-interaction --no-ansi >/dev/null
"$POETRY" run alembic upgrade head >/dev/null

HOST="${1:-127.0.0.1}"
PORT="${2:-8080}"

"$POETRY" run uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
