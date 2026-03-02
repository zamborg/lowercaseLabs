#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
POETRY="./scripts/poetry_cmd.sh"

"$POETRY" install --no-interaction --no-ansi >/dev/null
"$POETRY" run alembic upgrade head >/dev/null

"$POETRY" run python -m app.worker
