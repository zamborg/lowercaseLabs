#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"migration message\""
  exit 1
fi
POETRY="./scripts/poetry_cmd.sh"

"$POETRY" install --no-interaction --no-ansi >/dev/null

"$POETRY" run alembic revision --autogenerate -m "$1"
