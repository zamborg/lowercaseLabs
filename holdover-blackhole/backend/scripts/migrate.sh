#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -x ".venv312/bin/alembic" ]]; then
  exec .venv312/bin/alembic upgrade head
fi

if [[ -x ".venv/bin/alembic" ]]; then
  exec .venv/bin/alembic upgrade head
fi

exec alembic upgrade head

