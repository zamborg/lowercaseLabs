#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${1:-thevoid-local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/fly.toml"

if ! command -v fly >/dev/null 2>&1; then
  echo "fly CLI not found. Install from https://fly.io/docs/flyctl/install/"
  exit 1
fi

if ! fly secrets list -a "${APP_NAME}" --json | rg -q '"name":\s*"DATABASE_URL"'; then
  echo "DATABASE_URL secret is missing for app '${APP_NAME}'."
  echo "Set DATABASE_URL before deploying (managed or self-hosted Postgres)."
  exit 1
fi

cd "${REPO_ROOT}"
exec fly deploy --config "${CONFIG_PATH}" --app "${APP_NAME}"
