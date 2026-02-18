#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${1:-thevoid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/fly.toml"

if ! command -v fly >/dev/null 2>&1; then
  echo "fly CLI not found. Install from https://fly.io/docs/flyctl/install/"
  exit 1
fi

if ! fly secrets list -a "${APP_NAME}" | awk 'NR>1 {print $1}' | grep -qx "DATABASE_URL"; then
  echo "DATABASE_URL secret is missing for app '${APP_NAME}'."
  echo "Attach Fly Managed Postgres before deploying:"
  echo "  fly mpg create --name ${APP_NAME}-db --region sjc"
  echo "  fly mpg attach ${APP_NAME}-db -a ${APP_NAME}"
  exit 1
fi

cd "${REPO_ROOT}"
exec fly deploy --config "${CONFIG_PATH}" --app "${APP_NAME}"
