#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${1:-thevoid-local}"
DB_APP="${2:-${APP_NAME}-postgres}"
POSTGRES_DB="${3:-thevoid}"
POSTGRES_USER="${4:-thevoid}"
DEPLOY_AFTER_SWITCH="${DEPLOY_AFTER_SWITCH:-0}"

if ! command -v fly >/dev/null 2>&1; then
  echo "fly CLI not found. Install from https://fly.io/docs/flyctl/install/"
  exit 1
fi

if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
  echo "POSTGRES_PASSWORD is required."
  echo "Example:"
  echo "  POSTGRES_PASSWORD='...' $0 ${APP_NAME} ${DB_APP} ${POSTGRES_DB} ${POSTGRES_USER}"
  exit 1
fi

DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${DB_APP}.internal:5432/${POSTGRES_DB}"

echo "Updating DATABASE_URL secret for app '${APP_NAME}'..."
fly secrets set -a "${APP_NAME}" DATABASE_URL="${DATABASE_URL}"

echo "DATABASE_URL switched to self-hosted Postgres (${DB_APP}.internal)."
echo "DATABASE_URL now points to:"
echo "  postgresql://${POSTGRES_USER}:***@${DB_APP}.internal:5432/${POSTGRES_DB}"

if [[ "${DEPLOY_AFTER_SWITCH}" == "1" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo "Deploying app '${APP_NAME}' to apply and run release migration command..."
  "${SCRIPT_DIR}/fly_deploy.sh" "${APP_NAME}"
fi
