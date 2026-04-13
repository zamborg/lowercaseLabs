#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${1:?app name required}"
DB_APP="${2:?db app name required}"
POSTGRES_USER="${3:-blackhole}"
POSTGRES_DB="${4:-blackhole}"

if ! command -v fly >/dev/null 2>&1; then
  echo "fly CLI not found"
  exit 1
fi

if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
  echo "Set POSTGRES_PASSWORD in the environment before running this script."
  exit 1
fi

DATABASE_URL="postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${DB_APP}.internal:5432/${POSTGRES_DB}"
fly secrets set -a "${APP_NAME}" DATABASE_URL="${DATABASE_URL}"

echo "DATABASE_URL updated on ${APP_NAME}"
