#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${1:-blackhole-app}"
REGION="${2:-sjc}"
VOLUME_NAME="${3:-blackhole_data}"
VOLUME_SIZE_GB="${4:-10}"

if ! command -v fly >/dev/null 2>&1; then
  echo "fly CLI not found"
  exit 1
fi

if ! fly apps list | awk 'NR > 1 {print $1}' | grep -qx "${APP_NAME}"; then
  fly apps create "${APP_NAME}"
else
  echo "Fly app ${APP_NAME} already exists."
fi

if ! fly volumes list -a "${APP_NAME}" --json | grep -q "\"name\": *\"${VOLUME_NAME}\""; then
  fly volumes create "${VOLUME_NAME}" --app "${APP_NAME}" --region "${REGION}" --size "${VOLUME_SIZE_GB}" --yes
else
  echo "Volume ${VOLUME_NAME} already exists."
fi

echo "Next steps:"
echo "1) Run backend/scripts/fly_selfhosted_postgres_bootstrap.sh ${APP_NAME}-postgres ${REGION}"
echo "2) Set DATABASE_URL on ${APP_NAME} to the internal Postgres URL"
echo "3) Set JWT_SECRET and OPENAI_API_KEY secrets"
echo "4) fly deploy -a ${APP_NAME}"

