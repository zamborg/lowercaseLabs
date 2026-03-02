#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${1:-thevoid-local}"
REGION="${2:-sjc}"
VOLUME_NAME="${3:-thevoid_data}"
VOLUME_SIZE_GB="${4:-10}"

if ! command -v fly >/dev/null 2>&1; then
  echo "fly CLI not found. Install from https://fly.io/docs/flyctl/install/"
  exit 1
fi

echo "Ensuring Fly app '${APP_NAME}' exists..."
if ! fly apps list | awk '{print $1}' | grep -qx "${APP_NAME}"; then
  fly apps create "${APP_NAME}"
else
  echo "App ${APP_NAME} already exists."
fi

echo "Ensuring volume '${VOLUME_NAME}' exists in region '${REGION}'..."
if ! fly volumes list -a "${APP_NAME}" --json | rg -q "\"name\":\\s*\"${VOLUME_NAME}\""; then
  fly volumes create "${VOLUME_NAME}" \
    --app "${APP_NAME}" \
    --region "${REGION}" \
    --size "${VOLUME_SIZE_GB}" \
    --yes
else
  echo "Volume ${VOLUME_NAME} already exists."
fi

if fly secrets list -a "${APP_NAME}" --json | rg -q '"name":\s*"DATABASE_URL"'; then
  echo "DATABASE_URL secret already set."
else
  echo "DATABASE_URL secret is missing."
  echo "Set DATABASE_URL before deploy."
  echo "If using self-hosted Postgres on Fly:"
  echo "  backend/scripts/fly_selfhosted_postgres_bootstrap.sh ${APP_NAME}-postgres ${REGION}"
  echo "  POSTGRES_PASSWORD='<db-password>' backend/scripts/fly_switch_database_url.sh ${APP_NAME} ${APP_NAME}-postgres"
fi

echo "Bootstrap complete. Next:"
echo "1) Ensure DATABASE_URL secret exists (managed or self-hosted Postgres)"
echo "2) fly secrets set -a ${APP_NAME} JWT_SECRET=<strong-random> OPENAI_API_KEY=<key> ADMIN_USERNAME=<user> ADMIN_PASSWORD=<pass> APPLE_ALLOWED_AUDIENCES=<bundle-id>"
echo "3) fly deploy -a ${APP_NAME}"
