#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${1:-thevoid}"
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
if ! fly volumes list -a "${APP_NAME}" | awk 'NR>1 {print $1}' | grep -qx "${VOLUME_NAME}"; then
  fly volumes create "${VOLUME_NAME}" \
    --app "${APP_NAME}" \
    --region "${REGION}" \
    --size "${VOLUME_SIZE_GB}"
else
  echo "Volume ${VOLUME_NAME} already exists."
fi

if fly secrets list -a "${APP_NAME}" | awk 'NR>1 {print $1}' | grep -qx "DATABASE_URL"; then
  echo "DATABASE_URL secret already set."
else
  echo "DATABASE_URL secret is missing."
  echo "Attach managed Postgres before deploy:"
  echo "  fly mpg create --name ${APP_NAME}-db --region ${REGION}"
  echo "  fly mpg attach ${APP_NAME}-db -a ${APP_NAME}"
fi

echo "Bootstrap complete. Next:"
echo "1) Ensure DATABASE_URL exists (via fly mpg attach ...)"
echo "2) fly secrets set -a ${APP_NAME} JWT_SECRET=<strong-random> OPENAI_API_KEY=<key> ADMIN_USERNAME=<user> ADMIN_PASSWORD=<pass> APPLE_ALLOWED_AUDIENCES=<bundle-id>"
echo "3) fly deploy -a ${APP_NAME}"
