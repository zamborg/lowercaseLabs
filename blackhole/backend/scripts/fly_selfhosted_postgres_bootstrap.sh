#!/usr/bin/env bash
set -euo pipefail

DB_APP="${1:-blackhole-postgres}"
REGION="${2:-sjc}"
VOLUME_NAME="${3:-blackhole_db_data}"
VOLUME_SIZE_GB="${4:-20}"
POSTGRES_DB="${5:-blackhole}"
POSTGRES_USER="${6:-blackhole}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE_CONFIG="${REPO_ROOT}/fly.postgres.toml"

if ! command -v fly >/dev/null 2>&1; then
  echo "fly CLI not found"
  exit 1
fi

if ! fly apps list | awk 'NR > 1 {print $1}' | grep -qx "${DB_APP}"; then
  fly apps create "${DB_APP}"
else
  echo "Fly app ${DB_APP} already exists."
fi

if ! fly volumes list -a "${DB_APP}" --json | grep -q "\"name\": *\"${VOLUME_NAME}\""; then
  fly volumes create "${VOLUME_NAME}" --app "${DB_APP}" --region "${REGION}" --size "${VOLUME_SIZE_GB}" --yes
else
  echo "Volume ${VOLUME_NAME} already exists."
fi

POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
if [[ -z "${POSTGRES_PASSWORD}" ]]; then
  POSTGRES_PASSWORD="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)"
  echo "Generated POSTGRES_PASSWORD=${POSTGRES_PASSWORD}"
fi

fly secrets set -a "${DB_APP}" \
  POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  POSTGRES_DB="${POSTGRES_DB}" \
  POSTGRES_USER="${POSTGRES_USER}" \
  PGDATA="/data/postgres"

TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/blackhole-postgres.XXXXXX.toml")"
cp "${TEMPLATE_CONFIG}" "${TMP_CONFIG}"

perl -0pi -e "s/app = '.*?'/app = '${DB_APP}'/g" "${TMP_CONFIG}"
perl -0pi -e "s/primary_region = '.*?'/primary_region = '${REGION}'/g" "${TMP_CONFIG}"
perl -0pi -e "s/POSTGRES_DB = '.*?'/POSTGRES_DB = '${POSTGRES_DB}'/g" "${TMP_CONFIG}"
perl -0pi -e "s/POSTGRES_USER = '.*?'/POSTGRES_USER = '${POSTGRES_USER}'/g" "${TMP_CONFIG}"
perl -0pi -e "s/source = '.*?'/source = '${VOLUME_NAME}'/g" "${TMP_CONFIG}"
perl -0pi -e "s/initial_size = '.*?'/initial_size = '${VOLUME_SIZE_GB}gb'/g" "${TMP_CONFIG}"

fly deploy --config "${TMP_CONFIG}" --app "${DB_APP}" --ha=false --yes
rm -f "${TMP_CONFIG}"

echo "Internal DATABASE_URL:"
echo "postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${DB_APP}.internal:5432/${POSTGRES_DB}"
