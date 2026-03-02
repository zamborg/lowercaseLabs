#!/usr/bin/env bash
set -euo pipefail

DB_APP="${1:-thevoid-local-postgres}"
REGION="${2:-sjc}"
VOLUME_NAME="${3:-thevoid_db_data}"
VOLUME_SIZE_GB="${4:-20}"
POSTGRES_DB="${5:-thevoid}"
POSTGRES_USER="${6:-thevoid}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE_CONFIG="${REPO_ROOT}/fly.postgres.toml"

if ! command -v fly >/dev/null 2>&1; then
  echo "fly CLI not found. Install from https://fly.io/docs/flyctl/install/"
  exit 1
fi

if [[ ! -f "${TEMPLATE_CONFIG}" ]]; then
  echo "Missing template config: ${TEMPLATE_CONFIG}"
  exit 1
fi

echo "Ensuring Fly app '${DB_APP}' exists..."
if ! fly apps list | awk 'NR > 1 {print $1}' | grep -qx "${DB_APP}"; then
  fly apps create "${DB_APP}"
else
  echo "App ${DB_APP} already exists."
fi

echo "Ensuring volume '${VOLUME_NAME}' exists in '${REGION}'..."
if ! fly volumes list -a "${DB_APP}" --json | grep -q "\"name\": *\"${VOLUME_NAME}\""; then
  fly volumes create "${VOLUME_NAME}" \
    --app "${DB_APP}" \
    --region "${REGION}" \
    --size "${VOLUME_SIZE_GB}" \
    --yes
else
  echo "Volume ${VOLUME_NAME} already exists."
fi

POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
GENERATED_PASSWORD=0
if [[ -z "${POSTGRES_PASSWORD}" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    POSTGRES_PASSWORD="$(openssl rand -base64 24 | tr -d '\n' | tr '/+' '_-')"
  else
    POSTGRES_PASSWORD="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)"
  fi
  GENERATED_PASSWORD=1
fi

echo "Setting Postgres runtime secrets on '${DB_APP}'..."
fly secrets set -a "${DB_APP}" \
  POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  POSTGRES_DB="${POSTGRES_DB}" \
  POSTGRES_USER="${POSTGRES_USER}" \
  PGDATA="/data/postgres"

TMP_CONFIG_BASE="$(mktemp "${TMPDIR:-/tmp}/fly-postgres.XXXXXX")"
TMP_CONFIG="${TMP_CONFIG_BASE}.toml"
cp "${TEMPLATE_CONFIG}" "${TMP_CONFIG}"

replace_line() {
  local pattern="$1"
  local replacement="$2"
  local escaped
  escaped="$(printf '%s' "${replacement}" | sed 's/[&|]/\\&/g')"
  sed -i.bak "s|${pattern}|${escaped}|" "${TMP_CONFIG}"
}

replace_line '^app = ".*"' "app = \"${DB_APP}\""
replace_line '^primary_region = ".*"' "primary_region = \"${REGION}\""
replace_line '^  POSTGRES_DB = ".*"' "  POSTGRES_DB = \"${POSTGRES_DB}\""
replace_line '^  POSTGRES_USER = ".*"' "  POSTGRES_USER = \"${POSTGRES_USER}\""
replace_line '^  source = ".*"' "  source = \"${VOLUME_NAME}\""
replace_line '^  initial_size = ".*"' "  initial_size = \"${VOLUME_SIZE_GB}gb\""
rm -f "${TMP_CONFIG}.bak"

echo "Deploying self-hosted Postgres app '${DB_APP}'..."
fly deploy --config "${TMP_CONFIG}" --app "${DB_APP}" --ha=false --yes
rm -f "${TMP_CONFIG}" "${TMP_CONFIG_BASE}"

DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${DB_APP}.internal:5432/${POSTGRES_DB}"

echo
echo "Self-hosted Postgres is deployed."
if [[ "${GENERATED_PASSWORD}" -eq 1 ]]; then
  echo "Generated POSTGRES_PASSWORD (store this now): ${POSTGRES_PASSWORD}"
fi
echo "Backend DATABASE_URL target:"
echo "  ${DATABASE_URL}"
echo
echo "Next steps:"
echo "1) Run migrations against target DB (alembic upgrade head)."
echo "2) Run data sync from managed Postgres into this DB."
echo "3) Run parity check."
echo "4) Switch backend DATABASE_URL secret to this value and deploy."
