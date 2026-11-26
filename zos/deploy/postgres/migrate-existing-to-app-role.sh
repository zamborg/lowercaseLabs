#!/usr/bin/env bash
# Idempotent migration script for existing ZOS Postgres volumes:
# Creates a restricted non-superuser core role (CREATEROLE, NOSUPERUSER) and transfers
# ownership of the database, schema public, all core tables/sequences/views/functions,
# and existing mod_* module role administration to it.
#
# Usage:
#   ./deploy/postgres/migrate-existing-to-app-role.sh
#   ./deploy/postgres/migrate-existing-to-app-role.sh --env-file .env
#   ./deploy/postgres/migrate-existing-to-app-role.sh --container zos-pg-test --db zos_test --super-user zos --app-user zos_app --app-password secret
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ZOS_ENV_FILE:-${ZOS_DIR}/.env}"
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-}"
TARGET_CONTAINER=""
SUPER_USER=""
SUPER_PASSWORD=""
APP_USER=""
APP_PASSWORD=""
DB_NAME=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --project) COMPOSE_PROJECT="$2"; shift 2 ;;
    --container) TARGET_CONTAINER="$2"; shift 2 ;;
    --super-user) SUPER_USER="$2"; shift 2 ;;
    --super-password) SUPER_PASSWORD="$2"; shift 2 ;;
    --app-user) APP_USER="$2"; shift 2 ;;
    --app-password) APP_PASSWORD="$2"; shift 2 ;;
    --db) DB_NAME="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

read_env_var() {
  local key="$1"
  local default_val="${2:-}"
  [ -f "${ENV_FILE}" ] || { printf "%s" "${default_val}"; return; }
  python3 - "$ENV_FILE" "$key" "$default_val" <<'PY'
import sys
from pathlib import Path

env_path, target, default = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
val = default
for raw in env_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    if k.strip() != target:
        continue
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        v = v[1:-1]
    val = v
print(val, end="")
PY
}

SUPER_USER="${SUPER_USER:-$(read_env_var POSTGRES_USER zos)}"
SUPER_PASSWORD="${SUPER_PASSWORD:-$(read_env_var POSTGRES_PASSWORD zos)}"
DB_NAME="${DB_NAME:-$(read_env_var POSTGRES_DB zos)}"
APP_USER="${APP_USER:-$(read_env_var ZOS_DB_USER zos_app)}"
APP_PASSWORD="${APP_PASSWORD:-$(read_env_var ZOS_DB_PASSWORD "")}"

if [ "${APP_USER}" = "${SUPER_USER}" ]; then
  echo "Error: ZOS_DB_USER ('${APP_USER}') must differ from superuser POSTGRES_USER ('${SUPER_USER}')" >&2
  exit 1
fi

if [ -z "${APP_PASSWORD}" ]; then
  echo "Error: ZOS_DB_PASSWORD is required (set it in ${ENV_FILE} or pass --app-password)" >&2
  exit 1
fi

read -r -d '' MIGRATION_SQL <<'EOSQL' || true
SELECT format(
  'CREATE ROLE %I WITH LOGIN PASSWORD %L CREATEROLE NOSUPERUSER NOCREATEDB NOREPLICATION',
  :'app_user',
  :'app_password'
)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user')
\gexec

SELECT format(
  'ALTER ROLE %I WITH LOGIN PASSWORD %L CREATEROLE NOSUPERUSER NOCREATEDB NOREPLICATION',
  :'app_user',
  :'app_password'
)
\gexec

SELECT format('ALTER DATABASE %I OWNER TO %I', :'db_name', :'app_user') \gexec
SELECT format('ALTER SCHEMA public OWNER TO %I', :'app_user') \gexec
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

-- Transfer ownership of all core tables, sequences, views, materialized views, and non-extension functions in public
SELECT format('ALTER TABLE public.%I OWNER TO %I', tablename, :'app_user')
FROM pg_tables
WHERE schemaname = 'public'
\gexec

SELECT format('ALTER SEQUENCE public.%I OWNER TO %I', sequencename, :'app_user')
FROM pg_sequences
WHERE schemaname = 'public'
\gexec

SELECT format('ALTER VIEW public.%I OWNER TO %I', viewname, :'app_user')
FROM pg_views
WHERE schemaname = 'public'
\gexec

SELECT format('ALTER MATERIALIZED VIEW public.%I OWNER TO %I', matviewname, :'app_user')
FROM pg_matviews
WHERE schemaname = 'public'
\gexec

SELECT format('ALTER FUNCTION public.%I(%s) OWNER TO %I', p.proname, pg_get_function_identity_arguments(p.oid), :'app_user')
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public'
  AND NOT EXISTS (
    SELECT 1 FROM pg_depend d
    WHERE d.objid = p.oid AND d.deptype = 'e'
  )
\gexec

-- Ensure pg_trgm exists and is usable by the app role
CREATE EXTENSION IF NOT EXISTS pg_trgm;
SELECT format('GRANT ALL ON SCHEMA public TO %I', :'app_user') \gexec

-- Grant membership + ADMIN OPTION on every existing mod_* module role so the non-superuser
-- CREATEROLE app role can administer, back up, upgrade, and purge existing modules (PG 16+).
SELECT format('GRANT %I TO %I WITH ADMIN OPTION', rolname, :'app_user')
FROM pg_roles
WHERE rolname LIKE 'mod\_%' ESCAPE '\'
\gexec

-- Keep public schema and core tables locked down from PUBLIC and all mod_* roles
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC;

SELECT format('REVOKE ALL ON SCHEMA public FROM %I', rolname)
FROM pg_roles
WHERE rolname LIKE 'mod\_%' ESCAPE '\'
\gexec

SELECT format('REVOKE ALL ON ALL TABLES IN SCHEMA public FROM %I', rolname)
FROM pg_roles
WHERE rolname LIKE 'mod\_%' ESCAPE '\'
\gexec

SELECT format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM %I', rolname)
FROM pg_roles
WHERE rolname LIKE 'mod\_%' ESCAPE '\'
\gexec
EOSQL

if [ -n "${TARGET_CONTAINER}" ]; then
  printf "%s\n" "${MIGRATION_SQL}" | docker exec -i -e "PGPASSWORD=${SUPER_PASSWORD}" "${TARGET_CONTAINER}" \
    psql -v ON_ERROR_STOP=1 -U "${SUPER_USER}" -d "${DB_NAME}" \
      -v app_user="${APP_USER}" \
      -v app_password="${APP_PASSWORD}" \
      -v db_name="${DB_NAME}" >/dev/null
else
  COMPOSE_CMD=(docker compose --env-file "${ENV_FILE}" -f deploy/docker-compose.yml)
  if [ -n "${COMPOSE_PROJECT}" ]; then
    COMPOSE_CMD+=(-p "${COMPOSE_PROJECT}")
  fi
  cd "${ZOS_DIR}"
  printf "%s\n" "${MIGRATION_SQL}" | "${COMPOSE_CMD[@]}" exec -T -e "PGPASSWORD=${SUPER_PASSWORD}" postgres \
    psql -v ON_ERROR_STOP=1 -U "${SUPER_USER}" -d "${DB_NAME}" \
      -v app_user="${APP_USER}" \
      -v app_password="${APP_PASSWORD}" \
      -v db_name="${DB_NAME}" >/dev/null
fi

echo "Migrated db=${DB_NAME} ownership and module role grants to non-superuser role=${APP_USER}"
