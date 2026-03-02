#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATABASE_URL="${SOURCE_DATABASE_URL:-${1:-}}"
TARGET_DATABASE_URL="${TARGET_DATABASE_URL:-${2:-}}"
SYNC_MODE="${SYNC_MODE:-replace}" # replace|append
EXCLUDE_TABLE_DATA="${EXCLUDE_TABLE_DATA:-public.alembic_version}"

if [[ -z "${SOURCE_DATABASE_URL}" || -z "${TARGET_DATABASE_URL}" ]]; then
  echo "Usage:"
  echo "  SOURCE_DATABASE_URL=... TARGET_DATABASE_URL=... $0"
  echo "  or: $0 <source_database_url> <target_database_url>"
  echo
  echo "Optional env:"
  echo "  SYNC_MODE=replace|append         (default: replace)"
  echo "  EXCLUDE_TABLE_DATA=public.alembic_version,public.some_table"
  exit 1
fi

if [[ "${SYNC_MODE}" != "replace" && "${SYNC_MODE}" != "append" ]]; then
  echo "Invalid SYNC_MODE='${SYNC_MODE}'. Use 'replace' or 'append'."
  exit 1
fi

USE_DOCKER_PGTOOLS=0
for cmd in pg_dump pg_restore psql; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    USE_DOCKER_PGTOOLS=1
  fi
done

if [[ "${USE_DOCKER_PGTOOLS}" == "1" ]] && ! command -v docker >/dev/null 2>&1; then
  echo "Missing pg_dump/pg_restore/psql and docker is not available."
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pg-sync.XXXXXX")"
DUMP_FILE="${TMP_DIR}/source_data.dump"
trap 'rm -rf "${TMP_DIR}"' EXIT

run_pg_dump() {
  if [[ "${USE_DOCKER_PGTOOLS}" == "1" ]]; then
    docker run --rm postgres:16 pg_dump "$@"
  else
    pg_dump "$@"
  fi
}

run_pg_restore() {
  if [[ "${USE_DOCKER_PGTOOLS}" == "1" ]]; then
    docker run --rm -v "${TMP_DIR}:${TMP_DIR}" postgres:16 pg_restore "$@"
  else
    pg_restore "$@"
  fi
}

run_psql() {
  if [[ "${USE_DOCKER_PGTOOLS}" == "1" ]]; then
    docker run --rm -i postgres:16 psql "$@"
  else
    psql "$@"
  fi
}

if [[ "${USE_DOCKER_PGTOOLS}" == "1" ]]; then
  echo "pg_dump/pg_restore/psql not found locally; using dockerized postgres:16 client tools."
fi

declare -a EXCLUDE_FLAGS=()
declare -a EXCLUDED_TABLE_NAMES=()
IFS=',' read -r -a EXCLUDED_ENTRIES <<< "${EXCLUDE_TABLE_DATA}"
for raw in "${EXCLUDED_ENTRIES[@]}"; do
  trimmed="$(echo "${raw}" | tr -d '[:space:]')"
  [[ -z "${trimmed}" ]] && continue
  EXCLUDE_FLAGS+=("--exclude-table-data=${trimmed}")
  EXCLUDED_TABLE_NAMES+=("${trimmed##*.}")
done

echo "Dumping source data (data-only) ..."
run_pg_dump "${SOURCE_DATABASE_URL}" \
  --format=custom \
  --data-only \
  --no-owner \
  --no-privileges \
  "${EXCLUDE_FLAGS[@]}" > "${DUMP_FILE}"

if [[ "${SYNC_MODE}" == "replace" ]]; then
  echo "Truncating target tables before restore ..."
  truncate_filter="TRUE"
  if [[ ${#EXCLUDED_TABLE_NAMES[@]} -gt 0 ]]; then
    quoted_tables=""
    for table_name in "${EXCLUDED_TABLE_NAMES[@]}"; do
      escaped="${table_name//\'/\'\'}"
      if [[ -n "${quoted_tables}" ]]; then
        quoted_tables="${quoted_tables}, "
      fi
      quoted_tables="${quoted_tables}'${escaped}'"
    done
    truncate_filter="tablename NOT IN (${quoted_tables})"
  fi

  run_psql "${TARGET_DATABASE_URL}" -v ON_ERROR_STOP=1 <<SQL
DO \$\$
DECLARE
  r RECORD;
BEGIN
  FOR r IN
    SELECT quote_ident(schemaname) || '.' || quote_ident(tablename) AS fqtn
    FROM pg_tables
    WHERE schemaname = 'public'
      AND ${truncate_filter}
    ORDER BY tablename
  LOOP
    EXECUTE 'TRUNCATE TABLE ' || r.fqtn || ' RESTART IDENTITY CASCADE';
  END LOOP;
END
\$\$;
SQL
fi

echo "Restoring source data into target ..."
run_pg_restore "${DUMP_FILE}" \
  --dbname "${TARGET_DATABASE_URL}" \
  --data-only \
  --no-owner \
  --no-privileges \
  --disable-triggers \
  --single-transaction \
  --exit-on-error

echo "Data sync complete."
echo "Recommended next step: run backend/scripts/postgres_parity_check.py"
