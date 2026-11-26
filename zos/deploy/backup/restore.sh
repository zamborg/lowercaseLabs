#!/usr/bin/env bash
# Decrypt and restore an encrypted ZOS Postgres backup created by deploy/backup/backup.sh.
#
# Usage:
#   deploy/backup/restore.sh <path/to/zos-backup-*.dump.enc>
#   deploy/backup/restore.sh <path/to/backup.dump.enc> --container zos-pg-test --db zos_restore_test --user zos_app
#   deploy/backup/restore.sh <path/to/backup.dump.enc> --project zos-rehearsal --env-file .env.rehearsal
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ZOS_ENV_FILE:-${ZOS_DIR}/.env}"
OPENSSL_BIN="${ZOS_OPENSSL_BIN:-$(command -v openssl || echo /usr/bin/openssl)}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <backup.dump.enc> [--container NAME] [--project NAME] [--env-file PATH] [--db DBNAME] [--user PGUSER] [--no-clean]" >&2
  exit 2
fi

BACKUP_FILE="$1"
shift

TARGET_CONTAINER=""
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-}"
TARGET_DB=""
TARGET_USER=""
TARGET_PASSWORD=""
DO_CLEAN="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --container) TARGET_CONTAINER="$2"; shift 2 ;;
    --project) COMPOSE_PROJECT="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --db) TARGET_DB="$2"; shift 2 ;;
    --user) TARGET_USER="$2"; shift 2 ;;
    --password) TARGET_PASSWORD="$2"; shift 2 ;;
    --no-clean) DO_CLEAN="0"; shift 1 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ ! -f "${BACKUP_FILE}" ]; then
  echo "Error: backup file not found: ${BACKUP_FILE}" >&2
  exit 1
fi

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

ZOS_BACKUP_KEY="${ZOS_BACKUP_KEY:-$(read_env_var ZOS_BACKUP_KEY "")}"
if [ -z "${ZOS_BACKUP_KEY}" ]; then
  echo "Error: ZOS_BACKUP_KEY is not set (export it or set it in ${ENV_FILE})" >&2
  exit 1
fi
export ZOS_BACKUP_KEY

DEFAULT_PG_USER="$(read_env_var POSTGRES_USER zos)"
PG_USER="${TARGET_USER:-$(read_env_var ZOS_DB_USER "${DEFAULT_PG_USER}")}"
PG_PASS="${TARGET_PASSWORD:-$(read_env_var ZOS_DB_PASSWORD "$(read_env_var POSTGRES_PASSWORD zos)")}"
PG_DB="${TARGET_DB:-$(read_env_var POSTGRES_DB zos)}"

# Build pg_restore flags. On non-superuser restores, extensions owned by superuser (such as
# pg_trgm created during init) emit harmless errors if pg_restore tries to DROP/COMMENT them;
# `--no-comments` avoids COMMENT ON EXTENSION errors while preserving all schema & table data.
RESTORE_FLAGS=(--no-owner --no-privileges --no-comments)
if [ "${DO_CLEAN}" = "1" ]; then
  RESTORE_FLAGS+=(--clean --if-exists)
fi

if [ -n "${TARGET_CONTAINER}" ]; then
  "${OPENSSL_BIN}" enc -d -aes-256-cbc -pbkdf2 -pass env:ZOS_BACKUP_KEY -in "${BACKUP_FILE}" \
    | docker exec -i -e "PGPASSWORD=${PG_PASS}" "${TARGET_CONTAINER}" \
      pg_restore -U "${PG_USER}" -d "${PG_DB}" "${RESTORE_FLAGS[@]}"
else
  COMPOSE_CMD=(docker compose --env-file "${ENV_FILE}" -f deploy/docker-compose.yml)
  if [ -n "${COMPOSE_PROJECT}" ]; then
    COMPOSE_CMD+=(-p "${COMPOSE_PROJECT}")
  fi
  cd "${ZOS_DIR}"
  "${OPENSSL_BIN}" enc -d -aes-256-cbc -pbkdf2 -pass env:ZOS_BACKUP_KEY -in "${BACKUP_FILE}" \
    | "${COMPOSE_CMD[@]}" exec -T -e "PGPASSWORD=${PG_PASS}" postgres \
      pg_restore -U "${PG_USER}" -d "${PG_DB}" "${RESTORE_FLAGS[@]}"
fi

echo "Restored $(basename "${BACKUP_FILE}") into db=${PG_DB} (user=${PG_USER}, container=${TARGET_CONTAINER:-compose:${COMPOSE_PROJECT:-default}/postgres})"
