#!/usr/bin/env bash
# Decrypt and restore an encrypted ZOS Postgres backup created by deploy/backup/backup.sh.
#
# Usage:
#   deploy/backup/restore.sh <path/to/zos-backup-*.dump.enc>
#   deploy/backup/restore.sh <path/to/backup.dump.enc> --container zos-pg-test --db zos_restore_test --user postgres
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ZOS_ENV_FILE:-${ZOS_DIR}/.env}"
OPENSSL_BIN="/usr/bin/openssl"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <backup.dump.enc> [--container NAME] [--db DBNAME] [--user PGUSER]" >&2
  exit 2
fi

BACKUP_FILE="$1"
shift

TARGET_CONTAINER=""
TARGET_DB=""
TARGET_USER=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --container) TARGET_CONTAINER="$2"; shift 2 ;;
    --db) TARGET_DB="$2"; shift 2 ;;
    --user) TARGET_USER="$2"; shift 2 ;;
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

PG_USER="${TARGET_USER:-$(read_env_var POSTGRES_USER zos)}"
PG_DB="${TARGET_DB:-$(read_env_var POSTGRES_DB zos)}"

if [ -n "${TARGET_CONTAINER}" ]; then
  "${OPENSSL_BIN}" enc -d -aes-256-cbc -pbkdf2 -pass env:ZOS_BACKUP_KEY -in "${BACKUP_FILE}" \
    | docker exec -i "${TARGET_CONTAINER}" \
      pg_restore -U "${PG_USER}" -d "${PG_DB}" --clean --if-exists --no-owner --no-privileges
else
  cd "${ZOS_DIR}"
  "${OPENSSL_BIN}" enc -d -aes-256-cbc -pbkdf2 -pass env:ZOS_BACKUP_KEY -in "${BACKUP_FILE}" \
    | docker compose --env-file "${ENV_FILE}" -f deploy/docker-compose.yml exec -T postgres \
      pg_restore -U "${PG_USER}" -d "${PG_DB}" --clean --if-exists --no-owner --no-privileges
fi

echo "Restored $(basename "${BACKUP_FILE}") into db=${PG_DB} (container=${TARGET_CONTAINER:-compose:postgres})"
