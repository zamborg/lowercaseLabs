#!/usr/bin/env bash
# Nightly encrypted Postgres backup for ZOS.
# Dumps compose Postgres (`pg_dump -Fc`), encrypts with OpenSSL AES-256-CBC + PBKDF2,
# saves to ZOS_BACKUP_DIR keeping the newest N files, runs optional ZOS_BACKUP_POST_HOOK
# (e.g. for syncing to S3/R2/Backblaze), and records metadata in ZOS for /admin Overview.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ZOS_ENV_FILE:-${ZOS_DIR}/.env}"
OPENSSL_BIN="${ZOS_OPENSSL_BIN:-$(command -v openssl || echo /usr/bin/openssl)}"
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --project) COMPOSE_PROJECT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ ! -f "${ENV_FILE}" ]; then
  echo "Error: env file not found at ${ENV_FILE}" >&2
  exit 1
fi

read_env_var() {
  local key="$1"
  local default_val="${2:-}"
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

if [ "$(uname -s)" = "Darwin" ]; then
  DEFAULT_BACKUP_DIR="${HOME}/Library/Mobile Documents/com~apple~CloudDocs/zos-backups"
else
  DEFAULT_BACKUP_DIR="${ZOS_DIR}/backups"
fi

ZOS_BACKUP_KEY="${ZOS_BACKUP_KEY:-$(read_env_var ZOS_BACKUP_KEY "")}"
POSTGRES_USER="${POSTGRES_USER:-$(read_env_var POSTGRES_USER zos)}"
ZOS_DB_USER="${ZOS_DB_USER:-$(read_env_var ZOS_DB_USER "${POSTGRES_USER}")}"
ZOS_DB_PASSWORD="${ZOS_DB_PASSWORD:-$(read_env_var ZOS_DB_PASSWORD "$(read_env_var POSTGRES_PASSWORD zos)")}"
POSTGRES_DB="${POSTGRES_DB:-$(read_env_var POSTGRES_DB zos)}"
BACKUP_DIR="${ZOS_BACKUP_DIR:-$(read_env_var ZOS_BACKUP_DIR "${DEFAULT_BACKUP_DIR}")}"
RETAIN_COUNT="${ZOS_BACKUP_RETAIN:-$(read_env_var ZOS_BACKUP_RETAIN 14)}"
POST_HOOK="${ZOS_BACKUP_POST_HOOK:-$(read_env_var ZOS_BACKUP_POST_HOOK "")}"

if [ -z "${ZOS_BACKUP_KEY}" ]; then
  echo "Error: ZOS_BACKUP_KEY is not set in ${ENV_FILE}" >&2
  exit 1
fi
export ZOS_BACKUP_KEY

mkdir -p "${BACKUP_DIR}"
chmod 700 "${BACKUP_DIR}"

TS_FILE="$(date -u +"%Y%m%d-%H%M%SZ")"
TS_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
FILENAME="zos-backup-${TS_FILE}.dump.enc"
FINAL_PATH="${BACKUP_DIR}/${FILENAME}"
TMP_PATH="${BACKUP_DIR}/.${FILENAME}.tmp.$$"

cleanup() {
  rm -f "${TMP_PATH}"
}
trap cleanup EXIT

COMPOSE_CMD=(docker compose --env-file "${ENV_FILE}" -f deploy/docker-compose.yml)
if [ -n "${COMPOSE_PROJECT}" ]; then
  COMPOSE_CMD+=(-p "${COMPOSE_PROJECT}")
fi

cd "${ZOS_DIR}"
"${COMPOSE_CMD[@]}" exec -T -e "PGPASSWORD=${ZOS_DB_PASSWORD}" postgres \
  pg_dump -U "${ZOS_DB_USER}" -d "${POSTGRES_DB}" -Fc --no-owner --no-privileges \
  | "${OPENSSL_BIN}" enc -aes-256-cbc -pbkdf2 -salt -pass env:ZOS_BACKUP_KEY -out "${TMP_PATH}"

chmod 600 "${TMP_PATH}"
mv "${TMP_PATH}" "${FINAL_PATH}"

SIZE_BYTES="$(wc -c < "${FINAL_PATH}" | tr -d ' ')"

# Optional external upload hook (e.g. aws s3 cp / rclone copy to R2/S3/Backblaze)
if [ -n "${POST_HOOK}" ]; then
  ZOS_BACKUP_FILE="${FINAL_PATH}" ZOS_BACKUP_FILENAME="${FILENAME}" sh -c "${POST_HOOK}"
fi

# Keep only the newest RETAIN_COUNT backup files
REMOVED=0
while IFS= read -r old_file; do
  [ -n "${old_file}" ] || continue
  rm -f "${old_file}"
  REMOVED=$((REMOVED + 1))
done < <(ls -1t "${BACKUP_DIR}"/zos-backup-*.dump.enc 2>/dev/null | tail -n +"$((RETAIN_COUNT + 1))" || true)

RETAINED="$(ls -1 "${BACKUP_DIR}"/zos-backup-*.dump.enc 2>/dev/null | wc -l | tr -d ' ')"

# Record metadata in ZOS so /admin Overview shows last backup time & size
"${COMPOSE_CMD[@]}" exec -T zos \
  uv run --no-sync zos backup record \
  --filename "${FILENAME}" \
  --size-bytes "${SIZE_BYTES}" \
  --completed-at "${TS_ISO}" \
  --retained "${RETAINED}" >/dev/null

echo "backup=${FILENAME} size_bytes=${SIZE_BYTES} completed_at=${TS_ISO} retained=${RETAINED} pruned=${REMOVED}"
