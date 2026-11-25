#!/usr/bin/env bash
# Nightly encrypted Postgres backup for ZOS.
# Dumps compose Postgres (`pg_dump -Fc`), encrypts via macOS LibreSSL
# (`/usr/bin/openssl enc -aes-256-cbc -pbkdf2 -salt`), saves to iCloud Drive
# keeping the 14 newest files, and records metadata in ZOS for /admin Overview.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ZOS_ENV_FILE:-${ZOS_DIR}/.env}"
OPENSSL_BIN="/usr/bin/openssl"

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

ZOS_BACKUP_KEY="${ZOS_BACKUP_KEY:-$(read_env_var ZOS_BACKUP_KEY "")}"
POSTGRES_USER="${POSTGRES_USER:-$(read_env_var POSTGRES_USER zos)}"
POSTGRES_DB="${POSTGRES_DB:-$(read_env_var POSTGRES_DB zos)}"
BACKUP_DIR="${ZOS_BACKUP_DIR:-$(read_env_var ZOS_BACKUP_DIR "${HOME}/Library/Mobile Documents/com~apple~CloudDocs/zos-backups")}"
RETAIN_COUNT="${ZOS_BACKUP_RETAIN:-14}"

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

cd "${ZOS_DIR}"
docker compose --env-file "${ENV_FILE}" -f deploy/docker-compose.yml exec -T postgres \
  pg_dump -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -Fc \
  | "${OPENSSL_BIN}" enc -aes-256-cbc -pbkdf2 -salt -pass env:ZOS_BACKUP_KEY -out "${TMP_PATH}"

chmod 600 "${TMP_PATH}"
mv "${TMP_PATH}" "${FINAL_PATH}"

SIZE_BYTES="$(stat -f%z "${FINAL_PATH}")"

# Keep only the newest RETAIN_COUNT backup files
REMOVED=0
while IFS= read -r old_file; do
  [ -n "${old_file}" ] || continue
  rm -f "${old_file}"
  REMOVED=$((REMOVED + 1))
done < <(ls -1t "${BACKUP_DIR}"/zos-backup-*.dump.enc 2>/dev/null | tail -n +"$((RETAIN_COUNT + 1))" || true)

RETAINED="$(ls -1 "${BACKUP_DIR}"/zos-backup-*.dump.enc 2>/dev/null | wc -l | tr -d ' ')"

# Record metadata in ZOS so /admin Overview shows last backup time & size
docker compose --env-file "${ENV_FILE}" -f deploy/docker-compose.yml exec -T zos \
  uv run --no-sync zos backup record \
  --filename "${FILENAME}" \
  --size-bytes "${SIZE_BYTES}" \
  --completed-at "${TS_ISO}" \
  --retained "${RETAINED}" >/dev/null

echo "backup=${FILENAME} size_bytes=${SIZE_BYTES} completed_at=${TS_ISO} retained=${RETAINED} pruned=${REMOVED}"
