#!/usr/bin/env bash
# Idempotent ZOS installer and updater for fresh Ubuntu 24.04 servers (run as root, >=2GB RAM).
#
# Examples:
#   sudo bash deploy/install.sh --edge exe
#   sudo bash deploy/install.sh --domain zos.example.com --edge caddy
#   sudo bash deploy/install.sh --domain zos.example.com --edge cloudflared
#   sudo bash deploy/install.sh --edge none
#   sudo bash deploy/install.sh --update
set -euo pipefail

EDGE=""
EDGE_SET=0
DOMAIN=""
DOMAIN_SET=0
INSTALL_DIR="/opt/zos"
REPO_URL=""
REPO_URL_SET=0
GIT_REF=""
GIT_REF_SET=0
LOCAL_SOURCE=""
LOCAL_SOURCE_SET=0
UPDATE_MODE=0
SKIP_PASSPHRASE=0

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Options:
  --edge caddy|exe|cloudflared|none  Edge reverse proxy / TLS mode (default: saved or caddy)
  --domain <hostname>                Public domain (required for --edge caddy; optional custom domain for exe; default: saved)
  --dir <path>                    Install directory (default: /opt/zos)
  --repo <git-url>                Git repository URL (default: saved or $ZOS_REPO_URL)
  --ref <git-ref>                 Git branch/tag/commit to checkout (default: saved or main)
  --local-source <path>           Copy ZOS source from a local directory instead of git clone
  --update                        Back up first, pull/rebuild, migrate, and recreate keeping data
  --skip-passphrase               Skip interactive owner passphrase prompt (set later before login)
  -h, --help                      Show this help

Registry images (optional, digest-pinned <repo>@sha256:<64-hex>):
  ZOS_IMAGE_CORE                  Use prebuilt core image instead of building from source
  ZOS_IMAGE_MEMORY                Pin memory module manifest to digest image
  ZOS_IMAGE_INBOX                 Pin inbox module manifest to digest image
  ZOS_IMAGE_GMAIL                 Pin gmail module manifest to digest image
  ZOS_IMAGE_GCAL                  Pin gcal module manifest to digest image
  ZOS_IMAGE_TASKS                 Pin tasks module manifest to digest image
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --edge) EDGE="$2"; EDGE_SET=1; shift 2 ;;
    --domain) DOMAIN="$2"; DOMAIN_SET=1; shift 2 ;;
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; REPO_URL_SET=1; shift 2 ;;
    --ref) GIT_REF="$2"; GIT_REF_SET=1; shift 2 ;;
    --local-source) LOCAL_SOURCE="$2"; LOCAL_SOURCE_SET=1; shift 2 ;;
    --update) UPDATE_MODE=1; shift 1 ;;
    --skip-passphrase) SKIP_PASSPHRASE=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "Error: deploy/install.sh must be run as root (use sudo)." >&2
  exit 1
fi

read_env_file_val() {
  local file="$1"
  local key="$2"
  [ -f "${file}" ] || return 0
  python3 - "${file}" "${key}" <<'PY'
import sys
from pathlib import Path

path, target = Path(sys.argv[1]), sys.argv[2]
val = ""
for raw in path.read_text(encoding="utf-8").splitlines():
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

# Reuse persisted install settings from existing .env unless explicitly overridden on CLI
# so re-runs (Step 3 passphrase, edge switch, --update) never silently move a tag-pinned
# or --local-source install onto main.
ZOS_DIR="${INSTALL_DIR}/zos"
ENV_FILE="${ZOS_DIR}/.env"
if [ -f "${ENV_FILE}" ] && command -v python3 >/dev/null 2>&1; then
  SAVED_EDGE="$(read_env_file_val "${ENV_FILE}" ZOS_INSTALL_EDGE)"
  SAVED_DOMAIN="$(read_env_file_val "${ENV_FILE}" ZOS_INSTALL_DOMAIN)"
  SAVED_REF="$(read_env_file_val "${ENV_FILE}" ZOS_INSTALL_REF)"
  SAVED_SOURCE="$(read_env_file_val "${ENV_FILE}" ZOS_INSTALL_SOURCE)"
  SAVED_REPO="$(read_env_file_val "${ENV_FILE}" ZOS_INSTALL_REPO)"

  if [ "${EDGE_SET}" -eq 0 ] && [ -n "${SAVED_EDGE}" ]; then
    EDGE="${SAVED_EDGE}"
  fi
  if [ "${DOMAIN_SET}" -eq 0 ]; then
    if [ "${EDGE_SET}" -eq 1 ] && { [ "${EDGE}" = "none" ] || [ "${EDGE}" = "exe" ]; }; then
      DOMAIN=""
    elif [ -n "${SAVED_DOMAIN}" ]; then
      DOMAIN="${SAVED_DOMAIN}"
    fi
  fi
  if [ "${REPO_URL_SET}" -eq 0 ] && [ -n "${SAVED_REPO}" ]; then
    REPO_URL="${SAVED_REPO}"
  fi
  if [ "${LOCAL_SOURCE_SET}" -eq 0 ] && [ "${GIT_REF_SET}" -eq 0 ]; then
    if [ -n "${SAVED_SOURCE}" ]; then
      LOCAL_SOURCE="${SAVED_SOURCE}"
    elif [ -n "${SAVED_REF}" ]; then
      GIT_REF="${SAVED_REF}"
    fi
  fi
fi

# Apply defaults for first-time installs (or when no prior value was persisted)
EDGE="${EDGE:-caddy}"
REPO_URL="${REPO_URL:-${ZOS_REPO_URL:-https://github.com/zamborg/lowercaseLabs.git}}"
if [ -z "${LOCAL_SOURCE}" ]; then
  GIT_REF="${GIT_REF:-${ZOS_GIT_REF:-main}}"
fi

if [ "${LOCAL_SOURCE_SET}" -eq 1 ]; then
  if [ ! -d "${LOCAL_SOURCE}" ]; then
    echo "Error: --local-source directory not found: ${LOCAL_SOURCE}" >&2
    exit 2
  fi
  LOCAL_SOURCE="$(cd "${LOCAL_SOURCE}" && pwd -P)"
fi

case "${EDGE}" in
  caddy|exe|cloudflared|none) ;;
  *) echo "Error: --edge must be one of: caddy, exe, cloudflared, none" >&2; exit 2 ;;
esac

if [ "${EDGE}" = "caddy" ] && [ -z "${DOMAIN}" ]; then
  echo "Error: --domain <hostname> is required when --edge caddy" >&2
  exit 2
fi

if [ -n "${DOMAIN}" ]; then
  if ! printf "%s" "${DOMAIN}" | grep -Eq '^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$'; then
    echo "Error: invalid --domain '${DOMAIN}'" >&2
    exit 2
  fi
fi

validate_digest_image() {
  local name="$1"
  local val="$2"
  [ -z "${val}" ] && return 0
  if ! printf "%s" "${val}" | grep -Eq '@sha256:[0-9a-f]{64}$'; then
    echo "Error: ${name} must be digest-pinned (<repo>@sha256:<64-hex>)" >&2
    exit 2
  fi
}

validate_digest_image "ZOS_IMAGE_CORE" "${ZOS_IMAGE_CORE:-}"
validate_digest_image "ZOS_IMAGE_MEMORY" "${ZOS_IMAGE_MEMORY:-}"
validate_digest_image "ZOS_IMAGE_INBOX" "${ZOS_IMAGE_INBOX:-}"
validate_digest_image "ZOS_IMAGE_GMAIL" "${ZOS_IMAGE_GMAIL:-}"
validate_digest_image "ZOS_IMAGE_GCAL" "${ZOS_IMAGE_GCAL:-}"
validate_digest_image "ZOS_IMAGE_TASKS" "${ZOS_IMAGE_TASKS:-}"

# ── 1. Install prerequisites & Docker Engine (Ubuntu 24.04) ────────────────
install_docker() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq </dev/null
  apt-get install -y -qq ca-certificates curl gnupg git openssl python3 </dev/null >/dev/null

  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "[install] Docker and Docker Compose plugin already installed."
    return 0
  fi

  echo "[install] Installing Docker Engine + Compose plugin..."
  install -m 0755 -d /etc/apt/keyrings
  if [ ! -f /etc/apt/keyrings/docker.asc ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
  fi
  . /etc/os-release
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin >/dev/null
  if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    systemctl enable --now docker
  fi
}

install_docker

# ── 2. Fetch or update ZOS source code ─────────────────────────────────────
mkdir -p "${INSTALL_DIR}"
ZOS_DIR="${INSTALL_DIR}/zos"

if [ "${UPDATE_MODE}" -eq 1 ] && [ -f "${ZOS_DIR}/.env" ] && [ -x "${ZOS_DIR}/deploy/backup/backup.sh" ]; then
  echo "[update] Taking pre-update encrypted backup..."
  ZOS_BACKUP_DIR="${ZOS_BACKUP_DIR:-/var/backups/zos}" "${ZOS_DIR}/deploy/backup/backup.sh"
fi

if [ -n "${LOCAL_SOURCE}" ]; then
  if [ -d "${LOCAL_SOURCE}" ]; then
    echo "[install] Syncing source from local directory ${LOCAL_SOURCE}..."
    if [ -d "${LOCAL_SOURCE}/zos" ]; then
      SRC_ZOS="${LOCAL_SOURCE}/zos"
    else
      SRC_ZOS="${LOCAL_SOURCE}"
    fi
    mkdir -p "${ZOS_DIR}"
    python3 - "${SRC_ZOS}" "${ZOS_DIR}" <<'PY'
import shutil, sys
from pathlib import Path

src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if src == dst:
    sys.exit(0)

skip_names = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".env", ".owner-passphrase", "backups"}
for item in src.iterdir():
    if item.name in skip_names:
        continue
    target = dst / item.name
    if item.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(item, target, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "*.pyc"))
    else:
        shutil.copy2(item, target)
PY
  elif [ "${UPDATE_MODE}" -eq 0 ] && [ -f "${ZOS_DIR}/deploy/docker-compose.yml" ]; then
    echo "[install] Saved local source ${LOCAL_SOURCE} not present; reusing installed files in ${ZOS_DIR}."
  else
    echo "Error: saved --local-source directory not found: ${LOCAL_SOURCE} (pass --local-source <path> or --ref <tag>)" >&2
    exit 1
  fi
else
  if [ ! -d "${INSTALL_DIR}/.git" ]; then
    if [ -z "$(ls -A "${INSTALL_DIR}" 2>/dev/null)" ]; then
      echo "[install] Cloning ${REPO_URL} (ref=${GIT_REF}) into ${INSTALL_DIR}..."
      git clone --depth 1 --branch "${GIT_REF}" "${REPO_URL}" "${INSTALL_DIR}" </dev/null
    else
      echo "[install] Initializing git checkout (${REPO_URL}, ref=${GIT_REF}) in ${INSTALL_DIR}..."
      git -C "${INSTALL_DIR}" init -q
      git -C "${INSTALL_DIR}" remote add origin "${REPO_URL}"
      git -C "${INSTALL_DIR}" fetch --depth 1 origin "${GIT_REF}" </dev/null
      git -C "${INSTALL_DIR}" checkout -f FETCH_HEAD
    fi
  else
    echo "[install] Updating git checkout in ${INSTALL_DIR} to ref=${GIT_REF}..."
    git -C "${INSTALL_DIR}" remote set-url origin "${REPO_URL}"
    git -C "${INSTALL_DIR}" fetch --depth 1 origin "${GIT_REF}" </dev/null
    git -C "${INSTALL_DIR}" checkout -f FETCH_HEAD
  fi
fi

if [ ! -f "${ZOS_DIR}/deploy/docker-compose.yml" ]; then
  echo "Error: ${ZOS_DIR}/deploy/docker-compose.yml not found." >&2
  exit 1
fi

cd "${ZOS_DIR}"

# ── 3. Generate secrets into .env (0600, never echoed) ─────────────────────
ENV_FILE="${ZOS_DIR}/.env"
touch "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

get_env_val() {
  read_env_file_val "${ENV_FILE}" "$1"
}

set_env_val() {
  local key="$1"
  local val="$2"
  local quote_single="${3:-0}"
  python3 - "${ENV_FILE}" "${key}" "${val}" "${quote_single}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key, val, q = sys.argv[2], sys.argv[3], sys.argv[4] == "1"
rendered = f"{key}='{val}'" if q else f"{key}={val}"
lines = path.read_text(encoding="utf-8").splitlines()
replaced = False
out = []
for line in lines:
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        k = stripped.split("=", 1)[0].strip()
        if k == key:
            out.append(rendered)
            replaced = True
            continue
    out.append(line)
if not replaced:
    out.append(rendered)
path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
  chmod 600 "${ENV_FILE}"
}

gen_b64_key() {
  python3 -c "import os, base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
}

gen_pw() {
  python3 -c "import secrets; print(secrets.token_urlsafe(32))"
}

ensure_env_default() {
  local key="$1"
  local fallback="$2"
  local cur
  cur="$(get_env_val "${key}")"
  if [ -z "${cur}" ]; then
    set_env_val "${key}" "${fallback}"
  fi
}

ensure_env_default "ZOS_PORT" "8787"
EXE_VM_NAME="$(hostname -s 2>/dev/null || hostname)"

if [ -n "${DOMAIN}" ]; then
  BASE_URL="https://${DOMAIN}"
elif [ "${EDGE}" = "exe" ]; then
  BASE_URL="https://${EXE_VM_NAME}.exe.xyz"
elif [ "${EDGE}" = "none" ] && [ "${EDGE_SET}" -eq 1 ]; then
  BASE_URL="http://localhost:$(get_env_val ZOS_PORT)"
else
  BASE_URL="$(get_env_val ZOS_BASE_URL)"
  BASE_URL="${BASE_URL:-http://localhost:$(get_env_val ZOS_PORT)}"
fi

set_env_val "ZOS_BASE_URL" "${BASE_URL}"
set_env_val "ZOS_INSTALL_EDGE" "${EDGE}"
set_env_val "ZOS_INSTALL_DOMAIN" "${DOMAIN}"
if [ -n "${LOCAL_SOURCE}" ]; then
  set_env_val "ZOS_INSTALL_SOURCE" "${LOCAL_SOURCE}"
  set_env_val "ZOS_INSTALL_REF" ""
else
  set_env_val "ZOS_INSTALL_REF" "${GIT_REF}"
  set_env_val "ZOS_INSTALL_SOURCE" ""
  set_env_val "ZOS_INSTALL_REPO" "${REPO_URL}"
fi
ensure_env_default "ZOS_LOG_LEVEL" "INFO"
ensure_env_default "POSTGRES_USER" "zos"
ensure_env_default "POSTGRES_DB" "zos"
ensure_env_default "ZOS_DB_USER" "zos_app"
ensure_env_default "ZOS_BACKUP_DIR" "/var/backups/zos"

[ -n "$(get_env_val POSTGRES_PASSWORD)" ] || set_env_val "POSTGRES_PASSWORD" "$(gen_pw)"
[ -n "$(get_env_val ZOS_DB_PASSWORD)" ] || set_env_val "ZOS_DB_PASSWORD" "$(gen_pw)"
[ -n "$(get_env_val ZOS_SECRET_KEY)" ] || set_env_val "ZOS_SECRET_KEY" "$(gen_b64_key)"
[ -n "$(get_env_val ZOS_BACKUP_KEY)" ] || set_env_val "ZOS_BACKUP_KEY" "$(gen_b64_key)"

# ── 4. Prepare edge and optional digest-pinned image overrides ─────────────
COMPOSE_ARGS=(--env-file "${ENV_FILE}" -f "${ZOS_DIR}/deploy/docker-compose.yml")

if [ -n "${ZOS_IMAGE_CORE:-}" ]; then
  set_env_val "ZOS_IMAGE_CORE" "${ZOS_IMAGE_CORE}"
  MOD_CAT_OUT="${ZOS_DIR}/deploy/modules-catalog"
  rm -rf "${MOD_CAT_OUT}"
  cp -R "${ZOS_DIR}/catalog/modules" "${MOD_CAT_OUT}"
  python3 - "${MOD_CAT_OUT}" <<'PY'
import os, re, sys
from pathlib import Path

cat_dir = Path(sys.argv[1])
for slug, env_name in (
    ("memory", "ZOS_IMAGE_MEMORY"),
    ("inbox", "ZOS_IMAGE_INBOX"),
    ("gmail", "ZOS_IMAGE_GMAIL"),
    ("gcal", "ZOS_IMAGE_GCAL"),
    ("tasks", "ZOS_IMAGE_TASKS"),
):
    img = os.environ.get(env_name, "").strip()
    if not img:
        continue
    toml_path = cat_dir / f"{slug}.toml"
    if toml_path.is_file():
        text = toml_path.read_text(encoding="utf-8")
        text = re.sub(r'^build\s*=\s*"[^"]*"\s*$', f'image = "{img}"', text, flags=re.MULTILINE)
        toml_path.write_text(text, encoding="utf-8")
PY
  COMPOSE_ARGS+=(-f "${ZOS_DIR}/deploy/docker-compose.images.yml")
fi

if [ "${EDGE}" = "caddy" ]; then
  sed "s/__DOMAIN__/${DOMAIN}/g" "${ZOS_DIR}/deploy/caddy/Caddyfile.template" > "${ZOS_DIR}/deploy/caddy/Caddyfile"
  COMPOSE_ARGS+=(-f "${ZOS_DIR}/deploy/docker-compose.caddy.yml")
elif [ "${EDGE}" = "exe" ]; then
  COMPOSE_ARGS+=(-f "${ZOS_DIR}/deploy/docker-compose.filter.yml" -f "${ZOS_DIR}/deploy/docker-compose.exe.yml")
elif [ "${EDGE}" = "cloudflared" ]; then
  if [ -n "$(get_env_val CLOUDFLARED_TOKEN)" ]; then
    COMPOSE_ARGS+=(-f "${ZOS_DIR}/deploy/docker-compose.filter.yml" -f "${ZOS_DIR}/deploy/docker-compose.cloudflared.yml")
  else
    echo "[install] Note: CLOUDFLARED_TOKEN is not set in .env; configure host cloudflared using deploy/cloudflared/config.yml.example (http://127.0.0.1:8787)."
  fi
fi

# ── 5. Start Postgres, ensure zos_app role, start ZOS ──────────────────────
echo "[install] Starting Postgres..."
docker compose "${COMPOSE_ARGS[@]}" up -d --wait postgres </dev/null
PG_SUPER_USER="$(get_env_val POSTGRES_USER)"
PG_SUPER_PW="$(get_env_val POSTGRES_PASSWORD)"
PG_DB_NAME="$(get_env_val POSTGRES_DB)"
for _ in $(seq 1 30); do
  if docker compose "${COMPOSE_ARGS[@]}" exec -T -e "PGPASSWORD=${PG_SUPER_PW}" postgres \
      psql -U "${PG_SUPER_USER}" -d "${PG_DB_NAME}" -c "SELECT 1" </dev/null >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
unset PG_SUPER_PW

# Ensure non-superuser zos_app role & extensions schema exist (idempotent for both fresh and existing volumes)
"${ZOS_DIR}/deploy/postgres/migrate-existing-to-app-role.sh" --env-file "${ENV_FILE}" </dev/null

echo "[install] Building/starting ZOS stack..."
if [ -n "${ZOS_IMAGE_CORE:-}" ]; then
  docker compose "${COMPOSE_ARGS[@]}" pull zos </dev/null
  docker compose "${COMPOSE_ARGS[@]}" up -d </dev/null
else
  docker compose "${COMPOSE_ARGS[@]}" up -d --build </dev/null
fi

for _ in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$(get_env_val ZOS_PORT)/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# ── 6. Initialize ZOS principal/space & install default modules ────────────
docker compose "${COMPOSE_ARGS[@]}" exec -T zos uv run --no-sync zos init --principal owner </dev/null

install_or_upgrade_module() {
  local slug="$1"
  local listed
  listed="$(docker compose "${COMPOSE_ARGS[@]}" exec -T zos uv run --no-sync zos module list </dev/null 2>/dev/null || true)"
  if printf "%s\n" "${listed}" | grep -Eq "^${slug}[[:space:]]"; then
    if [ "${UPDATE_MODE}" -eq 1 ]; then
      echo "[update] Upgrading module '${slug}'..."
      docker compose "${COMPOSE_ARGS[@]}" exec -T zos uv run --no-sync zos module upgrade "${slug}" </dev/null
    else
      echo "[install] Module '${slug}' already installed."
    fi
  else
    echo "[install] Installing module '${slug}'..."
    docker compose "${COMPOSE_ARGS[@]}" exec -T zos uv run --no-sync zos module install "${slug}" </dev/null
  fi
}

install_or_upgrade_module memory
install_or_upgrade_module inbox

if [ "${UPDATE_MODE}" -eq 1 ]; then
  # Upgrade any other user-installed modules (e.g. gmail, gcal) while preserving their data & secrets
  ALL_INSTALLED="$(docker compose "${COMPOSE_ARGS[@]}" exec -T zos uv run --no-sync zos module list </dev/null 2>/dev/null | awk '{print $1}' || true)"
  for extra_slug in ${ALL_INSTALLED}; do
    case "${extra_slug}" in
      ""|No|memory|inbox) continue ;;
      *)
        echo "[update] Upgrading installed module '${extra_slug}'..."
        docker compose "${COMPOSE_ARGS[@]}" exec -T zos uv run --no-sync zos module upgrade "${extra_slug}" </dev/null
        ;;
    esac
  done
fi

# Restart zos so in-process UpstreamManager loads newly installed/upgraded modules
docker compose "${COMPOSE_ARGS[@]}" restart zos </dev/null
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$(get_env_val ZOS_PORT)/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if [ "${EDGE}" = "exe" ]; then
  for _ in $(seq 1 20); do
    if curl -sf "http://127.0.0.1:8000/.well-known/oauth-protected-resource" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

# ── 7. Interactive owner passphrase prompt (when not already configured) ───
if [ -z "$(get_env_val ZOS_OWNER_PASSWORD_HASH)" ] && [ "${SKIP_PASSPHRASE}" -eq 0 ]; then
  echo ""
  echo "=== Owner Passphrase Setup ==="
  echo "Set the passphrase used to sign into ${BASE_URL}/admin and approve OAuth clients."
  if [ -t 0 ] && [ -r /dev/tty ]; then
    INPUT_FD="/dev/tty"
  else
    INPUT_FD="/dev/stdin"
  fi
  while true; do
    if [ "${INPUT_FD}" = "/dev/tty" ]; then
      printf "Owner passphrase: " > /dev/tty
      IFS= read -r -s PASS1 < /dev/tty
      printf "\nConfirm passphrase: " > /dev/tty
      IFS= read -r -s PASS2 < /dev/tty
      printf "\n" > /dev/tty
    else
      IFS= read -r PASS1 || PASS1=""
      IFS= read -r PASS2 || PASS2=""
    fi
    if [ -z "${PASS1}" ]; then
      echo "Passphrase cannot be empty." >&2
      [ "${INPUT_FD}" = "/dev/tty" ] || exit 1
      continue
    fi
    if [ "${PASS1}" != "${PASS2}" ]; then
      echo "Passphrases did not match. Try again." >&2
      [ "${INPUT_FD}" = "/dev/tty" ] || exit 1
      continue
    fi
    break
  done
  HASH="$(printf "%s" "${PASS1}" | docker compose "${COMPOSE_ARGS[@]}" exec -T zos uv run --no-sync python -c "import sys; from argon2 import PasswordHasher; print(PasswordHasher().hash(sys.stdin.read()))")"
  unset PASS1 PASS2
  set_env_val "ZOS_OWNER_PASSWORD_HASH" "${HASH}" 1
  unset HASH
  docker compose "${COMPOSE_ARGS[@]}" up -d --force-recreate zos
  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$(get_env_val ZOS_PORT)/healthz" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  echo "[install] Owner passphrase saved (hashed with Argon2id in ${ENV_FILE})."
fi

# ── 8. Install nightly encrypted backup systemd timer ──────────────────────
mkdir -p "$(get_env_val ZOS_BACKUP_DIR)"
chmod 700 "$(get_env_val ZOS_BACKUP_DIR)"
if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
  sed "s|__ZOS_DIR__|${ZOS_DIR}|g" "${ZOS_DIR}/deploy/systemd/zos-backup.service" > /etc/systemd/system/zos-backup.service
  cp "${ZOS_DIR}/deploy/systemd/zos-backup.timer" /etc/systemd/system/zos-backup.timer
  systemctl daemon-reload
  systemctl enable --now zos-backup.timer
  echo "[install] Enabled systemd timer: zos-backup.timer (nightly encrypted backups -> $(get_env_val ZOS_BACKUP_DIR))."
fi

# ── 9. Summary & next steps (zero secrets printed) ─────────────────────────
echo ""
echo "=== ZOS Ready ==="
echo "URL:         ${BASE_URL}"
echo "Admin UI:    ${BASE_URL}/admin"
echo "MCP URL:     ${BASE_URL}/mcp"
echo "Install dir: ${ZOS_DIR}"
echo "Backups:     $(get_env_val ZOS_BACKUP_DIR) (nightly via zos-backup.timer)"
if [ "${EDGE}" = "exe" ]; then
  echo "Make public: ssh exe.dev share set-public ${EXE_VM_NAME}   (run from your local machine, not on this VM)"
fi
if [ -z "$(get_env_val ZOS_OWNER_PASSWORD_HASH)" ]; then
  echo ""
  echo "Owner passphrase NOT set — run Step 3 (sudo bash ${ZOS_DIR}/deploy/install.sh) interactively before signing into /admin."
fi
echo ""
echo "Next steps:"
if [ -z "$(get_env_val ZOS_OWNER_PASSWORD_HASH)" ]; then
  echo "  1. Run Step 3 interactively to set your owner passphrase: sudo bash ${ZOS_DIR}/deploy/install.sh"
fi
if [ "${EDGE}" = "exe" ]; then
  echo "  - Expose your exe.dev VM publicly from your local machine: ssh exe.dev share set-public ${EXE_VM_NAME}"
fi
echo "  - Open ${BASE_URL}/admin and sign in with your owner passphrase."
echo "  - Connect Claude / ChatGPT / Codex to ${BASE_URL}/mcp (or mint a bearer token in /admin/clients)."
echo "  - Optional off-box backups: set ZOS_BACKUP_POST_HOOK in ${ENV_FILE} (see HOSTING.md)."
echo "  - Updates: re-run this script with --update."
