#!/usr/bin/env bash
set -euo pipefail

# Install the ZOS Web browser device (`zos-device` + pinned `@playwright/mcp` +
# Playwright's bundled Chromium) into ~/.zos-web with its own LaunchAgent (`com.zos.web`).
# Completely isolated from `~/.zos-actuator` and from personal Chrome/Safari profiles.
#
# Options:
#   --node <path>    Explicit Node.js (>=20) binary path (otherwise auto-detected incl. nvm)
#   --headless       Configure Playwright MCP to launch headless (default: headed)
#   -h, --help       Show usage

NODE_BIN_ARG="${ZOS_WEB_NODE_BIN:-}"
HEADLESS=false

usage() {
  cat <<'EOF'
Usage: install-web.sh [--node <path>] [--headless]

Installs the ZOS `web` browser device into ~/.zos-web:
  - Dedicated venv with `zos-device` (never touches ~/.zos-actuator)
  - Pinned `@playwright/mcp` (version from devices/web/VERSION)
  - Playwright bundled Chromium in ~/.zos-web/browsers
  - Isolated persistent browser profile in ~/.zos-web/profile
  - LaunchAgent ~/Library/LaunchAgents/com.zos.web.plist
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --node)
      if [ "$#" -lt 2 ]; then
        echo "Error: --node requires a path argument" >&2
        exit 2
      fi
      NODE_BIN_ARG="$2"
      shift 2
      ;;
    --headless)
      HEADLESS=true
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEVICE_SRC="${ZOS_ROOT}/device"
WEB_DIR="${ZOS_ROOT}/devices/web"
VERSION_FILE="${WEB_DIR}/VERSION"
PLIST_TEMPLATE="${WEB_DIR}/com.zos.web.plist"
CONFIG_EXAMPLE="${WEB_DIR}/config.toml.example"

INSTALL_DIR="${ZOS_WEB_INSTALL_DIR:-${HOME}/.zos-web}"
LAUNCH_AGENTS="${ZOS_WEB_LAUNCH_AGENTS:-${HOME}/Library/LaunchAgents}"
LOG_DIR="${ZOS_WEB_LOG_DIR:-${HOME}/Library/Logs}"
SKIP_INSTALL="${ZOS_WEB_SKIP_INSTALL:-0}"
MIN_NODE_MAJOR=20

if [ ! -f "${VERSION_FILE}" ]; then
  echo "Error: missing version pin file: ${VERSION_FILE}" >&2
  exit 1
fi
PLAYWRIGHT_MCP_VERSION="$(tr -d '[:space:]' < "${VERSION_FILE}")"
if [ -z "${PLAYWRIGHT_MCP_VERSION}" ]; then
  echo "Error: empty version in ${VERSION_FILE}" >&2
  exit 1
fi

resolve_realpath() {
  python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

resolve_node() {
  if [ -n "${NODE_BIN_ARG}" ]; then
    if [ ! -x "${NODE_BIN_ARG}" ]; then
      echo "Error: specified --node binary is not executable: ${NODE_BIN_ARG}" >&2
      exit 1
    fi
    resolve_realpath "${NODE_BIN_ARG}"
    return 0
  fi

  local found=""
  if found="$(command -v node 2>/dev/null || true)" && [ -n "${found}" ] && [ -x "${found}" ]; then
    resolve_realpath "${found}"
    return 0
  fi

  # Fall back to nvm (launchd and non-interactive shells don't source nvm) and Homebrew.
  local nvm_dir="${NVM_DIR:-${HOME}/.nvm}"
  local best=""
  if [ -d "${nvm_dir}/versions/node" ]; then
    best="$(python3 - "${nvm_dir}/versions/node" <<'PY'
import os, re, sys
root = sys.argv[1]
candidates = []
for name in os.listdir(root):
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)$", name)
    bin_path = os.path.join(root, name, "bin", "node")
    if m and os.access(bin_path, os.X_OK):
        candidates.append((tuple(int(x) for x in m.groups()), bin_path))
if candidates:
    candidates.sort()
    print(candidates[-1][1], end="")
PY
)"
    if [ -n "${best}" ]; then
      resolve_realpath "${best}"
      return 0
    fi
  fi

  for candidate in /opt/homebrew/bin/node /usr/local/bin/node; do
    if [ -x "${candidate}" ]; then
      resolve_realpath "${candidate}"
      return 0
    fi
  done

  echo "Error: Node.js >= ${MIN_NODE_MAJOR} is required but 'node' was not found on PATH, in nvm, or in Homebrew." >&2
  echo "Install Node.js >= ${MIN_NODE_MAJOR} (e.g. 'brew install node' or 'nvm install 22') or pass --node /path/to/node." >&2
  exit 1
}

NODE_BIN="$(resolve_node)"
NODE_DIR="$(dirname "${NODE_BIN}")"
NODE_VERSION="$("${NODE_BIN}" -v 2>/dev/null || true)"
NODE_MAJOR="$(printf "%s" "${NODE_VERSION}" | sed -E 's/^v?([0-9]+).*/\1/')"
if ! printf "%s" "${NODE_MAJOR}" | grep -Eq '^[0-9]+$' || [ "${NODE_MAJOR}" -lt "${MIN_NODE_MAJOR}" ]; then
  echo "Error: Node.js >= ${MIN_NODE_MAJOR} is required by @playwright/mcp@${PLAYWRIGHT_MCP_VERSION}; found '${NODE_VERSION:-unknown}' at ${NODE_BIN}." >&2
  exit 1
fi

NPM_BIN=""
if [ -x "${NODE_DIR}/npm" ]; then
  NPM_BIN="${NODE_DIR}/npm"
elif NPM_BIN="$(command -v npm 2>/dev/null || true)" && [ -n "${NPM_BIN}" ]; then
  :
elif [ "${SKIP_INSTALL}" != "1" ]; then
  echo "Error: npm was not found next to ${NODE_BIN} or on PATH." >&2
  exit 1
fi

echo "==> Installing ZOS web device (@playwright/mcp@${PLAYWRIGHT_MCP_VERSION}) into ${INSTALL_DIR}"
echo "    Node.js: ${NODE_VERSION} (${NODE_BIN})"

mkdir -p "${INSTALL_DIR}" "${INSTALL_DIR}/profile" "${INSTALL_DIR}/output" "${INSTALL_DIR}/browsers" "${LAUNCH_AGENTS}" "${LOG_DIR}"
chmod 700 "${INSTALL_DIR}/profile" "${INSTALL_DIR}/output"

if [ "${SKIP_INSTALL}" != "1" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required. Install via: brew install uv" >&2
    exit 1
  fi

  echo "==> Installing zos-device into ${INSTALL_DIR}/venv"
  uv venv --python 3.12 "${INSTALL_DIR}/venv" --allow-existing
  uv pip install --python "${INSTALL_DIR}/venv/bin/python" --upgrade "${DEVICE_SRC}"

  echo "==> Installing @playwright/mcp@${PLAYWRIGHT_MCP_VERSION} into ${INSTALL_DIR}"
  PATH="${NODE_DIR}:${PATH}" "${NPM_BIN}" install \
    --prefix "${INSTALL_DIR}" \
    --no-audit \
    --no-fund \
    --save-exact \
    "@playwright/mcp@${PLAYWRIGHT_MCP_VERSION}"

  echo "==> Installing Playwright bundled Chromium into ${INSTALL_DIR}/browsers"
  PLAYWRIGHT_BROWSERS_PATH="${INSTALL_DIR}/browsers" \
    "${NODE_BIN}" "${INSTALL_DIR}/node_modules/@playwright/mcp/cli.js" install-browser chromium
fi

# Render LaunchAgent plist
sed \
  -e "s|{{INSTALL_DIR}}|${INSTALL_DIR}|g" \
  -e "s|{{LOG_DIR}}|${LOG_DIR}|g" \
  -e "s|{{NODE_BIN}}|${NODE_BIN}|g" \
  -e "s|{{NODE_DIR}}|${NODE_DIR}|g" \
  "${PLIST_TEMPLATE}" > "${LAUNCH_AGENTS}/com.zos.web.plist"

if [ ! -f "${INSTALL_DIR}/config.toml" ]; then
  cp "${CONFIG_EXAMPLE}" "${INSTALL_DIR}/config.toml"
  chmod 600 "${INSTALL_DIR}/config.toml"
  echo "==> Created template ${INSTALL_DIR}/config.toml (chmod 600)"
fi

if [ ! -f "${INSTALL_DIR}/playwright-mcp.json" ]; then
  cat > "${INSTALL_DIR}/playwright-mcp.json" <<EOF
{
  "browser": {
    "launchOptions": {
      "headless": ${HEADLESS}
    }
  },
  "allowUnrestrictedFileAccess": false
}
EOF
  chmod 600 "${INSTALL_DIR}/playwright-mcp.json"
  echo "==> Created ${INSTALL_DIR}/playwright-mcp.json (headless=${HEADLESS})"
fi

echo ""
echo "==> Installed LaunchAgent:"
echo "    - ${LAUNCH_AGENTS}/com.zos.web.plist"
echo "    - Profile dir:  ${INSTALL_DIR}/profile (isolated from personal Chrome/Safari)"
echo "    - Browsers dir: ${INSTALL_DIR}/browsers (Playwright bundled Chromium)"
echo "    - Output dir:   ${INSTALL_DIR}/output (local file access root)"
echo ""
echo "==> Next steps:"
echo "    1. Register the 'web' device on your ZOS host:"
echo "       - Open https://<your-zos-host>/admin/modules -> 'Runs on your machine' -> add slug: web"
echo "       - (or on the ZOS host: zos device add web)"
echo "    2. Put the WebSocket URL and one-time token into ${INSTALL_DIR}/config.toml (chmod 600)."
echo "    3. Start the LaunchAgent:"
echo "       launchctl bootstrap gui/\$(id -u) ${LAUNCH_AGENTS}/com.zos.web.plist || true"
echo "    4. The headed Chromium window opens on your logged-in desktop when an agent uses it."
echo "       Sign in manually inside that window; logins persist in ${INSTALL_DIR}/profile."
echo "       To switch to headless later, set \"headless\": true in ${INSTALL_DIR}/playwright-mcp.json and restart com.zos.web."
