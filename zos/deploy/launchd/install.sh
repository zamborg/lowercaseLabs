#!/usr/bin/env bash
set -euo pipefail

# Install ZOS Mac device (`zos-device` bridge + `zos-mac` stdio MCP server) with a
# dedicated Python 3.12 interpreter copy (~/.zos-actuator/python/bin/python3.12)
# and venv (~/.zos-actuator/venv), plus LaunchAgents for isolated tmux and com.zos.actuator.
#
# Pass --upgrade-python to refresh ~/.zos-actuator/python/ from uv's Python 3.12
# (requires re-granting Full Disk Access in macOS System Settings afterward).

UPGRADE_PYTHON=false
for arg in "$@"; do
  case "${arg}" in
    --upgrade-python) UPGRADE_PYTHON=true ;;
    *)
      echo "Usage: $0 [--upgrade-python]" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEVICE_SRC="${ZOS_ROOT}/device"
MAC_SRC="${ZOS_ROOT}/devices/mac"
INSTALL_DIR="${HOME}/.zos-actuator"
PYTHON_DIR="${INSTALL_DIR}/python"
DEDICATED_PY="${PYTHON_DIR}/bin/python3.12"
VENV_DIR="${INSTALL_DIR}/venv"
LAUNCH_AGENTS="${HOME}/Library/LaunchAgents"
TMUX_BIN="$(command -v tmux || echo /opt/homebrew/bin/tmux)"

echo "==> Installing zos-device + zos-mac into ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}" "${LAUNCH_AGENTS}" "${HOME}/Library/Logs"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required. Install via: brew install uv" >&2
  exit 1
fi

# Provision a dedicated copy of uv's CPython 3.12 directory so macOS Full Disk
# Access is scoped strictly to the actuator interpreter (not uv's shared Python).
if [ ! -x "${DEDICATED_PY}" ] || [ "${UPGRADE_PYTHON}" = true ]; then
  echo "==> Provisioning dedicated Python 3.12 at ${PYTHON_DIR}"
  uv python install 3.12
  SRC_PY="$(uv python find 3.12)"
  SRC_PY_REAL="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${SRC_PY}")"
  SRC_PY_DIR="$(dirname "$(dirname "${SRC_PY_REAL}")")"

  TMP_PY_DIR="${INSTALL_DIR}/.python.tmp.$$"
  rm -rf "${TMP_PY_DIR}"
  # Copy the entire Python directory because bin/python3.12 links
  # @executable_path/../lib/libpython3.12.dylib.
  cp -a "${SRC_PY_DIR}" "${TMP_PY_DIR}"
  rm -rf "${PYTHON_DIR}"
  mv "${TMP_PY_DIR}" "${PYTHON_DIR}"
fi

uv venv --python "${DEDICATED_PY}" "${VENV_DIR}" --allow-existing
uv pip install --python "${VENV_DIR}/bin/python" --upgrade "${DEVICE_SRC}" "${MAC_SRC}"

RESOLVED_PYTHON="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${VENV_DIR}/bin/python")"

render_plist() {
  local src="$1"
  local dst="$2"
  sed \
    -e "s|{{HOME}}|${HOME}|g" \
    -e "s|{{TMUX_BIN}}|${TMUX_BIN}|g" \
    "${src}" > "${dst}"
}

render_plist "${SCRIPT_DIR}/com.zos.tmux.plist" "${LAUNCH_AGENTS}/com.zos.tmux.plist"
render_plist "${SCRIPT_DIR}/com.zos.actuator.plist" "${LAUNCH_AGENTS}/com.zos.actuator.plist"

if [ ! -f "${INSTALL_DIR}/config.toml" ]; then
  cat > "${INSTALL_DIR}/config.toml" <<'EOF'
# Add a device in /admin/modules ("Runs on your machine") or run: zos device add mac
url = "wss://<your-zos-host>/actuator"
token = "REPLACE_WITH_ZOS_DEVICE_TOKEN"
EOF
  chmod 600 "${INSTALL_DIR}/config.toml"
  echo "==> Created template ${INSTALL_DIR}/config.toml (edit url and token before loading agent)"
fi

echo ""
echo "==> Installed LaunchAgents:"
echo "    - ${LAUNCH_AGENTS}/com.zos.tmux.plist"
echo "    - ${LAUNCH_AGENTS}/com.zos.actuator.plist"
echo ""
echo "==> IMPORTANT: Grant macOS permissions to the dedicated interpreter:"
echo "    1. Full Disk Access (reading Messages & Contacts):"
echo "       System Settings -> Privacy & Security -> Full Disk Access -> add:"
echo "       ${RESOLVED_PYTHON}"
echo "       (Dedicated to zos-actuator; re-grant only if you pass --upgrade-python.)"
echo "    2. Automation -> Messages (needed for mac_messages_send):"
echo "       macOS prompts on the first send (\"python3.12\" wants access to control \"Messages\") — click OK."
echo "       If you dismissed it, enable it manually at:"
echo "       System Settings -> Privacy & Security -> Automation -> python3.12 -> Messages"
echo ""
echo "==> To start agents now (after setting token in ${INSTALL_DIR}/config.toml):"
echo "    launchctl bootstrap gui/$(id -u) ${LAUNCH_AGENTS}/com.zos.tmux.plist || true"
echo "    launchctl bootstrap gui/$(id -u) ${LAUNCH_AGENTS}/com.zos.actuator.plist || true"
