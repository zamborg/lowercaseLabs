#!/usr/bin/env bash
set -euo pipefail

# Install ZOS Mac actuator into a fixed venv (~/.zos-actuator/venv) and install launchd agents.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ACTUATOR_SRC="${ZOS_ROOT}/actuator"
INSTALL_DIR="${HOME}/.zos-actuator"
VENV_DIR="${INSTALL_DIR}/venv"
LAUNCH_AGENTS="${HOME}/Library/LaunchAgents"
TMUX_BIN="$(command -v tmux || echo /opt/homebrew/bin/tmux)"

echo "==> Installing zos-actuator into ${VENV_DIR}"
mkdir -p "${INSTALL_DIR}" "${LAUNCH_AGENTS}" "${HOME}/Library/Logs"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required. Install via: brew install uv" >&2
  exit 1
fi

uv venv --python 3.12 "${VENV_DIR}" --allow-existing
uv pip install --python "${VENV_DIR}/bin/python" --upgrade "${ACTUATOR_SRC}"

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
# Generate token with: zos actuator add mac
url = "ws://127.0.0.1:8787/actuator"
token = "REPLACE_WITH_ZOS_ACTUATOR_TOKEN"
EOF
  chmod 600 "${INSTALL_DIR}/config.toml"
  echo "==> Created template ${INSTALL_DIR}/config.toml (edit token before loading agent)"
fi

echo ""
echo "==> Installed LaunchAgents:"
echo "    - ${LAUNCH_AGENTS}/com.zos.tmux.plist"
echo "    - ${LAUNCH_AGENTS}/com.zos.actuator.plist"
echo ""
echo "==> IMPORTANT: Grant Full Disk Access for mac_messages_search:"
echo "    System Settings -> Privacy & Security -> Full Disk Access -> add:"
echo "    ${RESOLVED_PYTHON}"
echo "    (Note: re-grant Full Disk Access if you upgrade Python.)"
echo ""
echo "==> To start agents now (after setting token in ${INSTALL_DIR}/config.toml):"
echo "    launchctl bootstrap gui/$(id -u) ${LAUNCH_AGENTS}/com.zos.tmux.plist || true"
echo "    launchctl bootstrap gui/$(id -u) ${LAUNCH_AGENTS}/com.zos.actuator.plist || true"
