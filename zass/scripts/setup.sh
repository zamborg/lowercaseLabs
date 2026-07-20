#!/usr/bin/env bash
# zass setup: install server deps and verify the MCP server boots.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }

echo "==> Syncing zass-mcp dependencies"
uv sync --project mcp/zass

echo "==> Running store tests"
uv run --project mcp/zass pytest mcp/zass/tests -q

echo "==> Verifying server entrypoint imports"
uv run --project mcp/zass python -c "from zass_mcp.server import mcp; print('zass-mcp OK:', mcp.name)"

cat <<'EOF'

zass is ready.
  - Run your agent in this directory (claude-code picks up .mcp.json automatically).
  - Try /brief, /triage, /capture, /weekly-review.
  - Wire Gmail/Calendar: see connectors/README.md
  - macOS will ask once for Automation permission when Messages/Contacts
    tools are first used — approve the prompt.
EOF
