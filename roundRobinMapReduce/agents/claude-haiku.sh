#!/usr/bin/env bash
set -euo pipefail

export RUFFLE_CLAUDE_MODEL="${RUFFLE_CLAUDE_MODEL:-haiku}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/claude.sh" "$@"
