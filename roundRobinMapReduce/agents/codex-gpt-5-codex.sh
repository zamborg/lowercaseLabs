#!/usr/bin/env bash
set -euo pipefail

export RUFFLE_CODEX_MODEL="${RUFFLE_CODEX_MODEL:-gpt-5-codex}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/codex.sh" "$@"
