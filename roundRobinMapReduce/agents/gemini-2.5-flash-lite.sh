#!/usr/bin/env bash
set -euo pipefail

export RUFFLE_GEMINI_MODEL="${RUFFLE_GEMINI_MODEL:-gemini-2.5-flash-lite}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gemini.sh" "$@"
