#!/usr/bin/env bash
set -euo pipefail

# Codex CLI's coding-tuned GPT-5.2 model is exposed as gpt-5.2-codex.
export RUFFLE_CODEX_MODEL="${RUFFLE_CODEX_MODEL:-gpt-5.2-codex}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/codex.sh" "$@"
