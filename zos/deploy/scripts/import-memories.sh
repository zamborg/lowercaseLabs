#!/usr/bin/env bash
# Host wrapper that discovers local agent memories (Claude Code & Codex) and streams them
# as JSONL into the containerized `memory` module (`zos-mod-memory`).
# Prints counts only (never memory content).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MEMORY_CONTAINER="${ZOS_MEMORY_CONTAINER:-zos-mod-memory}"
cd "${ZOS_DIR}"

SOURCES=("$@")
if [ "${#SOURCES[@]}" -eq 0 ]; then
  SOURCES=("claude-code" "codex")
fi

for src in "${SOURCES[@]}"; do
  printf "[%s] " "${src}"
  uv run zos memory export "${src}" \
    | docker exec -i "${MEMORY_CONTAINER}" \
      uv run --no-sync python -m zos_memory.import_cli --source "${src}" -
done
