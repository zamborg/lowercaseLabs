#!/usr/bin/env bash
# Host wrapper that discovers local agent memories (Claude Code & Codex) and streams them
# as JSONL into the containerized ZOS instance via `docker compose exec -T`.
# Prints counts only (never memory content).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ZOS_DIR}"

SOURCES=("$@")
if [ "${#SOURCES[@]}" -eq 0 ]; then
  SOURCES=("claude-code" "codex")
fi

for src in "${SOURCES[@]}"; do
  printf "[%s] " "${src}"
  uv run zos memory export "${src}" \
    | docker compose --env-file .env -f deploy/docker-compose.yml exec -T zos \
      uv run --no-sync zos memory import - --source-label "${src}"
done
