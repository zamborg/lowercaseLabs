#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${1:?task.md path required}"
CLAUDE_STATE_DIR="${HOME:-/home/claude}/.claude"
CLAUDE_STATE_FILE="${HOME:-/home/claude}/.claude.json"

if [[ "${RUFFLE_CLAUDE_DEMOTED:-0}" != "1" && "$(id -u)" = "0" ]] && id claude >/dev/null 2>&1; then
  export RUFFLE_CLAUDE_DEMOTED=1
  exec su -m -s /bin/bash claude -c "cd $(printf '%q' "$PWD") && exec $(printf '%q' "$0") $(printf '%q' "$TASK_FILE")"
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not found in PATH" >&2
  exit 1
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" && ! -d "$CLAUDE_STATE_DIR" && ! -f "$CLAUDE_STATE_FILE" ]]; then
  echo "Missing Claude credentials. Set ANTHROPIC_API_KEY/CLAUDE_CODE_OAUTH_TOKEN or mount Claude auth state." >&2
  exit 1
fi

PROMPT="$(cat <<EOF
You are one worker in a MapReduce-style coding run.

Read the task description from $(basename "$TASK_FILE") in the current directory and execute it end to end.

Rules:
- Work only in the current workspace.
- If a directory named workspaces exists, you may inspect workspaces/mapper-* as needed.
- Create or update files directly in this workspace.
- Write a concise RESULT.md summarizing what you changed or selected.
- Do not wait for confirmation. Finish the task and exit.
EOF
)"

ARGS=(
  -p
  "$PROMPT"
  --output-format text
  --permission-mode bypassPermissions
  --dangerously-skip-permissions
)

if [[ -n "${RUFFLE_CLAUDE_MODEL:-}" ]]; then
  ARGS+=(--model "$RUFFLE_CLAUDE_MODEL")
fi

claude "${ARGS[@]}"
