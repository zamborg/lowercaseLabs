#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${1:?task.md path required}"
GEMINI_SETTINGS_DIR="${HOME:-/root}/.gemini"
GEMINI_SETTINGS_FILE="$GEMINI_SETTINGS_DIR/settings.json"
GEMINI_OAUTH_FILE="$GEMINI_SETTINGS_DIR/oauth_creds.json"

if ! command -v gemini >/dev/null 2>&1; then
  echo "gemini CLI not found in PATH" >&2
  exit 1
fi

if [[ -n "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  export GEMINI_API_KEY="$GOOGLE_API_KEY"
fi

mkdir -p "$GEMINI_SETTINGS_DIR"

if [[ -z "${GEMINI_API_KEY:-}" && ! -f "$GEMINI_SETTINGS_FILE" && ! -f "$GEMINI_OAUTH_FILE" ]]; then
  echo "Missing Gemini credentials. Set GEMINI_API_KEY/GOOGLE_API_KEY or mount ~/.gemini auth state." >&2
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
  --approval-mode yolo
  --output-format text
)

if [[ -n "${RUFFLE_GEMINI_MODEL:-}" ]]; then
  ARGS+=(-m "$RUFFLE_GEMINI_MODEL")
fi

gemini "${ARGS[@]}"
