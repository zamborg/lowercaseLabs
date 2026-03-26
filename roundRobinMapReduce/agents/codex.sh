#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${1:?task.md path required}"
CODEX_AUTH_FILE="${HOME:-/root}/.codex/auth.json"

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" && ! -f "$CODEX_AUTH_FILE" ]]; then
  echo "Missing Codex credentials. Set OPENAI_API_KEY or mount ~/.codex/auth.json." >&2
  exit 1
fi

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  export HOME="$(pwd)/.ruffle-codex-home"
  mkdir -p "$HOME/.codex"
  python3 - <<'PY'
import json, os
home = os.environ["HOME"]
key = os.environ["OPENAI_API_KEY"]
with open(os.path.join(home, ".codex", "auth.json"), "w") as f:
    json.dump({"OPENAI_API_KEY": key}, f)
PY
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
  exec
  -
  --skip-git-repo-check
  --dangerously-bypass-approvals-and-sandbox
)

if [[ -n "${RUFFLE_CODEX_MODEL:-}" ]]; then
  ARGS+=(--model "$RUFFLE_CODEX_MODEL")
fi

printf '%s\n' "$PROMPT" | codex "${ARGS[@]}"
