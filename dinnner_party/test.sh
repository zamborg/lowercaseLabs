#!/bin/zsh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
API_BASE="http://127.0.0.1:8000"
LOG_DIR="$ROOT/logs"
PROMPT_TEMPLATE="$ROOT/task.md"
DRY_RUN="${DRY_RUN:-0}"
AUTO_OPEN="${AUTO_OPEN:-1}"
STARTED_SERVER=0
AGENT_PIDS=()
MODEL="${MODEL:-gpt-5.1}"
AGENT_LINES="$(find "$ROOT/people" -mindepth 2 -maxdepth 2 -name person.json -print | sed 's#/person.json$##' | xargs -n1 basename 2>/dev/null | sort)"
AGENTS=()
if [[ -n "$AGENT_LINES" ]]; then
  AGENTS=("${(@f)AGENT_LINES}")
fi

mkdir -p "$LOG_DIR"

cleanup() {
  for pid in "${AGENT_PIDS[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  if [[ "$STARTED_SERVER" == "1" && -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TMP_DIR:-}" && -d "${TMP_DIR:-}" ]]; then
    rm -rf "$TMP_DIR"
  fi
}

trap cleanup EXIT INT TERM

if [[ "${#AGENTS[@]}" -eq 0 ]]; then
  echo "No registered agents found. Create them from the web UI first."
  exit 1
fi

wait_for_server() {
  for _ in {1..50}; do
    if curl -sf "$API_BASE/api/spec" >/dev/null; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

if ! curl -sf "$API_BASE/api/spec" >/dev/null; then
  echo "Starting local server..."
  python3 "$ROOT/server.py" >"$LOG_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  if ! wait_for_server; then
    echo "Server did not become ready. See $LOG_DIR/server.log"
    exit 1
  fi
else
  echo "Using existing server at $API_BASE"
fi

echo "Resetting room..."
curl -sf -X POST "$API_BASE/api/reset" >/dev/null

TMP_DIR="$(mktemp -d)"

for agent in "${AGENTS[@]}"; do
  prompt_file="$TMP_DIR/$agent.md"
  soul_path="$ROOT/people/$agent/soul.md"
  person_path="$ROOT/people/$agent/person.json"
  participants="$(printf '%s\n' "${AGENTS[@]}" | grep -vx "$agent" | paste -sd ',' - | sed 's/,/, /g')"
  if [[ -z "$participants" ]]; then
    participants="no one else yet"
  fi
  sed \
    -e "s/__AGENT__/$agent/g" \
    -e "s/__MODEL__/$MODEL/g" \
    -e "s#__SOUL_PATH__#$soul_path#g" \
    -e "s/__PARTICIPANTS__/$participants/g" \
    "$PROMPT_TEMPLATE" >"$prompt_file"
  {
    echo
    echo "# Person Record"
    echo
    echo '```json'
    cat "$person_path"
    echo
    echo '```'
    echo
    echo "# Soul"
    echo
    cat "$soul_path"
    echo
  } >>"$prompt_file"
done

if [[ "$AUTO_OPEN" == "1" ]] && command -v open >/dev/null 2>&1; then
  open "$API_BASE" >/dev/null 2>&1 || true
fi

echo "Launching codex exec participants..."

for agent in "${AGENTS[@]}"; do
  log_file="$LOG_DIR/$agent.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] $agent -> $log_file"
    continue
  fi

  codex exec \
    --skip-git-repo-check \
    --dangerously-bypass-approvals-and-sandbox \
    -m "$MODEL" \
    -C "$ROOT" \
    <"$TMP_DIR/$agent.md" \
    >"$log_file" 2>&1 &
  AGENT_PIDS+=("$!")
  echo "  $agent -> $log_file"
done

echo
echo "Chat UI: $API_BASE"
echo "Logs: $LOG_DIR"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only. No codex exec processes were started."
else
  echo "Press Ctrl+C to stop the run."
  wait "${AGENT_PIDS[@]}"
fi
