#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${1:?task.md path required}"
best_file=""
best_lines=""

for candidate in workspaces/mapper-*/hello.py; do
  lines="$(wc -l < "$candidate" | tr -d ' ')"
  if [[ -z "$best_lines" || "$lines" -lt "$best_lines" ]]; then
    best_lines="$lines"
    best_file="$candidate"
  fi
done

cp "$best_file" hello.py

{
  echo "# Reduced Result"
  echo
  echo "Task: $TASK_FILE"
  echo "Chosen: $best_file"
  echo "LineCount: $best_lines"
  echo
  echo '```python'
  cat hello.py
  echo '```'
} > RESULT.md

printf 'reducer chose %s with %s lines\n' "$best_file" "$best_lines"
