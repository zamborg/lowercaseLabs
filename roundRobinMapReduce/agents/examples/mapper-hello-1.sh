#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${1:?task.md path required}"

cat > hello.py <<'PY'
print("hello world")
PY

{
  echo "# Mapper Result"
  echo
  echo "Candidate: mapper-hello-1"
  echo "Task: $TASK_FILE"
  echo "File: hello.py"
  echo "LineCount: $(wc -l < hello.py | tr -d ' ')"
  echo
  echo '```python'
  cat hello.py
  echo '```'
} > RESULT.md

printf 'mapper-hello-1 wrote hello.py with %s line\n' "$(wc -l < hello.py | tr -d ' ')"
