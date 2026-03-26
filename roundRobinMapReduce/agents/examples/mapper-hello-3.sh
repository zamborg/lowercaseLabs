#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${1:?task.md path required}"

cat > hello.py <<'PY'
import sys

sys.stdout.write("hello world\n")
PY

{
  echo "# Mapper Result"
  echo
  echo "Candidate: mapper-hello-3"
  echo "Task: $TASK_FILE"
  echo "File: hello.py"
  echo "LineCount: $(wc -l < hello.py | tr -d ' ')"
  echo
  echo '```python'
  cat hello.py
  echo '```'
} > RESULT.md

printf 'mapper-hello-3 wrote hello.py with %s lines\n' "$(wc -l < hello.py | tr -d ' ')"
