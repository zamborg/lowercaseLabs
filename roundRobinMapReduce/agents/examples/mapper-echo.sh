#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${1:?task.md path required}"
{
  echo "# Mapper Result"
  echo
  echo "Workspace: $(pwd)"
  echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  echo "## Task Excerpt"
  sed -n '1,24p' "$TASK_FILE"
  echo
  echo "## Files"
  find . -maxdepth 2 -type f | sort
} > RESULT.md

printf 'mapper completed in %s\n' "$(pwd)"
