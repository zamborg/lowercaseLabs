#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${1:?task.md path required}"
{
  echo "# Reduced Result"
  echo
  echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  echo "## Task Excerpt"
  sed -n '1,24p' "$TASK_FILE"
  echo
  echo "## Mapper Summaries"
  for result in workspaces/mapper-*/RESULT.md; do
    echo
    echo "### $result"
    sed -n '1,40p' "$result"
  done
} > RESULT.md

printf 'reducer completed in %s\n' "$(pwd)"
