#!/usr/bin/env bash
set -euo pipefail

ports=(5173 8080 8081)
for p in "${ports[@]}"; do
  if lsof -ti :"$p" >/dev/null 2>&1; then
    echo "Killing processes on :$p"
    lsof -ti :"$p" | xargs kill -9
  fi
done

echo "Done."

