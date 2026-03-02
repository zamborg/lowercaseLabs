#!/usr/bin/env bash
set -euo pipefail

if poetry --version >/dev/null 2>&1; then
  exec poetry "$@"
fi

if command -v pyenv >/dev/null 2>&1; then
  while IFS= read -r candidate; do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if PYENV_VERSION="$candidate" poetry --version >/dev/null 2>&1; then
      exec env PYENV_VERSION="$candidate" poetry "$@"
    fi
  done < <(pyenv versions --bare 2>/dev/null)

  if pyenv exec poetry --version >/dev/null 2>&1; then
    exec pyenv exec poetry "$@"
  fi
fi

echo "Poetry is required but was not found in PATH or pyenv environments." >&2
exit 1
