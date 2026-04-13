#!/bin/bash

set -euo pipefail

export PYTHONPATH=/app/src

if python -m pytest /tests/test_outputs.py -q; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
