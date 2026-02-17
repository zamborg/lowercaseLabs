#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -f "$ROOT_DIR/otp/data/graph.obj" ]]; then
  echo "Missing otp/data/graph.obj. Run: npm run otp:build" >&2
  exit 1
fi

docker compose -f "$ROOT_DIR/docker-compose.otp.yml" up -d otp
docker compose -f "$ROOT_DIR/docker-compose.otp.yml" logs -f --tail=50 otp

