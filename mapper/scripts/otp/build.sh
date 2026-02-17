#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -f "$ROOT_DIR/otp/data/osm.pbf" ]]; then
  echo "Missing otp/data/osm.pbf. Run: npm run otp:download" >&2
  exit 1
fi

if ! ls "$ROOT_DIR/otp/data/"gtfs-*.zip >/dev/null 2>&1; then
  echo "Missing otp/data/gtfs-*.zip. Run: npm run otp:download" >&2
  exit 1
fi

docker compose -f "$ROOT_DIR/docker-compose.otp.yml" run --rm otp-build

