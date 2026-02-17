#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$ROOT_DIR/otp/data"
mkdir -p "$DATA_DIR"

OSM_URL="https://download.bbbike.org/osm/bbbike/SanFrancisco/SanFrancisco.osm.pbf"
BART_GTFS_URL="https://www.bart.gov/dev/schedules/google_transit.zip"

echo "Downloading OSM (SF extract)…"
curl -L --fail --retry 3 --retry-delay 1 -o "$DATA_DIR/osm.pbf" "$OSM_URL"

echo "Downloading GTFS (BART)…"
curl -L --fail --retry 3 --retry-delay 1 -o "$DATA_DIR/gtfs-bart.zip" "$BART_GTFS_URL"

echo "Done."
echo "Files:"
ls -lah "$DATA_DIR"

