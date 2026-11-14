#!/bin/bash
# Builds the image worker (its own package, its own MLX version) into build-worker/.
# Swift is built -Onone on purpose: z-image-swift crashes in a specialized TaskLocal.withValue under -O (Swift 6.3).
# GPU work is unaffected; only host-side Swift is unoptimized.
set -euo pipefail
cd "$(dirname "$0")/../ImageWorker"
xcodebuild -scheme parlor-imagegen -destination 'platform=macOS' -configuration Release \
  -derivedDataPath ../build-worker SWIFT_OPTIMIZATION_LEVEL=-Onone build 2>&1 | grep -E "error:|BUILD" || true
test -x ../build-worker/Build/Products/Release/parlor-imagegen && echo "worker ok"
