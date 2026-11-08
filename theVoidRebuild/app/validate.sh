#!/bin/bash
# Local validation: build the Mac harness and run the shipped Transcriber + Tagger against the bundled weights.
set -euo pipefail
cd "$(dirname "$0")"
[ -d Model/LFM2-1.2B-4bit ] || ./fetch-model.sh
xcodegen generate >/dev/null
xcodebuild -project TheVoid.xcodeproj -scheme MacHarness -configuration Release \
  -destination 'platform=macOS' -derivedDataPath build/mac \
  -skipMacroValidation -skipPackagePluginValidation build 2>&1 | grep -E "error:|BUILD" || true
[ "${1:-}" = "--eval" ] && export VOID_EVAL=1
VOID_MODEL_DIR="$PWD/Model/LFM2-1.2B-4bit" build/mac/Build/Products/Release/MacHarness "$@"
