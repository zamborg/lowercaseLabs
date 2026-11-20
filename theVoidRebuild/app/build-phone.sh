#!/bin/bash
# Build Release and install on the connected iPhone. Zubin's phone is for deliberate checks, not the default loop.
#   ./build-phone.sh            iCloud off (works with a plain development profile)
#   ./build-phone.sh --icloud   iCloud + dots on (needs Xcode signed into the Apple ID: SETUP.md step 1)
set -euo pipefail
cd "$(dirname "$0")"
[ -d Model/LFM2-1.2B-4bit ] || ./fetch-model.sh
if [ "${1:-}" = "--icloud" ]; then export VOID_ICLOUD=true VOID_ENTITLEMENTS=Sources/TheVoid-iCloud.entitlements
else export VOID_ICLOUD=false VOID_ENTITLEMENTS=; fi
xcodegen generate >/dev/null
mkdir -p build
xcodebuild -project TheVoid.xcodeproj -scheme TheVoid -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath build/dev -allowProvisioningUpdates -skipMacroValidation -skipPackagePluginValidation build > build/dev-build.log 2>&1 \
  || { grep -E "error:|BUILD" build/dev-build.log | sort -u; echo "BUILD FAILED"; exit 1; }
DEVICE=$(xcrun devicectl list devices 2>/dev/null | awk '/connected/ {print $3; exit}')
[ -n "$DEVICE" ] || { echo "No connected iPhone (unlock it and plug it in)"; exit 1; }
xcrun devicectl device install app --device "$DEVICE" build/dev/Build/Products/Release-iphoneos/TheVoid.app
echo "Installed on $DEVICE (iCloud: $VOID_ICLOUD)"
