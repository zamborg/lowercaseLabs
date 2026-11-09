#!/bin/bash
# Archive Release and upload to App Store Connect, which makes it a TestFlight build.
#   ./ship.sh --archive-only   archive with distribution signing, stop before uploading (safe dry run)
#   ./ship.sh --yes            archive + upload. Refuses without --yes: shipping is Zubin's call (AGENTS.md, D10).
# Needs SETUP.md step 3 (API key in ~/.appstoreconnect/) and an App Store Connect app record for this bundle id.
# Secrets are read from the files below and never printed.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-}"
[ "$MODE" = "--yes" ] || [ "$MODE" = "--archive-only" ] || { echo "Refusing to upload without --yes (or use --archive-only)."; exit 2; }
CONFIG="$HOME/.appstoreconnect/config"
[ -f "$CONFIG" ] || { echo "Missing $CONFIG (SETUP.md step 3)"; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG"   # KEY_ID, ISSUER_ID
KEY="$HOME/.appstoreconnect/private_keys/AuthKey_${KEY_ID}.p8"
[ -f "$KEY" ] || { echo "Missing API key file for KEY_ID (SETUP.md step 3)"; exit 1; }
AUTH=(-allowProvisioningUpdates -authenticationKeyPath "$KEY" -authenticationKeyID "$KEY_ID" -authenticationKeyIssuerID "$ISSUER_ID")

BUILD="${BUILD_NUMBER:-$(date +%y%m%d%H%M)}"   # CFBundleVersion must increase on every upload
export VOID_ICLOUD=true VOID_ENTITLEMENTS=Sources/TheVoid-iCloud.entitlements
[ -d Model/LFM2-1.2B-4bit ] || ./fetch-model.sh
xcodegen generate >/dev/null
rm -rf build/ship; mkdir -p build/ship

cat > build/ship/ExportOptions.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>method</key><string>app-store-connect</string>
  <key>destination</key><string>upload</string>
  <key>teamID</key><string>9G57J2JMF7</string>
  <key>signingStyle</key><string>automatic</string>
  <key>uploadSymbols</key><true/>
</dict></plist>
PLIST

echo "Archiving build $BUILD..."
xcodebuild archive -project TheVoid.xcodeproj -scheme TheVoid -configuration Release \
  -destination 'generic/platform=iOS' -archivePath build/ship/TheVoid.xcarchive \
  CURRENT_PROJECT_VERSION="$BUILD" "${AUTH[@]}" -skipMacroValidation -skipPackagePluginValidation \
  > build/ship/archive.log 2>&1 || { grep -E "error:" build/ship/archive.log | sort -u | head; echo "ARCHIVE FAILED (build/ship/archive.log)"; exit 1; }

[ "$MODE" = "--archive-only" ] && { echo "Archive OK: build/ship/TheVoid.xcarchive (build $BUILD). Not uploaded."; exit 0; }
echo "Uploading..."
xcodebuild -exportArchive -archivePath build/ship/TheVoid.xcarchive -exportPath build/ship/export \
  -exportOptionsPlist build/ship/ExportOptions.plist "${AUTH[@]}" \
  > build/ship/export.log 2>&1 || { grep -E "error:|rror" build/ship/export.log | sort -u | head; echo "UPLOAD FAILED (build/ship/export.log)"; exit 1; }
echo "Uploaded build $BUILD. It appears in TestFlight after App Store Connect finishes processing (usually 5-20 min)."
