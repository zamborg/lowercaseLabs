#!/bin/bash
# Builds a distributable Parlor: a Release Parlor.app with the image worker inside, parlor-cli, and a .dmg.
#
#   scripts/package.sh                          ad-hoc signed (runs on this Mac; others right-click → Open the first time)
#   SIGN_ID="Developer ID Application: Name (TEAMID)" scripts/package.sh
#                                               signed with hardened runtime, ready to notarize
#   SIGN_ID=... NOTARY_PROFILE=parlor scripts/package.sh
#                                               also notarizes and staples (profile from `xcrun notarytool store-credentials`)
#
# Output: dist/Parlor-<version>.dmg, dist/Parlor.app, dist/parlor-cli
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SIGN_ID="${SIGN_ID:--}"
OUT="$ROOT/dist"
DD="$ROOT/build/dist-derived"

# 1. The image worker lives in its own package with its own MLX pin.
if [ ! -x build-worker/Build/Products/Release/parlor-imagegen ]; then scripts/build-worker.sh; fi
test -x build-worker/Build/Products/Release/parlor-imagegen

# 2. The app (the build phase embeds the worker) and the CLI.
xcodegen generate >/dev/null
xcodebuild -scheme Parlor -configuration Release -destination 'platform=macOS' -derivedDataPath "$DD" \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_ALLOWED=YES build 2>&1 | grep -E "error:|BUILD" || true
xcodebuild -scheme parlor-cli -configuration Release -destination 'platform=macOS' -derivedDataPath "$DD" build 2>&1 | grep -E "error:|BUILD" || true
APP="$DD/Build/Products/Release/Parlor.app"
test -d "$APP/Contents/Resources/ImageWorker" || { echo "image worker missing from the app"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
cp -R "$APP" "$OUT/Parlor.app"
cp "$DD/Build/Products/Release/parlor-cli" "$OUT/parlor-cli"
APP="$OUT/Parlor.app"

# 3. Sign inside out: worker and its bundles, then the app, then the CLI.
FLAGS=(--force --timestamp=none)
if [ "$SIGN_ID" != "-" ]; then FLAGS=(--force --timestamp --options runtime); fi
W="$APP/Contents/Resources/ImageWorker"
for b in "$W"/*.bundle; do codesign "${FLAGS[@]}" -s "$SIGN_ID" "$b"; done
codesign "${FLAGS[@]}" -s "$SIGN_ID" --entitlements App/Parlor.entitlements "$W/parlor-imagegen"
codesign "${FLAGS[@]}" -s "$SIGN_ID" --entitlements App/Parlor.entitlements "$APP"
codesign "${FLAGS[@]}" -s "$SIGN_ID" --entitlements App/Parlor.entitlements "$OUT/parlor-cli"
codesign --verify --deep --strict "$APP"

VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP/Contents/Info.plist")
DMG="$OUT/Parlor-$VERSION.dmg"

# 4. The disk image: the app and an Applications shortcut.
STAGE="$ROOT/build/dmg-stage"; rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/Parlor.app"; ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Parlor" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
if [ "$SIGN_ID" != "-" ]; then codesign --force --timestamp -s "$SIGN_ID" "$DMG"; fi

# 5. Notarize (needs a Developer ID and a stored notarytool profile).
if [ -n "${NOTARY_PROFILE:-}" ] && [ "$SIGN_ID" != "-" ]; then
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
fi
echo "built $DMG ($(du -h "$DMG" | cut -f1))"
if [ "$SIGN_ID" = "-" ]; then echo "ad-hoc signed: on another Mac, right-click → Open the first time (System Settings → Privacy & Security → Open Anyway)."; fi
