#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
scratch="${VEIL_BUILD_CACHE:-$PWD/.build/universal}"
swift build --scratch-path "$scratch" --configuration release --arch arm64 --arch x86_64
bin_dir="$(swift build --scratch-path "$scratch" --configuration release --arch arm64 --arch x86_64 --show-bin-path)"
app="$PWD/build/universal/Veil.app"
mkdir -p "$app/Contents/MacOS" releases
cp "$bin_dir/Veil" "$app/Contents/MacOS/Veil"
cp Resources/Info.plist "$app/Contents/Info.plist"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"
lipo "$app/Contents/MacOS/Veil" -verify_arch arm64 x86_64
version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' Resources/Info.plist)"
archive="Veil-$version-macos-universal.zip"
ditto -c -k --sequesterRsrc --keepParent "$app" "releases/$archive"
cd releases
shasum -a 256 Veil-*-macos-universal.zip > SHA256SUMS
printf 'Created %s\n' "$PWD/$archive"
