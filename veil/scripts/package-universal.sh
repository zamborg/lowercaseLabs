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
ditto -c -k --sequesterRsrc --keepParent "$app" releases/Veil-0.1.0-macos-universal.zip
cd releases
shasum -a 256 Veil-0.1.0-macos-universal.zip > SHA256SUMS
printf 'Created %s\n' "$PWD/Veil-0.1.0-macos-universal.zip"
