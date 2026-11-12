#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release
app="$PWD/build/Veil.app"
mkdir -p "$app/Contents/MacOS"
cp .build/release/Veil "$app/Contents/MacOS/Veil"
cp Resources/Info.plist "$app/Contents/Info.plist"
codesign --force --sign - "$app"
printf 'Built %s\n' "$app"
