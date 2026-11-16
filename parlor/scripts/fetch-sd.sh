#!/bin/bash
# Fetches stable-diffusion.cpp's sd-cli (the Stable Diffusion image backend) into build-sd/, arm64 only.
# Pinned to one release and checked against its SHA-256. Run by scripts/package.sh; the Xcode build phase embeds the result.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="master-1104-5dcfa79"
ZIP="sd-master-5dcfa79-bin-Darwin-macOS-26.7.2-arm64.zip"
SHA="c1d93bfc0bdcddbf3e62d41dd63597cae69ec712804993bd4ad11599bbdcdd4b"
if [ -x build-sd/sd-cli ] && [ -f build-sd/libstable-diffusion.dylib ]; then exit 0; fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
curl -fsSL -o "$TMP/sd.zip" "https://github.com/leejet/stable-diffusion.cpp/releases/download/$TAG/$ZIP"
echo "$SHA  $TMP/sd.zip" | shasum -a 256 -c - >/dev/null || { echo "sd.cpp checksum mismatch"; exit 1; }
unzip -q "$TMP/sd.zip" -d "$TMP/x"
mkdir -p build-sd
lipo -thin arm64 "$TMP/x/sd-cli" -output build-sd/sd-cli
lipo -thin arm64 "$TMP/x/libstable-diffusion.dylib" -output build-sd/libstable-diffusion.dylib
chmod +x build-sd/sd-cli
# lipo drops the signature, and arm64 refuses unsigned code; package.sh signs again for release.
codesign --force -s - build-sd/libstable-diffusion.dylib build-sd/sd-cli
cp "$TMP/x/stable-diffusion.cpp.txt" "$TMP/x/ggml.txt" build-sd/ 2>/dev/null || true
echo "fetched sd-cli $TAG"
