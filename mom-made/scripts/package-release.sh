#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

identity=$(security find-identity -v -p codesigning | awk '/"Developer ID Application:/ && !found { print $2; found = 1 }')
if [[ -z "$identity" ]]; then
  echo 'Install a valid Developer ID Application certificate and its private key in Keychain before building a release.' >&2
  exit 1
fi
: "${APPLE_KEYCHAIN_PROFILE:?Set APPLE_KEYCHAIN_PROFILE to a profile saved by xcrun notarytool store-credentials.}"

notary_args=(--keychain-profile "$APPLE_KEYCHAIN_PROFILE")
if [[ -n "${APPLE_KEYCHAIN:-}" ]]; then
  notary_args+=(--keychain "$APPLE_KEYCHAIN")
fi
xcrun notarytool history "${notary_args[@]}" >/dev/null

mkdir -p release
staging=$(mktemp -d "$PWD/release/notarizing.XXXXXX")
version=$(node -p 'require("./package.json").version')
app="$staging/mac-arm64/mom-made.app"
dmg="$staging/mom-made-$version-arm64.dmg"

npm run dist:dmg -- \
  --config.directories.output="$staging" \
  --config.forceCodeSigning=true \
  --config.mac.identity="$identity" \
  --config.mac.hardenedRuntime=true \
  --config.mac.notarize=true \
  --config.dmg.sign=true

codesign --verify --deep --strict "$app"
xcrun stapler validate "$app"
syspolicy_check distribution "$app"
codesign --verify "$dmg"
xcrun notarytool submit "$dmg" "${notary_args[@]}" --wait
xcrun stapler staple "$dmg"
xcrun stapler validate "$dmg"
hdiutil verify "$dmg"

mkdir -p release/distribution
mv "$dmg" release/distribution/
printf 'Verified release DMG: %s/release/distribution/%s\n' "$PWD" "$(basename "$dmg")"
