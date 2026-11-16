#!/bin/bash
# Xcode build phase: copy the image worker and its resource bundles into the app's Resources/ImageWorker folder.
SRC="${SRCROOT}/build-worker/Build/Products/Release"
DEST="${TARGET_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/Resources/ImageWorker"
if [ ! -x "$SRC/parlor-imagegen" ]; then
  echo "note: image worker not built (run scripts/build-worker.sh); Z-Image will be unavailable"
  SKIP_WORKER=1
fi
if [ -z "${SKIP_WORKER:-}" ]; then
  mkdir -p "$DEST"
  rsync -a "$SRC/parlor-imagegen" "$DEST/"
  for b in "$SRC"/*.bundle; do rsync -a "$b" "$DEST/"; done
fi

# Stable Diffusion backend (sd-cli), if fetched by scripts/fetch-sd.sh.
SD="${SRCROOT}/build-sd"
if [ -x "$SD/sd-cli" ]; then
  mkdir -p "${TARGET_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/Resources/SD"
  rsync -a "$SD/sd-cli" "$SD/libstable-diffusion.dylib" "$SD"/*.txt "${TARGET_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/Resources/SD/" 2>/dev/null || true
fi
