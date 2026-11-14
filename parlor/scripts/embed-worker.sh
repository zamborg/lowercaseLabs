#!/bin/bash
# Xcode build phase: copy the image worker and its resource bundles into the app's Resources/ImageWorker folder.
SRC="${SRCROOT}/build-worker/Build/Products/Release"
DEST="${TARGET_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/Resources/ImageWorker"
if [ ! -x "$SRC/parlor-imagegen" ]; then
  echo "note: image worker not built (run scripts/build-worker.sh); images will be unavailable"
  exit 0
fi
mkdir -p "$DEST"
rsync -a "$SRC/parlor-imagegen" "$DEST/"
for b in "$SRC"/*.bundle; do rsync -a "$b" "$DEST/"; done
