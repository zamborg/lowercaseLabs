#!/usr/bin/env bash
# Copies the exact LFM2-1.2B-4bit snapshot the Mac eval picked into the app bundle folder.
# (Weights are 633 MB and gitignored. Run once after cloning, then `xcodegen generate`.)
set -euo pipefail
cd "$(dirname "$0")"
SNAP=$(ls -d ../eval/.cache/huggingface/hub/models--mlx-community--LFM2-1.2B-4bit/snapshots/3843e4ad0fcb8b7ed8a050908ac8f0bb5320d1bf)
DEST=Model/LFM2-1.2B-4bit
rm -rf "$DEST"; mkdir -p "$DEST"
cp -L "$SNAP"/{config.json,generation_config.json,model.safetensors,model.safetensors.index.json,special_tokens_map.json,tokenizer.json,tokenizer_config.json,chat_template.jinja} "$DEST"/
du -sh "$DEST"
