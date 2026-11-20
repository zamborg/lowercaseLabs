"""Shared schema, paths and durable JSON helpers. No model imports here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAXONOMY_PATH = ROOT.parent / "taxonomy.json"
TAXONOMY = json.loads(TAXONOMY_PATH.read_text())
TAGS = [tag["id"] for tag in TAXONOMY["tags"]]
MODEL_CACHE = ROOT / ".cache" / "huggingface" / "hub"
MODEL_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"]
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1, "maxLength": 60},
        "tags": {
            "type": "array", "items": {"type": "string", "enum": TAGS},
            "minItems": 1, "maxItems": 2,
        },
    },
    "required": ["title", "tags"],
    "additionalProperties": False,
}


def configure_caches():
    # Defaults deliberately stay inside the authorized workspace.
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Outputs must stay under {ROOT}: {path}")
    return path


def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
