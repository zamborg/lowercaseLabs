"""Pluggable memory importers (`claude-code`, `codex`) and JSONL wire format."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from zos.memory_service import MemoryImportItem

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_MULTI_DASH_RE = re.compile(r"-{2,}")


def _slugify(raw: str) -> str:
    cleaned = raw.strip().lower().replace(" ", "-").replace("_", "-")
    cleaned = _SLUG_RE.sub("-", cleaned)
    cleaned = _MULTI_DASH_RE.sub("-", cleaned).strip("-")
    return cleaned or "unnamed"


def _strip_quotes(val: str) -> str:
    s = val.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse minimal YAML frontmatter (top-level scalars and 1-level nested dicts) + body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()

    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text.strip()

    meta: dict[str, Any] = {}
    current_parent: str | None = None
    for raw_line in lines[1:end_idx]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in raw_line:
            continue
        key, raw_val = raw_line.split(":", 1)
        key = key.strip()
        val = _strip_quotes(raw_val)
        if indent == 0:
            if not val:
                meta[key] = {}
                current_parent = key
            else:
                meta[key] = val
                current_parent = None
        elif current_parent is not None and isinstance(meta.get(current_parent), dict):
            meta[current_parent][key] = val

    body = "\n".join(lines[end_idx + 1 :]).strip()
    return meta, body


def _extract_type(fm: dict[str, Any]) -> str:
    nested = fm.get("metadata")
    if isinstance(nested, dict) and nested.get("type"):
        return str(nested["type"]).strip()
    if fm.get("type"):
        return str(fm["type"]).strip()
    return ""


def _combine_content(description: str, body: str) -> str:
    desc = description.strip()
    b = body.strip()
    if desc and b:
        return f"{desc}\n\n{b}"
    return desc or b


def _project_slug_from_dir(dir_name: str) -> str:
    """Turn a Claude Code project dir name (e.g. '-Users-foo-bar-myproj') into a clean slug."""
    stripped = dir_name.strip().lstrip("-")
    return _slugify(stripped) if stripped else "default"


@runtime_checkable
class MemoryImporter(Protocol):
    source: str

    def discover(self, root: Path) -> list[MemoryImportItem]:
        """Discover normalized memory items under `root`."""
        ...


class ClaudeCodeMemoryImporter:
    """Imports `~/.claude/projects/*/memory/*.md` (skipping `MEMORY.md` index files)."""

    source: str = "claude-code"

    def default_root(self) -> Path:
        return Path("~/.claude/projects").expanduser()

    def discover(self, root: Path | None = None) -> list[MemoryImportItem]:
        base = (root or self.default_root()).expanduser()
        if not base.is_dir():
            return []

        raw_records: list[tuple[str, str, str, str, tuple[str, ...]]] = []
        for md_path in sorted(base.glob("*/memory/*.md")):
            if not md_path.is_file() or md_path.name.upper() == "MEMORY.MD":
                continue
            project_slug = _project_slug_from_dir(md_path.parents[1].name)
            text = md_path.read_text(encoding="utf-8", errors="replace")
            fm, body = parse_frontmatter(text)
            raw_name = str(fm.get("name") or md_path.stem).strip()
            base_key = _slugify(raw_name) if raw_name else _slugify(md_path.stem)
            stem_key = _slugify(md_path.stem)
            description = str(fm.get("description") or "").strip()
            content = _combine_content(description, body)
            if not content:
                continue

            mem_type = _extract_type(fm)
            tags = [f"source:{self.source}", f"project:{project_slug}"]
            if mem_type:
                tags.append(f"type:{_slugify(mem_type)}")
            raw_records.append((base_key, project_slug, stem_key, content, tuple(sorted(set(tags)))))

        key_counts = Counter(rec[0] for rec in raw_records)
        scoped_counts = Counter(f"{rec[1]}/{rec[0]}" for rec in raw_records)

        items: list[MemoryImportItem] = []
        for base_key, project_slug, stem_key, content, tags in raw_records:
            if key_counts[base_key] > 1:
                scoped = f"{project_slug}/{base_key}"
                final_key = f"{project_slug}/{stem_key}" if scoped_counts[scoped] > 1 else scoped
            else:
                final_key = base_key
            items.append(MemoryImportItem(key=final_key, content=content, tags=tags))
        return items


class CodexMemoryImporter:
    """Imports `~/.codex/memories/*.md` (never reads Codex SQLite internals)."""

    source: str = "codex"

    def default_root(self) -> Path:
        return Path("~/.codex/memories").expanduser()

    def discover(self, root: Path | None = None) -> list[MemoryImportItem]:
        base = (root or self.default_root()).expanduser()
        if not base.is_dir():
            return []

        items: list[MemoryImportItem] = []
        seen: set[str] = set()
        for md_path in sorted(base.glob("*.md")):
            if not md_path.is_file() or md_path.name.upper() == "MEMORY.MD":
                continue
            text = md_path.read_text(encoding="utf-8", errors="replace")
            fm, body = parse_frontmatter(text)
            raw_name = str(fm.get("name") or md_path.stem).strip()
            key = _slugify(raw_name) if raw_name else _slugify(md_path.stem)
            if key in seen:
                key = _slugify(md_path.stem)
            seen.add(key)
            description = str(fm.get("description") or "").strip()
            content = _combine_content(description, body)
            if not content:
                continue

            mem_type = _extract_type(fm)
            tags = [f"source:{self.source}"]
            if mem_type:
                tags.append(f"type:{_slugify(mem_type)}")
            items.append(MemoryImportItem(key=key, content=content, tags=tuple(sorted(set(tags)))))
        return items


IMPORTERS: dict[str, ClaudeCodeMemoryImporter | CodexMemoryImporter] = {
    "claude-code": ClaudeCodeMemoryImporter(),
    "codex": CodexMemoryImporter(),
}


def dumps_jsonl(items: list[MemoryImportItem]) -> str:
    """Serialize normalized MemoryImportItem records as JSONL for container stdin transport."""
    lines = [
        json.dumps(
            {"key": item.key, "content": item.content, "tags": list(item.tags)},
            ensure_ascii=False,
        )
        for item in items
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def loads_jsonl(raw: str) -> list[MemoryImportItem]:
    """Parse JSONL stream into normalized MemoryImportItem records."""
    items: list[MemoryImportItem] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if not isinstance(obj, dict) or not obj.get("key") or not isinstance(obj.get("content"), str):
            raise ValueError(f"Invalid JSONL memory item on line {lineno}")
        raw_tags = obj.get("tags") or []
        tags = tuple(sorted({str(t).strip() for t in raw_tags if str(t).strip()}))
        items.append(MemoryImportItem(key=str(obj["key"]).strip(), content=obj["content"], tags=tags))
    return items
