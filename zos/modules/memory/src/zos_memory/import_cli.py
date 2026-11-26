"""Batch JSONL importer for the ZOS Memory module (reads stdin or file, prints counts only)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TextIO

from zos_memory.store import MemoryStore, create_memory_pool


async def import_jsonl_stream(
    store: MemoryStore,
    stream: TextIO,
    *,
    source_name: str = "import",
) -> dict[str, int]:
    discovered = 0
    created = 0
    updated = 0
    unchanged = 0
    for line in stream:
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        key = str(obj.get("key", "")).strip()
        content = obj.get("content")
        tags = obj.get("tags") or []
        if not key or not isinstance(content, str):
            continue
        discovered += 1
        _row, status = await store.put_memory(
            key=key,
            content=content,
            tags=tags,
            client_id=f"import:{source_name}",
            client_name=source_name,
        )
        if status == "created":
            created += 1
        elif status == "updated":
            updated += 1
        else:
            unchanged += 1
    return {
        "discovered": discovered,
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
    }


async def _run(file_path: str | None, source_name: str) -> None:
    database_url = os.environ.get("ZOS_MODULE_DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("ZOS_MODULE_DATABASE_URL is required")
    pool = create_memory_pool(database_url)
    await pool.open()
    try:
        store = MemoryStore(pool)
        await store.init_schema()
        if file_path and file_path != "-":
            with Path(file_path).open("r", encoding="utf-8") as fh:
                summary = await import_jsonl_stream(store, fh, source_name=source_name)
        else:
            summary = await import_jsonl_stream(store, sys.stdin, source_name=source_name)
    finally:
        await pool.close()
    print(json.dumps(summary))


def main() -> None:
    args = sys.argv[1:]
    file_path: str | None = None
    source_name = "import"
    i = 0
    while i < len(args):
        if args[i] == "--source" and i + 1 < len(args):
            source_name = args[i + 1]
            i += 2
        else:
            file_path = args[i]
            i += 1
    asyncio.run(_run(file_path, source_name))


if __name__ == "__main__":
    main()
