#!/usr/bin/env python3
"""One-time, idempotent data migration: public.memories -> mod_memory.memories.

Copies id, key, content, tags, created_at, updated_at, created_by_client_id, and
created_by_client_name (resolved via LEFT JOIN public.clients) into the `memory`
module schema, then verifies row counts and a SHA-256 checksum over
`(id, key, content, tags, updated_at)`.

Leaves `public.memories` untouched and prints ONLY counts and checksums (never content).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row


def _canonical_row(row: dict[str, Any]) -> str:
    updated: datetime = row["updated_at"]
    updated_iso = updated.astimezone(UTC).isoformat()
    tags = list(row["tags"] or [])
    return json.dumps(
        [str(row["id"]), str(row["key"]), str(row["content"]), tags, updated_iso],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _compute_checksum(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(_canonical_row(r).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def migrate_memories(database_url: str, *, allow_superset: bool = False) -> dict[str, Any]:
    """Copy `public.memories` -> `mod_memory.memories` idempotently and verify checksum."""
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        # Ensure target table exists (module should already be installed and migrated)
        exists_cur = conn.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'mod_memory' AND table_name = 'memories'
            """
        )
        if exists_cur.fetchone() is None:
            raise RuntimeError(
                "Target table mod_memory.memories does not exist. "
                "Install and start the 'memory' module first."
            )

        src_rows = conn.execute(
            """
            SELECT m.id, m.key, m.content, m.tags,
                   m.created_by_client_id, c.name AS created_by_client_name,
                   m.created_at, m.updated_at
            FROM public.memories m
            LEFT JOIN public.clients c ON c.id = m.created_by_client_id
            ORDER BY m.id ASC
            """
        ).fetchall()

        inserted = 0
        skipped = 0
        with conn.transaction():
            for row in src_rows:
                cur = conn.execute(
                    """
                    INSERT INTO mod_memory.memories (
                        id, key, content, tags,
                        created_by_client_id, created_by_client_name,
                        created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        row["id"],
                        row["key"],
                        row["content"],
                        list(row["tags"] or []),
                        str(row["created_by_client_id"]) if row["created_by_client_id"] else None,
                        row["created_by_client_name"],
                        row["created_at"],
                        row["updated_at"],
                    ),
                )
                if (cur.rowcount or 0) > 0:
                    inserted += 1
                else:
                    skipped += 1

        if allow_superset:
            src_ids = [r["id"] for r in src_rows]
            dst_rows = (
                conn.execute(
                    """
                    SELECT id, key, content, tags, updated_at
                    FROM mod_memory.memories
                    WHERE id = ANY(%s)
                    ORDER BY id ASC
                    """,
                    (src_ids,),
                ).fetchall()
                if src_ids
                else []
            )
            dst_total = int(
                (conn.execute("SELECT COUNT(*) AS cnt FROM mod_memory.memories").fetchone() or {"cnt": 0})[
                    "cnt"
                ]
            )
        else:
            dst_rows = conn.execute(
                """
                SELECT id, key, content, tags, updated_at
                FROM mod_memory.memories
                ORDER BY id ASC
                """
            ).fetchall()
            dst_total = len(dst_rows)

    src_count = len(src_rows)
    dst_matched_count = len(dst_rows)
    src_sha256 = _compute_checksum(src_rows)
    dst_sha256 = _compute_checksum(dst_rows)

    if src_count != dst_matched_count or src_sha256 != dst_sha256:
        raise RuntimeError(
            f"Verification failed: source_count={src_count} target_count={dst_matched_count} "
            f"source_sha256={src_sha256} target_sha256={dst_sha256}"
        )

    return {
        "ok": True,
        "source_count": src_count,
        "target_count": dst_total,
        "inserted": inserted,
        "skipped": skipped,
        "sha256": src_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Idempotently copy public.memories into mod_memory.memories and verify checksum."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", ""),
        help="Postgres DSN (defaults to DATABASE_URL env var).",
    )
    parser.add_argument(
        "--allow-superset",
        action="store_true",
        help="Allow mod_memory.memories to contain additional rows beyond public.memories.",
    )
    args = parser.parse_args()
    if not args.database_url.strip():
        print("Error: DATABASE_URL is required (env or --database-url).", file=sys.stderr)
        sys.exit(2)

    result = migrate_memories(args.database_url.strip(), allow_superset=args.allow_superset)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
