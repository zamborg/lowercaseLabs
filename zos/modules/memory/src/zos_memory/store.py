"""Postgres FTS + pg_trgm storage layer for the ZOS Memory module (uses only its own schema)."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

MAX_SEARCH_LIMIT = 50
MAX_LIST_LIMIT = 100
PutStatus = Literal["created", "updated", "unchanged"]

MIGRATIONS: list[tuple[str, str]] = [
    (
        "0001_init",
        """
        CREATE TABLE IF NOT EXISTS memories (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            key TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL,
            tags TEXT[] NOT NULL DEFAULT '{}',
            fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', key || ' ' || content)) STORED,
            created_by_client_id TEXT NULL,
            created_by_client_name TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS memories_fts_gin_idx ON memories USING GIN (fts);
        CREATE INDEX IF NOT EXISTS memories_content_trgm_gin_idx ON memories USING GIN (content gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS memories_tags_gin_idx ON memories USING GIN (tags);
        CREATE INDEX IF NOT EXISTS memories_updated_at_idx ON memories (updated_at DESC, id DESC);
        """,
    ),
]


def normalize_tags(raw_tags: Iterable[str] | None) -> list[str]:
    if not raw_tags:
        return []
    return sorted({str(t).strip() for t in raw_tags if str(t).strip()})


def encode_list_cursor(updated_at: datetime, row_id: UUID) -> str:
    payload = json.dumps(
        {"u": updated_at.isoformat(), "i": str(row_id)},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_list_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        padded = cursor.strip() + "=" * ((-len(cursor.strip())) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        return datetime.fromisoformat(str(data["u"])), UUID(str(data["i"]))
    except Exception as exc:
        raise ValueError("Invalid memory_list cursor") from exc


def serialize_memory(row: dict[str, Any], *, score: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": str(row["key"]),
        "content": str(row["content"]),
        "tags": list(row["tags"] or []),
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        "author": row.get("created_by_client_name"),
    }
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


def serialize_memory_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": str(row["key"]),
        "tags": list(row["tags"] or []),
        "updated_at": row["updated_at"].isoformat(),
        "author": row.get("created_by_client_name"),
    }


def create_memory_pool(database_url: str) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"row_factory": dict_row},
    )


class MemoryStore:
    """Async Postgres store for key/content/tags memories inside `mod_memory`."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> None:
        self.pool = pool

    async def init_schema(self) -> list[str]:
        """Run idempotent, versioned schema migrations inside the module's own schema."""
        applied_now: list[str] = []
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.commit()

            cur = await conn.execute("SELECT version FROM memory_schema_migrations")
            already = {str(r["version"]) for r in await cur.fetchall()}

            for version, sql_body in MIGRATIONS:
                if version in already:
                    continue
                async with conn.transaction():
                    for stmt in sql_body.split(";"):
                        s = stmt.strip()
                        if s:
                            await conn.execute(s)
                    await conn.execute(
                        "INSERT INTO memory_schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
                        (version,),
                    )
                applied_now.append(version)
        return applied_now

    async def put_memory(
        self,
        *,
        key: str,
        content: str,
        tags: Iterable[str] | None = None,
        client_id: str | None = None,
        client_name: str | None = None,
    ) -> tuple[dict[str, Any], PutStatus]:
        """Idempotently create or update a memory entry by key."""
        clean_key = key.strip()
        if not clean_key or not isinstance(content, str):
            raise ValueError("memory_put requires non-empty string 'key' and 'content'.")
        norm_tags = normalize_tags(tags)

        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                SELECT id, key, content, tags, created_by_client_id, created_by_client_name,
                       created_at, updated_at
                FROM memories
                WHERE key = %s
                FOR UPDATE
                """,
                (clean_key,),
            )
            existing = await cur.fetchone()
            if existing is not None:
                existing_tags = normalize_tags(existing["tags"])
                if existing["content"] == content and existing_tags == norm_tags:
                    return serialize_memory(dict(existing)), "unchanged"

                cur = await conn.execute(
                    """
                    UPDATE memories
                    SET content = %s,
                        tags = %s,
                        updated_at = now()
                    WHERE id = %s
                    RETURNING id, key, content, tags, created_by_client_id, created_by_client_name,
                              created_at, updated_at
                    """,
                    (content, norm_tags, existing["id"]),
                )
                updated = await cur.fetchone()
                assert updated is not None
                return serialize_memory(dict(updated)), "updated"

            cur = await conn.execute(
                """
                INSERT INTO memories (key, content, tags, created_by_client_id, created_by_client_name)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, key, content, tags, created_by_client_id, created_by_client_name,
                          created_at, updated_at
                """,
                (clean_key, content, norm_tags, client_id, client_name),
            )
            inserted = await cur.fetchone()
            assert inserted is not None
            return serialize_memory(dict(inserted)), "created"

    async def get_memory(self, *, key: str) -> dict[str, Any] | None:
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("memory_get requires non-empty string 'key'.")
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, key, content, tags, created_by_client_id, created_by_client_name,
                       created_at, updated_at
                FROM memories
                WHERE key = %s
                """,
                (clean_key,),
            )
            row = await cur.fetchone()
        return serialize_memory(dict(row)) if row is not None else None

    async def get_memory_by_id(self, *, memory_id: str) -> dict[str, Any] | None:
        uid = UUID(memory_id.strip())
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, key, content, tags, created_by_client_id, created_by_client_name,
                       created_at, updated_at
                FROM memories
                WHERE id = %s
                """,
                (uid,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        out = serialize_memory(dict(row))
        out["id"] = str(row["id"])
        out["created_by_client_id"] = row["created_by_client_id"]
        return out

    async def search_memories(
        self,
        *,
        query: str = "",
        tags: Iterable[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        clean_query = (query or "").strip()
        clean_tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
        capped_limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))

        async with self.pool.connection() as conn:
            if not clean_query:
                cur = await conn.execute(
                    """
                    SELECT id, key, content, tags, created_by_client_id, created_by_client_name,
                           created_at, updated_at, 1.0::float AS score
                    FROM memories
                    WHERE (%s::text[] = '{}' OR tags @> %s::text[])
                    ORDER BY updated_at DESC, id DESC
                    LIMIT %s
                    """,
                    (clean_tags, clean_tags, capped_limit),
                )
                rows = await cur.fetchall()
                mode = "recent"
            else:
                cur = await conn.execute(
                    """
                    SELECT id, key, content, tags, created_by_client_id, created_by_client_name,
                           created_at, updated_at,
                           ts_rank(fts, websearch_to_tsquery('english', %s)) AS score
                    FROM memories
                    WHERE fts @@ websearch_to_tsquery('english', %s)
                      AND (%s::text[] = '{}' OR tags @> %s::text[])
                    ORDER BY score DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (clean_query, clean_query, clean_tags, clean_tags, capped_limit),
                )
                rows = await cur.fetchall()
                mode = "fts"

                if not rows:
                    like_pat = f"%{clean_query}%"
                    cur = await conn.execute(
                        """
                        SELECT id, key, content, tags, created_by_client_id, created_by_client_name,
                               created_at, updated_at,
                               GREATEST(similarity(content, %s), similarity(key, %s)) AS score
                        FROM memories
                        WHERE (
                            similarity(content, %s) > 0.1
                            OR similarity(key, %s) > 0.1
                            OR content ILIKE %s
                            OR key ILIKE %s
                        )
                          AND (%s::text[] = '{}' OR tags @> %s::text[])
                        ORDER BY score DESC, updated_at DESC
                        LIMIT %s
                        """,
                        (
                            clean_query,
                            clean_query,
                            clean_query,
                            clean_query,
                            like_pat,
                            like_pat,
                            clean_tags,
                            clean_tags,
                            capped_limit,
                        ),
                    )
                    rows = await cur.fetchall()
                    mode = "trigram"

        matches = [serialize_memory(dict(r), score=r["score"]) for r in rows]
        return {"mode": mode, "count": len(matches), "matches": matches}

    async def list_memories(
        self,
        *,
        tag: str = "",
        limit: int = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        clean_tag = (tag or "").strip()
        capped_limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        cursor_ts: datetime | None = None
        cursor_id: UUID | None = None
        if cursor and cursor.strip():
            cursor_ts, cursor_id = decode_list_cursor(cursor)

        where: list[str] = []
        params: list[Any] = []
        if clean_tag:
            where.append("tags @> ARRAY[%s]::text[]")
            params.append(clean_tag)
        if cursor_ts is not None and cursor_id is not None:
            where.append("(updated_at < %s OR (updated_at = %s AND id < %s))")
            params.extend([cursor_ts, cursor_ts, cursor_id])

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(capped_limit + 1)
        sql = f"""
            SELECT id, key, tags, created_by_client_name, updated_at
            FROM memories
            {where_sql}
            ORDER BY updated_at DESC, id DESC
            LIMIT %s
        """
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, tuple(params))
            rows = await cur.fetchall()

        has_more = len(rows) > capped_limit
        page_rows = rows[:capped_limit]
        next_cursor = (
            encode_list_cursor(page_rows[-1]["updated_at"], page_rows[-1]["id"])
            if has_more and page_rows
            else None
        )
        items = [serialize_memory_summary(dict(r)) for r in page_rows]
        return {"count": len(items), "items": items, "next_cursor": next_cursor}

    async def delete_memory(self, *, key: str) -> bool:
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("memory_delete requires non-empty string 'key'.")
        async with self.pool.connection() as conn:
            cur = await conn.execute("DELETE FROM memories WHERE key = %s", (clean_key,))
            deleted = (cur.rowcount or 0) > 0
            await conn.commit()
        return deleted

    async def browse_for_ui(
        self,
        *,
        query: str = "",
        tag: str = "",
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], int, list[tuple[str, int]]]:
        """Return `(rows_with_ids, total_count, [(tag, count), ...])` for the owner UI."""
        clean_query = (query or "").strip()
        clean_tag = (tag or "").strip()
        capped_limit = max(1, min(int(limit), 200))

        async with self.pool.connection() as conn:
            cnt_row = await (await conn.execute("SELECT COUNT(*) AS cnt FROM memories")).fetchone()
            total_count = int(cnt_row["cnt"]) if cnt_row else 0

            tag_cur = await conn.execute(
                """
                SELECT t AS tag, COUNT(*) AS cnt
                FROM memories, unnest(tags) AS t
                GROUP BY t
                ORDER BY cnt DESC, t ASC
                LIMIT 30
                """
            )
            tag_counts = [(str(r["tag"]), int(r["cnt"])) for r in await tag_cur.fetchall()]

            tag_arr = [clean_tag] if clean_tag else []
            if not clean_query:
                cur = await conn.execute(
                    """
                    SELECT id, key, content, tags, created_by_client_id, created_by_client_name,
                           created_at, updated_at
                    FROM memories
                    WHERE (%s::text[] = '{}' OR tags @> %s::text[])
                    ORDER BY updated_at DESC, id DESC
                    LIMIT %s
                    """,
                    (tag_arr, tag_arr, capped_limit),
                )
                rows = await cur.fetchall()
            else:
                cur = await conn.execute(
                    """
                    SELECT id, key, content, tags, created_by_client_id, created_by_client_name,
                           created_at, updated_at,
                           ts_rank(fts, websearch_to_tsquery('english', %s)) AS score
                    FROM memories
                    WHERE fts @@ websearch_to_tsquery('english', %s)
                      AND (%s::text[] = '{}' OR tags @> %s::text[])
                    ORDER BY score DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (clean_query, clean_query, tag_arr, tag_arr, capped_limit),
                )
                rows = await cur.fetchall()
                if not rows:
                    like_pat = f"%{clean_query}%"
                    cur = await conn.execute(
                        """
                        SELECT id, key, content, tags, created_by_client_id, created_by_client_name,
                               created_at, updated_at,
                               GREATEST(similarity(content, %s), similarity(key, %s)) AS score
                        FROM memories
                        WHERE (
                            similarity(content, %s) > 0.1
                            OR similarity(key, %s) > 0.1
                            OR content ILIKE %s
                            OR key ILIKE %s
                        )
                          AND (%s::text[] = '{}' OR tags @> %s::text[])
                        ORDER BY score DESC, updated_at DESC
                        LIMIT %s
                        """,
                        (
                            clean_query,
                            clean_query,
                            clean_query,
                            clean_query,
                            like_pat,
                            like_pat,
                            tag_arr,
                            tag_arr,
                            capped_limit,
                        ),
                    )
                    rows = await cur.fetchall()

        items: list[dict[str, Any]] = []
        for r in rows:
            item = serialize_memory(dict(r))
            item["id"] = str(r["id"])
            item["created_by_client_id"] = r["created_by_client_id"]
            items.append(item)
        return items, total_count, tag_counts
