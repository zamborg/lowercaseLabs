"""Hosted Postgres-backed memory provider (`memory_put/get/search/delete`)."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import mcp_types as types
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.auth.context import CallContext
from zos.providers.base import ProviderKind, ProviderTool

MAX_SEARCH_LIMIT = 50


def _serialize_memory(row: dict[str, Any], *, score: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": row["key"],
        "content": row["content"],
        "tags": list(row["tags"] or []),
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


class MemoryProvider:
    """Hosted memory ToolProvider backed by Postgres FTS + pg_trgm."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        *,
        provider_id: UUID | None = None,
    ) -> None:
        self.pool = pool
        self._provider_id = provider_id

    @property
    def namespace(self) -> str:
        return "memory"

    @property
    def kind(self) -> ProviderKind:
        return "hosted"

    @property
    def provider_id(self) -> UUID | None:
        return self._provider_id

    def is_online(self) -> bool:
        return True

    async def list_tools(self) -> list[ProviderTool]:
        return [
            ProviderTool(
                name="put",
                description="Store or update a memory entry by key.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Unique key within the space."},
                        "content": {"type": "string", "description": "Memory text content."},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for filtering.",
                        },
                    },
                    "required": ["key", "content"],
                },
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
                name="get",
                description="Fetch a memory entry by exact key.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Exact memory key."},
                    },
                    "required": ["key"],
                },
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
                name="search",
                description="Search memories using full-text search with trigram fallback and optional tag filters.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string (optional if tags provided)."},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter to memories containing all specified tags.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SEARCH_LIMIT,
                            "default": 20,
                        },
                    },
                },
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
                name="delete",
                description="Delete a memory entry by exact key.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Exact memory key to delete."},
                    },
                    "required": ["key"],
                },
                destructive_hint=True,
                idempotent_hint=True,
            ),
        ]

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        ctx: CallContext,
    ) -> types.CallToolResult:
        if name == "put":
            return await self._put(args, ctx)
        if name == "get":
            return await self._get(args, ctx)
        if name == "search":
            return await self._search(args, ctx)
        if name == "delete":
            return await self._delete(args, ctx)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Unknown memory tool: {name}")],
            is_error=True,
        )

    async def _put(self, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        key = str(args.get("key", "")).strip()
        content = args.get("content")
        if not key or not isinstance(content, str):
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="memory_put requires non-empty string 'key' and 'content'.")],
                is_error=True,
            )
        raw_tags = args.get("tags") or []
        tags = sorted({str(t).strip() for t in raw_tags if str(t).strip()})

        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO memories (principal_id, space_id, key, content, tags, created_by_client_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (space_id, key) DO UPDATE
                SET content = EXCLUDED.content,
                    tags = EXCLUDED.tags,
                    updated_at = now()
                RETURNING key, content, tags, created_at, updated_at
                """,
                (ctx.principal_id, ctx.space_id, key, content, tags, ctx.client_id),
            )
            row = await cur.fetchone()
            await conn.commit()
        assert row is not None
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(_serialize_memory(row), indent=2))]
        )

    async def _get(self, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        key = str(args.get("key", "")).strip()
        if not key:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="memory_get requires non-empty string 'key'.")],
                is_error=True,
            )
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT key, content, tags, created_at, updated_at
                FROM memories
                WHERE principal_id = %s AND space_id = %s AND key = %s
                """,
                (ctx.principal_id, ctx.space_id, key),
            )
            row = await cur.fetchone()

        if row is None:
            payload: dict[str, Any] = {"found": False, "key": key}
        else:
            payload = {"found": True, "memory": _serialize_memory(row)}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, indent=2))]
        )

    async def _search(self, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        query = str(args.get("query", "") or "").strip()
        raw_tags = args.get("tags") or []
        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        try:
            limit = int(args.get("limit", 20))
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))

        async with self.pool.connection() as conn:
            if not query:
                cur = await conn.execute(
                    """
                    SELECT key, content, tags, created_at, updated_at, 1.0::float AS score
                    FROM memories
                    WHERE principal_id = %s AND space_id = %s
                      AND (%s::text[] = '{}' OR tags @> %s::text[])
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (ctx.principal_id, ctx.space_id, tags, tags, limit),
                )
                rows = await cur.fetchall()
                mode = "recent"
            else:
                # 1. Primary: Postgres full-text search
                cur = await conn.execute(
                    """
                    SELECT key, content, tags, created_at, updated_at,
                           ts_rank(fts, websearch_to_tsquery('english', %s)) AS score
                    FROM memories
                    WHERE principal_id = %s AND space_id = %s
                      AND fts @@ websearch_to_tsquery('english', %s)
                      AND (%s::text[] = '{}' OR tags @> %s::text[])
                    ORDER BY score DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (query, ctx.principal_id, ctx.space_id, query, tags, tags, limit),
                )
                rows = await cur.fetchall()
                mode = "fts"

                # 2. Fallback: pg_trgm similarity + substring match for typos / partial tokens
                if not rows:
                    like_pat = f"%{query}%"
                    cur = await conn.execute(
                        """
                        SELECT key, content, tags, created_at, updated_at,
                               GREATEST(similarity(content, %s), similarity(key, %s)) AS score
                        FROM memories
                        WHERE principal_id = %s AND space_id = %s
                          AND (
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
                            query,
                            query,
                            ctx.principal_id,
                            ctx.space_id,
                            query,
                            query,
                            like_pat,
                            like_pat,
                            tags,
                            tags,
                            limit,
                        ),
                    )
                    rows = await cur.fetchall()
                    mode = "trigram"

        matches = [_serialize_memory(r, score=r["score"]) for r in rows]
        payload = {"mode": mode, "count": len(matches), "matches": matches}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, indent=2))]
        )

    async def _delete(self, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        key = str(args.get("key", "")).strip()
        if not key:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="memory_delete requires non-empty string 'key'.")],
                is_error=True,
            )
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                DELETE FROM memories
                WHERE principal_id = %s AND space_id = %s AND key = %s
                """,
                (ctx.principal_id, ctx.space_id, key),
            )
            deleted = (cur.rowcount or 0) > 0
            await conn.commit()

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps({"deleted": deleted, "key": key}))]
        )
