"""Hosted Postgres-backed memory provider (`memory_put/get/search/list/delete`)."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any
from uuid import UUID

import mcp_types as types
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.auth.context import CallContext
from zos.memory_service import MemoryService
from zos.providers.base import ProviderKind, ProviderTool

MAX_SEARCH_LIMIT = 50
MAX_LIST_LIMIT = 100


def _encode_list_cursor(updated_at: datetime, row_id: UUID) -> str:
    payload = json.dumps({"u": updated_at.isoformat(), "i": str(row_id)}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_list_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        padded = cursor.strip() + "=" * ((-len(cursor.strip())) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        return datetime.fromisoformat(str(data["u"])), UUID(str(data["i"]))
    except Exception as exc:
        raise ValueError("Invalid memory_list cursor") from exc


def _serialize_memory(row: dict[str, Any], *, score: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": row["key"],
        "content": row["content"],
        "tags": list(row["tags"] or []),
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        "author": row.get("author"),
    }
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


def _serialize_memory_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": row["key"],
        "tags": list(row["tags"] or []),
        "updated_at": row["updated_at"].isoformat(),
        "author": row.get("author"),
    }


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
                name="list",
                description="List memory keys, tags, updated_at, and author (without full content) ordered by updated_at desc.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string", "description": "Optional single tag filter."},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_LIST_LIMIT,
                            "default": 20,
                        },
                        "cursor": {"type": "string", "description": "Opaque pagination cursor from a previous list call."},
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
        if self._provider_id is None:
            await self._resolve_provider_id(ctx.principal_id)
        if name == "put":
            return await self._put(args, ctx)
        if name == "get":
            return await self._get(args, ctx)
        if name == "search":
            return await self._search(args, ctx)
        if name == "list":
            return await self._list(args, ctx)
        if name == "delete":
            return await self._delete(args, ctx)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Unknown memory tool: {name}")],
            is_error=True,
        )

    async def _resolve_provider_id(self, principal_id: UUID) -> None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id FROM providers WHERE principal_id = %s AND slug = 'memory' AND kind = 'hosted' LIMIT 1",
                (principal_id,),
            )
            row = await cur.fetchone()
        if row is not None:
            self._provider_id = row["id"]

    async def _put(self, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        key = str(args.get("key", "")).strip()
        content = args.get("content")
        if not key or not isinstance(content, str):
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="memory_put requires non-empty string 'key' and 'content'.")],
                is_error=True,
            )
        raw_tags = args.get("tags") or []
        row, _status = await MemoryService(self.pool).put_memory(
            principal_id=ctx.principal_id,
            space_id=ctx.space_id,
            client_id=ctx.client_id,
            key=key,
            content=content,
            tags=raw_tags,
        )
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
                SELECT m.key, m.content, m.tags, m.created_at, m.updated_at, c.name AS author
                FROM memories m
                LEFT JOIN clients c ON c.id = m.created_by_client_id
                WHERE m.principal_id = %s AND m.space_id = %s AND m.key = %s
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
                    SELECT m.key, m.content, m.tags, m.created_at, m.updated_at,
                           c.name AS author, 1.0::float AS score
                    FROM memories m
                    LEFT JOIN clients c ON c.id = m.created_by_client_id
                    WHERE m.principal_id = %s AND m.space_id = %s
                      AND (%s::text[] = '{}' OR m.tags @> %s::text[])
                    ORDER BY m.updated_at DESC
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
                    SELECT m.key, m.content, m.tags, m.created_at, m.updated_at,
                           c.name AS author,
                           ts_rank(m.fts, websearch_to_tsquery('english', %s)) AS score
                    FROM memories m
                    LEFT JOIN clients c ON c.id = m.created_by_client_id
                    WHERE m.principal_id = %s AND m.space_id = %s
                      AND m.fts @@ websearch_to_tsquery('english', %s)
                      AND (%s::text[] = '{}' OR m.tags @> %s::text[])
                    ORDER BY score DESC, m.updated_at DESC
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
                        SELECT m.key, m.content, m.tags, m.created_at, m.updated_at,
                               c.name AS author,
                               GREATEST(similarity(m.content, %s), similarity(m.key, %s)) AS score
                        FROM memories m
                        LEFT JOIN clients c ON c.id = m.created_by_client_id
                        WHERE m.principal_id = %s AND m.space_id = %s
                          AND (
                              similarity(m.content, %s) > 0.1
                              OR similarity(m.key, %s) > 0.1
                              OR m.content ILIKE %s
                              OR m.key ILIKE %s
                          )
                          AND (%s::text[] = '{}' OR m.tags @> %s::text[])
                        ORDER BY score DESC, m.updated_at DESC
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

    async def _list(self, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        tag = str(args.get("tag") or "").strip()
        try:
            limit = int(args.get("limit", 20))
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, MAX_LIST_LIMIT))

        cursor_raw = str(args.get("cursor") or "").strip()
        cursor_ts: datetime | None = None
        cursor_id: UUID | None = None
        if cursor_raw:
            try:
                cursor_ts, cursor_id = _decode_list_cursor(cursor_raw)
            except ValueError as exc:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(exc))],
                    is_error=True,
                )

        where = ["m.principal_id = %s", "m.space_id = %s"]
        params: list[Any] = [ctx.principal_id, ctx.space_id]
        if tag:
            where.append("m.tags @> ARRAY[%s]::text[]")
            params.append(tag)
        if cursor_ts is not None and cursor_id is not None:
            where.append("(m.updated_at < %s OR (m.updated_at = %s AND m.id < %s))")
            params.extend([cursor_ts, cursor_ts, cursor_id])

        params.append(limit + 1)
        sql = f"""
            SELECT m.id, m.key, m.tags, m.updated_at, c.name AS author
            FROM memories m
            LEFT JOIN clients c ON c.id = m.created_by_client_id
            WHERE {' AND '.join(where)}
            ORDER BY m.updated_at DESC, m.id DESC
            LIMIT %s
        """
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, tuple(params))
            rows = await cur.fetchall()

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        next_cursor = (
            _encode_list_cursor(page_rows[-1]["updated_at"], page_rows[-1]["id"])
            if has_more and page_rows
            else None
        )
        items = [_serialize_memory_summary(r) for r in page_rows]
        payload = {"count": len(items), "items": items, "next_cursor": next_cursor}
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
