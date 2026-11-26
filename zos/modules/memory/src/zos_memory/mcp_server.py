"""Streamable-HTTP MCP server for the ZOS Memory module (`put`, `get`, `search`, `list`, `delete`)."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import mcp_types as types
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from zos_memory.caller import CallerVerificationError, verify_mcp_caller
from zos_memory.store import MAX_LIST_LIMIT, MAX_SEARCH_LIMIT, MemoryStore

MEMORY_TOOLS: list[types.Tool] = [
    types.Tool(
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
        annotations=types.ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    types.Tool(
        name="get",
        description="Fetch a memory entry by exact key.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Exact memory key."},
            },
            "required": ["key"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    types.Tool(
        name="search",
        description="Search memories using full-text search with trigram fallback and optional tag filters.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string (optional if tags provided).",
                },
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
        annotations=types.ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    types.Tool(
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
                "cursor": {
                    "type": "string",
                    "description": "Opaque pagination cursor from a previous list call.",
                },
            },
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    types.Tool(
        name="delete",
        description="Delete a memory entry by exact key.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Exact memory key to delete."},
            },
            "required": ["key"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
]


def _ok_json(data: Any, *, indent: int | None = 2) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, indent=indent, ensure_ascii=False))]
    )


def _err(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        is_error=True,
    )


def create_mcp_app(
    store: MemoryStore,
    *,
    module_secret: str,
    mcp_path: str = "/mcp",
) -> Starlette:
    """Create the Starlette ASGI app serving the Memory Streamable-HTTP MCP server."""

    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list(MEMORY_TOOLS))

    async def on_call_tool(rctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        name = params.name
        args = dict(params.arguments or {})
        meta = params.meta

        try:
            caller = verify_mcp_caller(
                meta,
                secret=module_secret,
                tool=name,
                args=args,
            )
        except CallerVerificationError as exc:
            return _err(f"CALLER_VERIFICATION_FAILED: {exc}")

        if caller is None:
            return _err(f"CALLER_REQUIRED: unsigned system calls to '{name}' are refused.")

        try:
            if name == "put":
                key = str(args.get("key", "")).strip()
                content = args.get("content")
                if not key or not isinstance(content, str):
                    return _err("memory_put requires non-empty string 'key' and 'content'.")
                raw_tags = args.get("tags") or []
                row, _status = await store.put_memory(
                    key=key,
                    content=content,
                    tags=raw_tags,
                    client_id=caller.client_id,
                    client_name=caller.client_name,
                )
                return _ok_json(row)

            if name == "get":
                key = str(args.get("key", "")).strip()
                if not key:
                    return _err("memory_get requires non-empty string 'key'.")
                row = await store.get_memory(key=key)
                if row is None:
                    return _ok_json({"found": False, "key": key})
                return _ok_json({"found": True, "memory": row})

            if name == "search":
                query = str(args.get("query", "") or "").strip()
                raw_tags = args.get("tags") or []
                try:
                    limit = int(args.get("limit", 20))
                except (TypeError, ValueError):
                    limit = 20
                res = await store.search_memories(query=query, tags=raw_tags, limit=limit)
                return _ok_json(res)

            if name == "list":
                tag = str(args.get("tag") or "").strip()
                try:
                    limit = int(args.get("limit", 20))
                except (TypeError, ValueError):
                    limit = 20
                cursor = str(args.get("cursor") or "").strip() or None
                res = await store.list_memories(tag=tag, limit=limit, cursor=cursor)
                return _ok_json(res)

            if name == "delete":
                key = str(args.get("key", "")).strip()
                if not key:
                    return _err("memory_delete requires non-empty string 'key'.")
                deleted = await store.delete_memory(key=key)
                return _ok_json({"deleted": deleted, "key": key}, indent=None)

            return _err(f"Unknown memory tool: {name}")
        except ValueError as exc:
            return _err(str(exc))

    server = Server("zos-memory", version="0.1.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "module": "memory", "service": "mcp"})

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route(mcp_path, endpoint=StreamableHTTPASGIApp(session_manager), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )
