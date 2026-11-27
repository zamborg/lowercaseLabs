"""Streamable-HTTP MCP server for the ZOS Tasks module (`list`, `get`, `add`, `update`, `complete`)."""

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

from zos_tasks.caller import CallerVerificationError, verify_mcp_caller
from zos_tasks.store import _UNSET, TasksStore

RESERVED_MCP_CALLERS = frozenset({"owner"})

TASKS_TOOLS: list[types.Tool] = [
    types.Tool(
        name="list",
        description=(
            "List todos on the human owner's personal todo list (things the human needs to do, "
            "not an agent work queue). Filter by status, due_before, tag, and paginate with limit/cursor."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "done", "cancelled", "all"],
                    "default": "open",
                    "description": "Filter by task status (default: 'open').",
                },
                "due_before": {
                    "type": "string",
                    "description": "Optional ISO date (YYYY-MM-DD); returns tasks due on or before this date.",
                },
                "tag": {
                    "type": "string",
                    "description": "Optional single tag filter.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                    "description": "Maximum number of tasks to return (1-100).",
                },
                "cursor": {
                    "type": "string",
                    "description": "Pagination cursor from a previous list response.",
                },
            },
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            open_world_hint=False,
        ),
    ),
    types.Tool(
        name="get",
        description=(
            "Get full details for a single todo on the human owner's personal todo list by id "
            "(for reviewing a human action item on the owner's behalf)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {
                    "anyOf": [{"type": "integer"}, {"type": "string"}],
                    "description": "Numeric task ID.",
                },
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            open_world_hint=False,
        ),
    ),
    types.Tool(
        name="add",
        description=(
            "Add a todo for the human owner (something THEY need to do). "
            "Only add when the user asks or clearly needs to act. "
            "Never use this to track your own agent work."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short, actionable title for what the human owner needs to do.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional extra context or details for the human.",
                },
                "due": {
                    "type": "string",
                    "description": "Optional due date in ISO format (YYYY-MM-DD).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of short tags (e.g. ['errands', 'health']).",
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            open_world_hint=False,
        ),
    ),
    types.Tool(
        name="update",
        description=(
            "Update a todo on the human owner's personal todo list (title, notes, due date, tags, or status) "
            "on the owner's behalf. Never use this as an agent scratchpad."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {
                    "anyOf": [{"type": "integer"}, {"type": "string"}],
                    "description": "Numeric task ID.",
                },
                "title": {"type": "string", "description": "Updated task title."},
                "notes": {"type": "string", "description": "Updated notes."},
                "due": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Updated ISO due date (YYYY-MM-DD), or null/empty string to clear.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replacement list of tags.",
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "done", "cancelled"],
                    "description": "Updated task status.",
                },
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            open_world_hint=False,
        ),
    ),
    types.Tool(
        name="complete",
        description=(
            "Mark a todo on the human owner's personal todo list as done "
            "when the human has completed it or asks you to check it off."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {
                    "anyOf": [{"type": "integer"}, {"type": "string"}],
                    "description": "Numeric task ID.",
                },
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            open_world_hint=False,
        ),
    ),
]


def _ok_json(data: Any) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]
    )


def _err(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        is_error=True,
    )


def create_mcp_app(
    store: TasksStore,
    *,
    module_secret: str,
    mcp_path: str = "/mcp",
) -> Starlette:
    """Create the Starlette ASGI app serving the Tasks Streamable-HTTP MCP server."""

    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list(TASKS_TOOLS))

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

        if caller.client_name.strip().lower() in RESERVED_MCP_CALLERS:
            return _err(
                f"CALLER_RESERVED: MCP client name '{caller.client_name.strip()}' is reserved; "
                "only the owner HTTP UI may act as 'owner'."
            )

        try:
            if name == "list":
                status = args.get("status", "open")
                due_before = args.get("due_before")
                tag = args.get("tag")
                limit = int(args.get("limit", 50))
                cursor = args.get("cursor")
                res = await store.list_tasks(
                    status=str(status) if status is not None else "open",
                    due_before=str(due_before) if due_before is not None else None,
                    tag=str(tag) if tag is not None else None,
                    limit=limit,
                    cursor=str(cursor) if cursor is not None else None,
                )
                return _ok_json(res)

            if name == "get":
                task_id = int(str(args["id"]))
                task = await store.get_task(task_id)
                return _ok_json({"task": task})

            if name == "add":
                title = str(args.get("title") or "")
                notes = args.get("notes")
                due = args.get("due")
                raw_tags = args.get("tags")
                if raw_tags is not None and (
                    not isinstance(raw_tags, list) or not all(isinstance(x, str) for x in raw_tags)
                ):
                    return _err("INVALID_ARGUMENT: 'tags' must be a list of strings.")
                task = await store.add_task(
                    title=title,
                    created_by=caller.client_name,
                    notes=str(notes) if notes is not None else None,
                    due=str(due) if due is not None else None,
                    tags=raw_tags,
                )
                return _ok_json({"added": True, "task": task})

            if name == "update":
                task_id = int(str(args["id"]))
                raw_tags = args.get("tags", _UNSET)
                if raw_tags is not _UNSET and raw_tags is not None and (
                    not isinstance(raw_tags, list) or not all(isinstance(x, str) for x in raw_tags)
                ):
                    return _err("INVALID_ARGUMENT: 'tags' must be a list of strings.")
                task = await store.update_task(
                    task_id,
                    title=str(args["title"]) if "title" in args else _UNSET,
                    notes=(str(args["notes"]) if args["notes"] is not None else "") if "notes" in args else _UNSET,
                    due=(str(args["due"]) if args["due"] is not None else None) if "due" in args else _UNSET,
                    tags=raw_tags,
                    status=str(args["status"]) if "status" in args else _UNSET,
                )
                return _ok_json({"updated": True, "task": task})

            if name == "complete":
                task_id = int(str(args["id"]))
                task = await store.complete_task(task_id)
                return _ok_json({"completed": True, "task": task})

            return _err(f"Unknown tool: {name}")
        except KeyError as exc:
            return _err(f"NOT_FOUND: {exc.args[0] if exc.args else exc}")
        except (ValueError, TypeError) as exc:
            return _err(f"INVALID_ARGUMENT: {exc}")

    server = Server("zos-tasks", version="0.1.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
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
        return JSONResponse({"ok": True, "module": "tasks", "service": "mcp"})

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route(mcp_path, endpoint=StreamableHTTPASGIApp(session_manager), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )
