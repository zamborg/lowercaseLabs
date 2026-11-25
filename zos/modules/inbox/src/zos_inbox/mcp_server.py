"""Streamable-HTTP MCP server for the ZOS Inbox module (`send`, `list`, `read`, `archive`, `directory`)."""

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

from zos_inbox.caller import CallerVerificationError, verify_mcp_caller
from zos_inbox.store import InboxStore

RESERVED_MCP_CALLERS = frozenset({"owner", "general"})

INBOX_TOOLS: list[types.Tool] = [
    types.Tool(
        name="send",
        description=(
            "Send an async email-like message to one or more mailboxes (e.g. ['owner'], ['general'], "
            "or another client's mailbox name). Sender is automatically set from your verified identity."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Recipient mailbox names (e.g. ['owner'], ['general'], ['codex']).",
                },
                "subject": {"type": "string", "description": "Message subject line."},
                "body": {"type": "string", "description": "Message body text."},
                "thread_id": {
                    "type": "string",
                    "description": "Optional existing thread UUID to reply to.",
                },
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False),
    ),
    types.Tool(
        name="list",
        description=(
            "List messages in your own mailbox (default) or 'general'. "
            "Clients cannot list another client's private mailbox."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "mailbox": {
                    "type": "string",
                    "description": "Mailbox to list (defaults to your own mailbox; 'general' is also allowed).",
                },
                "unread_only": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return only unread messages for you (default: true).",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                    "description": "Maximum number of messages to return (1-100).",
                },
                "cursor": {
                    "type": "string",
                    "description": "Pagination cursor from a previous list response.",
                },
            },
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False),
    ),
    types.Tool(
        name="read",
        description="Read a message by id and mark it as read for you.",
        input_schema={
            "type": "object",
            "properties": {
                "id": {
                    "anyOf": [{"type": "integer"}, {"type": "string"}],
                    "description": "Numeric message ID.",
                },
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False),
    ),
    types.Tool(
        name="archive",
        description="Archive a message by id for your mailbox.",
        input_schema={
            "type": "object",
            "properties": {
                "id": {
                    "anyOf": [{"type": "integer"}, {"type": "string"}],
                    "description": "Numeric message ID.",
                },
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False),
    ),
    types.Tool(
        name="directory",
        description="List known mailboxes, last activity timestamps, and unread counts for you.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False),
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
    store: InboxStore,
    *,
    module_secret: str,
    mcp_path: str = "/mcp",
) -> Starlette:
    """Create the Starlette ASGI app serving the Inbox Streamable-HTTP MCP server."""

    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list(INBOX_TOOLS))

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

        if caller is not None and caller.client_name.strip().lower() in RESERVED_MCP_CALLERS:
            return _err(
                f"CALLER_RESERVED: MCP client name '{caller.client_name.strip()}' is reserved; "
                "only the owner HTTP UI may act as 'owner'."
            )

        if name == "directory":
            entries = await store.directory(
                caller=caller.client_name if caller is not None else None
            )
            return _ok_json(
                {
                    "caller": caller.client_name if caller is not None else None,
                    "mailboxes": entries,
                }
            )

        # Unsigned (system) calls to send, list, read, or archive are strictly refused
        if caller is None:
            return _err(
                f"CALLER_REQUIRED: unsigned system calls to '{name}' are refused."
            )

        try:
            if name == "send":
                raw_to = args.get("to")
                if not isinstance(raw_to, list) or not all(isinstance(x, str) for x in raw_to):
                    return _err("Invalid 'to': expected a non-empty list of mailbox names.")
                subject = str(args.get("subject") or "")
                body = str(args.get("body") or "")
                thread_id = args.get("thread_id")
                msg = await store.send_message(
                    sender=caller.client_name,
                    to=raw_to,
                    subject=subject,
                    body=body,
                    thread_id=str(thread_id) if thread_id is not None else None,
                )
                return _ok_json({"sent": True, "message": msg})

            if name == "list":
                mb = args.get("mailbox")
                unread_only = bool(args.get("unread_only", True))
                limit = int(args.get("limit", 20))
                cursor = args.get("cursor")
                res = await store.list_messages(
                    caller=caller.client_name,
                    mailbox=str(mb) if mb is not None else None,
                    unread_only=unread_only,
                    limit=limit,
                    cursor=str(cursor) if cursor is not None else None,
                )
                return _ok_json(res)

            if name == "read":
                msg_id = int(str(args["id"]))
                msg = await store.read_message(caller=caller.client_name, message_id=msg_id)
                return _ok_json({"message": msg})

            if name == "archive":
                msg_id = int(str(args["id"]))
                msg = await store.archive_message(caller=caller.client_name, message_id=msg_id)
                return _ok_json({"archived": True, "message": msg})

            return _err(f"Unknown tool: {name}")
        except PermissionError as exc:
            return _err(f"PERMISSION_DENIED: {exc}")
        except KeyError as exc:
            return _err(f"NOT_FOUND: {exc.args[0] if exc.args else exc}")
        except (ValueError, TypeError) as exc:
            return _err(f"INVALID_ARGUMENT: {exc}")

    server = Server("zos-inbox", version="0.1.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
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
        return JSONResponse({"ok": True, "module": "inbox", "service": "mcp"})

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route(mcp_path, endpoint=StreamableHTTPASGIApp(session_manager), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )
