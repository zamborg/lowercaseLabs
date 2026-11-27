"""Streamable-HTTP MCP server for the ZOS Files module (`list`, `read`, `write`, `delete`, `move`).

Exposed through ZOS as `files_list`, `files_read`, `files_write`, `files_delete`, `files_move`.
"""

from __future__ import annotations

import base64
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

from zos_files.caller import CallerVerificationError, verify_mcp_caller
from zos_files.store import (
    ConflictError,
    FilesStore,
    FileTooLargeError,
    PathError,
    QuotaExceededError,
    decode_payload,
    is_text,
)

# "owner" is the identity the authenticated HTTP UI uses for `updated_by`.
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024

RESERVED_MCP_CALLERS = frozenset({"owner"})

SKILLS_NOTE = (
    "Shared drive for the owner and all agents. Convention: agent skills live at "
    "'skills/<name>/SKILL.md' plus supporting files under 'skills/<name>/', so every agent "
    "on every machine can sync the same skills. Paths are relative and '/'-separated "
    "(no leading '/', no '..'); folders are just path prefixes."
)


def _ann(*, read_only: bool, destructive: bool = False) -> types.ToolAnnotations:
    return types.ToolAnnotations(
        read_only_hint=read_only, destructive_hint=destructive, open_world_hint=False
    )


FILES_TOOLS: list[types.Tool] = [
    types.Tool(
        name="list",
        description=(
            f"List files in the shared drive, e.g. prefix='skills/' to enumerate skills. {SKILLS_NOTE} "
            "Returns path, size, sha256, updated_at, updated_by. With recursive=false, returns "
            "direct children and folders (type 'folder', path ends with '/')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Folder prefix such as 'skills/' or 'skills/my-skill'. Default: root.",
                },
                "recursive": {
                    "type": "boolean",
                    "default": True,
                    "description": "List every file under the prefix (true) or only direct children (false).",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                "cursor": {
                    "type": "string",
                    "description": "next_cursor from a previous list response.",
                },
            },
            "additionalProperties": False,
        },
        annotations=_ann(read_only=True),
    ),
    types.Tool(
        name="read",
        description=(
            f"Read a file from the shared drive (e.g. 'skills/<name>/SKILL.md'). {SKILLS_NOTE} "
            "Text files return encoding 'utf-8' with the text; binary files return encoding "
            "'base64'. Always includes sha256 (pass it as if_sha256 to files_write to avoid "
            "clobbering concurrent edits)."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path."}},
            "required": ["path"],
            "additionalProperties": False,
        },
        annotations=_ann(read_only=True),
    ),
    types.Tool(
        name="write",
        description=(
            f"Create or overwrite a file in the shared drive. {SKILLS_NOTE} "
            "Max 10 MiB per file plus a total quota. Pass if_sha256 (the sha256 you last read) "
            "for optimistic concurrency: the write is refused with CONFLICT if the file changed "
            "(or does not exist). Parent folders are implicit."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path, e.g. 'skills/my-skill/SKILL.md'.",
                },
                "content": {"type": "string", "description": "File content."},
                "encoding": {
                    "type": "string",
                    "enum": ["utf-8", "base64"],
                    "default": "utf-8",
                    "description": "How 'content' is encoded. Use base64 for binary files.",
                },
                "if_sha256": {
                    "type": "string",
                    "description": "Only write if the current file has this sha256.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        annotations=_ann(read_only=False, destructive=True),
    ),
    types.Tool(
        name="delete",
        description=f"Delete a single file from the shared drive. {SKILLS_NOTE}",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path."}},
            "required": ["path"],
            "additionalProperties": False,
        },
        annotations=_ann(read_only=False, destructive=True),
    ),
    types.Tool(
        name="move",
        description=(
            f"Move/rename a file in the shared drive. Refuses to overwrite an existing destination. "
            f"{SKILLS_NOTE}"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Existing file path."},
                "to": {"type": "string", "description": "New file path."},
            },
            "required": ["from", "to"],
            "additionalProperties": False,
        },
        annotations=_ann(read_only=False),
    ),
]


def _ok_json(data: Any) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]
    )


def _err(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], is_error=True)


def _req_str(args: dict[str, Any], key: str) -> str:
    val = args.get(key)
    if not isinstance(val, str):
        raise ValueError(f"'{key}' is required and must be a string")
    return val


def create_mcp_app(
    store: FilesStore,
    *,
    module_secret: str,
    mcp_path: str = "/mcp",
) -> Starlette:
    """Create the Starlette ASGI app serving the Files Streamable-HTTP MCP server."""

    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list(FILES_TOOLS))

    async def on_call_tool(rctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        name = params.name
        args = dict(params.arguments or {})

        try:
            caller = verify_mcp_caller(params.meta, secret=module_secret, tool=name, args=args)
        except CallerVerificationError as exc:
            return _err(f"CALLER_VERIFICATION_FAILED: {exc}")

        if caller is None:
            return _err(f"CALLER_REQUIRED: unsigned system calls to '{name}' are refused.")
        who = caller.client_name.strip()
        if who.lower() in RESERVED_MCP_CALLERS:
            return _err(
                f"CALLER_RESERVED: MCP client name '{who}' is reserved; "
                "only the owner HTTP UI may act as 'owner'."
            )

        try:
            if name == "list":
                raw_prefix = args.get("prefix")
                cursor = args.get("cursor")
                res = await store.list(
                    prefix=str(raw_prefix) if raw_prefix is not None else None,
                    recursive=bool(args.get("recursive", True)),
                    limit=int(args.get("limit", 100)),
                    cursor=str(cursor) if cursor is not None else None,
                )
                return _ok_json(res)

            if name == "read":
                meta, data = await store.read(_req_str(args, "path"))
                out = dict(meta)
                if is_text(data):
                    out["encoding"] = "utf-8"
                    out["content"] = data.decode("utf-8")
                else:
                    out["encoding"] = "base64"
                    out["content"] = base64.b64encode(data).decode("ascii")
                return _ok_json(out)

            if name == "write":
                data = decode_payload(_req_str(args, "content"), str(args.get("encoding", "utf-8")))
                if_sha = args.get("if_sha256")
                meta = await store.write(
                    _req_str(args, "path"),
                    data,
                    updated_by=who,
                    if_sha256=str(if_sha) if if_sha is not None else None,
                )
                return _ok_json({"written": True, "file": meta})

            if name == "delete":
                meta = await store.delete(_req_str(args, "path"))
                return _ok_json({"deleted": True, "file": meta})

            if name == "move":
                meta = await store.move(
                    _req_str(args, "from"), _req_str(args, "to"), updated_by=who
                )
                return _ok_json({"moved": True, "file": meta})

            return _err(f"Unknown tool: {name}")
        except PathError as exc:
            return _err(f"INVALID_PATH: {exc}")
        except FileTooLargeError as exc:
            return _err(f"FILE_TOO_LARGE: {exc}")
        except QuotaExceededError as exc:
            return _err(f"QUOTA_EXCEEDED: {exc}")
        except ConflictError as exc:
            detail = f" (current_sha256={exc.current_sha256})" if exc.current_sha256 else ""
            return _err(f"CONFLICT: {exc}{detail}")
        except KeyError as exc:
            return _err(f"NOT_FOUND: {exc.args[0] if exc.args else exc}")
        except (ValueError, TypeError) as exc:
            return _err(f"INVALID_ARGUMENT: {exc}")

    server = Server("zos-files", version="0.1.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
        # Default is 4 MiB; a 10 MiB file base64-encoded is ~14 MiB.
        max_request_body_size=MAX_REQUEST_BODY_BYTES,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "module": "files", "service": "mcp"})

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route(
                mcp_path,
                endpoint=StreamableHTTPASGIApp(session_manager),
                methods=["GET", "POST", "DELETE"],
            ),
        ],
        lifespan=lifespan,
    )
