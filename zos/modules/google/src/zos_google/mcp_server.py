"""Streamable-HTTP MCP server for ZOS Google modules (`gmail` or `gcal`)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any, Protocol

import mcp_types as types
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from zos_google.caller import CallerVerificationError, verify_mcp_caller


class GoogleToolHandler(Protocol):
    async def list_tools(self) -> list[types.Tool]: ...
    async def call_tool(self, name: str, args: dict[str, Any]) -> types.CallToolResult: ...


def _err(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        is_error=True,
    )


def create_mcp_app(
    handler: GoogleToolHandler,
    *,
    service: str,
    module_secret: str,
    mcp_path: str = "/mcp",
) -> Starlette:
    """Create the Starlette ASGI app serving Streamable-HTTP MCP for `gmail` or `gcal`."""

    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=await handler.list_tools())

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

        return await handler.call_tool(name, args)

    server = Server(
        f"zos-{service}",
        version="0.1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
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
        return JSONResponse({"ok": True, "module": service, "service": "mcp"})

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route(mcp_path, endpoint=StreamableHTTPASGIApp(session_manager), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )
