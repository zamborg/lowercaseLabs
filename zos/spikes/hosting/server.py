"""M0.2 throwaway hosting spike: minimal Streamable-HTTP MCP server.

Tools:
- ping: returns "pong"
- toggle: adds/removes `extra_tool` and broadcasts notifications/tools/list_changed
- extra_tool (when toggled on): returns "extra ok"
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mcp_types as types
import uvicorn
from mcp.server.connection import Connection
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zos.spike.hosting")


class SpikeServer(Server[Any]):
    """Low-level MCP Server that advertises tools.listChanged = True."""

    def create_initialization_options(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
        extensions: dict[str, dict[str, Any]] | None = None,
    ) -> InitializationOptions:
        return super().create_initialization_options(
            notification_options=notification_options or NotificationOptions(tools_changed=True),
            experimental_capabilities=experimental_capabilities,
            extensions=extensions,
        )


def build_app():
    state: dict[str, Any] = {
        "extra_enabled": False,
        "connections": set(),
    }

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        tools = [
            types.Tool(
                name="ping",
                description="Health check tool returning pong.",
                input_schema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="toggle",
                description="Toggle extra_tool on/off and emit notifications/tools/list_changed.",
                input_schema={"type": "object", "properties": {}},
            ),
        ]
        if state["extra_enabled"]:
            tools.append(
                types.Tool(
                    name="extra_tool",
                    description="Dynamically added/removed spike tool.",
                    input_schema={"type": "object", "properties": {}},
                )
            )
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name == "ping":
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="pong")],
            )
        if params.name == "toggle":
            state["extra_enabled"] = not state["extra_enabled"]
            conns: list[Connection] = list(state["connections"])
            for conn in conns:
                await conn.send_tool_list_changed()
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"extra_enabled={state['extra_enabled']} notified={len(conns)}",
                    )
                ],
            )
        if params.name == "extra_tool":
            if not state["extra_enabled"]:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text="extra_tool is disabled")],
                    is_error=True,
                )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="extra ok")],
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"unknown tool: {params.name}")],
            is_error=True,
        )

    async def on_initialized(ctx, params) -> None:
        conn: Connection = ctx.session._connection
        state["connections"].add(conn)
        conn.exit_stack.callback(state["connections"].discard, conn)
        logger.info(
            "Session initialized: id=%s active_connections=%d",
            conn.session_id,
            len(state["connections"]),
        )

    server = SpikeServer(
        "zos-hosting-spike",
        version="0.0.1",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    server.add_notification_handler(
        "notifications/initialized",
        types.NotificationParams,
        on_initialized,
    )

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "active_connections": len(state["connections"]),
                "extra_enabled": state["extra_enabled"],
            }
        )

    return server.streamable_http_app(
        streamable_http_path="/mcp",
        host="0.0.0.0",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        session_idle_timeout=1800,
        custom_starlette_routes=[Route("/healthz", healthz)],
    )


app = build_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=75,
    )
