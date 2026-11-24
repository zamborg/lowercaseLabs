"""ZOS ASGI application mounting Streamable HTTP MCP at /mcp and /healthz."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import mcp_types as types
import uvicorn
from mcp.server.connection import Connection
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Route

from zos.audit import DatabaseAuditRecorder
from zos.auth import BearerAuthASGIMiddleware, CallContext, current_call_context
from zos.auth.oauth import create_oauth_routes
from zos.config import Settings
from zos.db import create_pool, run_migrations
from zos.policy import DatabasePolicyEvaluator, request_tool_confirmation
from zos.providers.actuator import ActuatorManager
from zos.providers.base import ToolProvider
from zos.providers.google import GmailProvider, GoogleCalendarProvider, GoogleServiceHub
from zos.providers.memory import MemoryProvider
from zos.router import ProviderRegistry, SessionTracker, ToolRouter

logger = logging.getLogger(__name__)


class ZosMCPServer(Server[Any]):
    """Low-level MCP Server advertising tools.listChanged = True."""

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


@dataclass
class ZosRuntime:
    settings: Settings
    pool: AsyncConnectionPool[AsyncConnection[DictRow]]
    registry: ProviderRegistry
    sessions: SessionTracker
    router: ToolRouter
    mcp_server: ZosMCPServer
    session_manager: StreamableHTTPSessionManager
    actuator_manager: ActuatorManager
    google_hub: GoogleServiceHub


def _extract_call_context(rctx: Any) -> CallContext:
    req = getattr(rctx, "request", None)
    if req is not None:
        scope = getattr(req, "scope", None)
        if isinstance(scope, dict) and "zos_call_context" in scope:
            return scope["zos_call_context"]
    ctx = current_call_context.get()
    if ctx is None:
        raise RuntimeError("Missing CallContext on authenticated MCP request")
    return ctx


def create_app(
    settings: Settings | None = None,
    *,
    extra_providers: list[ToolProvider] | None = None,
    extra_routes_factory: Callable[[ZosRuntime], list[BaseRoute]] | None = None,
    run_migrations_on_startup: bool = True,
) -> Starlette:
    """Create the ZOS Starlette ASGI application."""
    cfg = settings or Settings.from_env()
    pool = create_pool(cfg.database_url, min_size=1, max_size=10)
    registry = ProviderRegistry()
    sessions = SessionTracker()
    policy_evaluator = DatabasePolicyEvaluator(pool)
    audit_recorder = DatabaseAuditRecorder(pool)
    router = ToolRouter(
        registry,
        sessions,
        load_policy_snapshot=policy_evaluator.load_snapshot,
        confirm_call=request_tool_confirmation,
        record_audit=audit_recorder.record_tool_call,
    )

    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        call_ctx = _extract_call_context(rctx)
        tools = await router.list_tools(call_ctx)
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(rctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        call_ctx = _extract_call_context(rctx)
        return await router.call_tool(
            params.name,
            params.arguments or {},
            call_ctx,
            server_session=rctx.session,
            request_id=rctx.request_id,
        )

    async def on_initialized(rctx: Any, params: Any) -> None:
        conn: Connection = rctx.session._connection
        sessions.register(conn)
        conn.exit_stack.callback(sessions.unregister, conn)

    mcp_server = ZosMCPServer(
        "zos",
        version="0.1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    mcp_server.add_notification_handler(
        "notifications/initialized",
        types.NotificationParams,
        on_initialized,
    )

    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        json_response=False,
        stateless=False,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        session_idle_timeout=1800,
    )

    actuator_manager = ActuatorManager(pool, registry, sessions)
    google_hub = GoogleServiceHub(pool, cfg, sessions)

    runtime = ZosRuntime(
        settings=cfg,
        pool=pool,
        registry=registry,
        sessions=sessions,
        router=router,
        mcp_server=mcp_server,
        session_manager=session_manager,
        actuator_manager=actuator_manager,
        google_hub=google_hub,
    )

    @contextlib.asynccontextmanager
    async def lifespan(starlette_app: Starlette) -> AsyncIterator[None]:
        await pool.open()
        try:
            if run_migrations_on_startup:
                await run_migrations(pool, cfg.migrations_dir)

            # Discover hosted memory provider UUID if initialized
            mem_provider_id = None
            async with pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT id FROM providers WHERE slug = 'memory' AND kind = 'hosted' LIMIT 1"
                )
                row = await cur.fetchone()
                if row is not None:
                    mem_provider_id = row["id"]

            registry.register(MemoryProvider(pool, provider_id=mem_provider_id))
            await actuator_manager.load_registered_actuators()
            await google_hub.load_existing_connection()
            registry.register(GoogleCalendarProvider(google_hub))
            registry.register(GmailProvider(google_hub))
            for provider in extra_providers or []:
                registry.register(provider)

            async with session_manager.run():
                yield
        finally:
            await pool.close()

    async def healthz(request: Request) -> JSONResponse:
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT 1 AS ok")
            await cur.fetchone()
        return JSONResponse(
            {
                "ok": True,
                "active_sessions": sessions.count,
                "providers": [
                    {"namespace": p.namespace, "kind": p.kind, "online": p.is_online()}
                    for p in registry.all_providers()
                ],
            }
        )

    mcp_asgi = BearerAuthASGIMiddleware(
        StreamableHTTPASGIApp(session_manager),
        pool,
        cfg.zos_base_url,
    )

    routes: list[BaseRoute] = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/mcp", endpoint=mcp_asgi, methods=["GET", "POST", "DELETE"]),
        *create_oauth_routes(pool, cfg),
        *actuator_manager.routes(),
        *google_hub.routes(),
    ]
    if extra_routes_factory is not None:
        routes.extend(extra_routes_factory(runtime))

    starlette_app = Starlette(routes=routes, lifespan=lifespan)
    starlette_app.state.zos = runtime
    return starlette_app


def main() -> None:
    logging.basicConfig(level=os.environ.get("ZOS_LOG_LEVEL", "INFO"))
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(
        "zos.app:create_app",
        factory=True,
        host=host,
        port=port,
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    main()
