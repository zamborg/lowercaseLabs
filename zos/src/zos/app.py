"""ZOS ASGI application mounting Streamable HTTP MCP at /mcp and /healthz."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

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
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from zos.audit import DatabaseAuditRecorder
from zos.auth import BearerAuthASGIMiddleware, CallContext, current_call_context
from zos.auth.oauth import create_oauth_routes
from zos.auth.rate_limit import GlobalPassphraseRateLimiter, GlobalSlidingWindowLimiter
from zos.config import Settings
from zos.db import create_pool, run_migrations
from zos.instructions import load_mcp_instructions
from zos.policy import DatabasePolicyEvaluator, request_tool_confirmation
from zos.providers.actuator import ActuatorManager
from zos.providers.base import ToolProvider
from zos.providers.google import (
    DEFAULT_GCAL_MCP_URL,
    GmailProvider,
    GoogleCalendarProvider,
    GoogleServiceHub,
)
from zos.providers.mcp_upstream import HttpClientFactory, McpUpstreamProvider
from zos.providers.memory import MemoryProvider
from zos.router import ProviderRegistry, SessionTracker, ToolRouter
from zos.web.admin import create_admin_routes

logger = logging.getLogger(__name__)


class PublicSchemeMiddleware:
    """Derive HTTP scope scheme from `ZOS_BASE_URL` for requests targeting the public host.

    Behind a TLS-terminating tunnel (e.g. cloudflared), upstream HTTP requests arrive on
    plain HTTP with the public `Host` header preserved. Rewriting `scope["scheme"]` to
    `"https"` when `ZOS_BASE_URL` is HTTPS and `Host` matches ensures Starlette-generated
    URLs (such as 307 trailing-slash redirects on `/admin/` or `/mcp/`) emit `https://`
    without trusting arbitrary `X-Forwarded-Proto` headers or hardcoding gateway IPs.
    """

    def __init__(self, app: ASGIApp, *, base_url: str) -> None:
        self.app = app
        parsed = urlparse(base_url.strip())
        self.public_scheme = parsed.scheme.lower()
        self.public_host = (parsed.hostname or "").lower()
        self.public_netloc = parsed.netloc.lower()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.public_scheme == "https" and self.public_host:
            req_host = ""
            for k, v in scope.get("headers", []):
                if k.lower() == b"host":
                    req_host = v.decode("latin-1").strip().lower()
                    break
            req_host_norm = req_host[:-4] if req_host.endswith(":443") else req_host
            if req_host_norm in (self.public_netloc, self.public_host):
                scope = dict(scope)
                scope["scheme"] = "https"
        await self.app(scope, receive, send)


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
    switch_provider_backend: Callable[[str, str], Awaitable[ToolProvider]] | None = None


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
    gcal_backend: str | None = None,
    gcal_mcp_url: str | None = None,
    gcal_http_client_factory: HttpClientFactory | None = None,
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
        instructions=load_mcp_instructions(cfg.instructions_file),
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

    passphrase_limiter = GlobalPassphraseRateLimiter()
    dcr_limiter = GlobalSlidingWindowLimiter(
        max_requests=cfg.dcr_max_per_hour,
        window_seconds=3600.0,
    )
    actuator_manager = ActuatorManager(pool, registry, sessions)
    google_hub = GoogleServiceHub(pool, cfg, sessions, passphrase_limiter=passphrase_limiter)

    active_gcal_mcp: McpUpstreamProvider | None = None
    gcal_switch_lock = asyncio.Lock()

    def _effective_gcal_mcp_url() -> str:
        gcal_cfg = google_hub.gcal_config
        return (
            (gcal_mcp_url or "").strip()
            or str(gcal_cfg.get("url") or "").strip()
            or cfg.gcal_mcp_url
            or DEFAULT_GCAL_MCP_URL
        )

    async def _on_google_connected() -> None:
        if active_gcal_mcp is not None:
            await active_gcal_mcp.ensure_connected(broadcast_on_change=False)

    async def _on_google_revoked() -> None:
        if active_gcal_mcp is not None:
            await active_gcal_mcp.disconnect(broadcast_on_change=False)

    google_hub.add_on_connect(_on_google_connected)
    google_hub.add_on_revoke(_on_google_revoked)

    async def _activate_gcal_backend(
        backend: str,
        *,
        broadcast: bool = True,
    ) -> ToolProvider:
        nonlocal active_gcal_mcp
        async with gcal_switch_lock:
            old_mcp = active_gcal_mcp
            active_gcal_mcp = None
            if old_mcp is not None:
                with contextlib.suppress(Exception):
                    await old_mcp.aclose()

            google_hub.set_gcal_backend_config(backend)
            provider: ToolProvider
            if backend == "api":
                provider = GoogleCalendarProvider(google_hub)
                registry.register(provider)
            else:
                new_mcp = McpUpstreamProvider(
                    namespace="gcal",
                    url=_effective_gcal_mcp_url(),
                    token_source=google_hub,
                    sessions=sessions,
                    provider_id_getter=lambda: google_hub.gcal_provider_id,
                    audit_elicitation=audit_recorder.record_elicitation,
                    http_client_factory=gcal_http_client_factory,
                )
                active_gcal_mcp = new_mcp
                registry.register(new_mcp)
                provider = new_mcp
                if google_hub.is_connected:
                    await new_mcp.ensure_connected(broadcast_on_change=False)

        if broadcast:
            await sessions.broadcast_tools_changed()
        return provider

    async def switch_provider_backend(slug: str, backend: str) -> ToolProvider:
        if slug != "gcal":
            raise ValueError(f"Hot backend switching is not supported for '{slug}'")
        return await _activate_gcal_backend(backend, broadcast=True)

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
        switch_provider_backend=switch_provider_backend,
    )

    @contextlib.asynccontextmanager
    async def lifespan(starlette_app: Starlette) -> AsyncIterator[None]:
        await pool.open()
        extra_closables: list[McpUpstreamProvider] = []
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

            gcal_cfg = google_hub.gcal_config
            effective_backend = (
                (gcal_backend or "").strip().lower()
                or str(gcal_cfg.get("backend") or "").strip().lower()
                or cfg.gcal_backend
                or "mcp"
            )
            await _activate_gcal_backend(effective_backend, broadcast=False)

            registry.register(GmailProvider(google_hub))
            for provider in extra_providers or []:
                registry.register(provider)
                if isinstance(provider, McpUpstreamProvider):
                    extra_closables.append(provider)

            async with session_manager.run():
                yield
        finally:
            if active_gcal_mcp is not None:
                with contextlib.suppress(Exception):
                    await active_gcal_mcp.aclose()
            for upstream in extra_closables:
                with contextlib.suppress(Exception):
                    await upstream.aclose()
            await pool.close()

    async def healthz(request: Request) -> JSONResponse:
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT 1 AS ok")
            await cur.fetchone()
        providers_health = []
        for p in registry.all_providers():
            entry: dict[str, Any] = {
                "namespace": p.namespace,
                "kind": p.kind,
                "online": p.is_online(),
            }
            if hasattr(p, "backend"):
                entry["backend"] = p.backend
            providers_health.append(entry)
        return JSONResponse(
            {
                "ok": True,
                "active_sessions": sessions.count,
                "providers": providers_health,
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
        *create_oauth_routes(
            pool,
            cfg,
            passphrase_limiter=passphrase_limiter,
            dcr_limiter=dcr_limiter,
        ),
        *actuator_manager.routes(),
        *google_hub.routes(),
        *create_admin_routes(
            pool,
            cfg,
            registry,
            sessions,
            passphrase_limiter=passphrase_limiter,
            switch_provider_backend=switch_provider_backend,
        ),
    ]
    if extra_routes_factory is not None:
        routes.extend(extra_routes_factory(runtime))

    starlette_app = Starlette(
        routes=routes,
        middleware=[Middleware(PublicSchemeMiddleware, base_url=cfg.zos_base_url)],
        lifespan=lifespan,
    )
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
