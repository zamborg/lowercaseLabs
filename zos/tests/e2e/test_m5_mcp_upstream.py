"""M5.4 E2E tests for generic McpUpstreamProvider and gcal Calendar MCP proxy."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import anyio
import httpx
import httpx2
import mcp_types as types
from argon2 import PasswordHasher
from mcp.client.session import ClientRequestContext, ClientSession, IncomingMessage
from mcp.client.streamable_http import streamable_http_client
from mcp.server.connection import Connection
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.config import Settings
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.providers.google import GOOGLE_SCOPES
from zos.providers.mcp_upstream import McpUpstreamProvider
from zos.vault import generate_secret_key


class FakeUpstreamServer(Server[Any]):
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


class FakeUpstreamControl:
    """Mutable controller for an in-process Streamable HTTP fake upstream MCP server."""

    def __init__(self, *, valid_tokens: set[str]) -> None:
        self.valid_tokens = set(valid_tokens)
        self.bonus_tool_enabled = False
        self.connections: set[Connection] = set()
        self.received_calls: list[tuple[str, dict[str, Any]]] = []
        self.seen_auth_headers: list[str] = []

    async def toggle_bonus_tool(self, enabled: bool) -> None:
        self.bonus_tool_enabled = enabled
        for conn in list(self.connections):
            await conn.send_tool_list_changed()


def _build_fake_upstream_app(ctrl: FakeUpstreamControl) -> Starlette:
    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        tools = [
            types.Tool(
                name="list_events",
                description="List upstream calendar events",
                input_schema={"type": "object", "properties": {"time_min": {"type": "string"}}},
                annotations=types.ToolAnnotations(read_only_hint=True),
            ),
            types.Tool(
                name="get_event",
                description="Get single event",
                input_schema={"type": "object", "properties": {"event_id": {"type": "string"}}},
            ),
            types.Tool(
                name="list_calendars",
                description="List calendars",
                input_schema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="suggest_time",
                description="Suggest a meeting time (may elicit slot choice from user)",
                input_schema={"type": "object", "properties": {"elicit": {"type": "boolean"}}},
            ),
            types.Tool(
                name="search_events",
                description="Search events",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
            types.Tool(
                name="create_event",
                description="Create event",
                input_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
                annotations=types.ToolAnnotations(destructive_hint=False),
            ),
            types.Tool(
                name="update_event",
                description="Update event",
                input_schema={"type": "object", "properties": {"event_id": {"type": "string"}}},
            ),
            types.Tool(
                name="delete_event",
                description="Delete event",
                input_schema={"type": "object", "properties": {"event_id": {"type": "string"}}},
                annotations=types.ToolAnnotations(destructive_hint=True),
            ),
            types.Tool(
                name="respond_to_event",
                description="RSVP to event",
                input_schema={"type": "object", "properties": {"status": {"type": "string"}}},
            ),
            # Unknown preview tool that claims readOnlyHint=True upstream — ZOS MUST still treat it as external!
            types.Tool(
                name="mystery_preview_tool",
                description="New upstream preview tool not in ZOS tier seeds",
                input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
                annotations=types.ToolAnnotations(read_only_hint=True, open_world_hint=False),
            ),
        ]
        if ctrl.bonus_tool_enabled:
            tools.append(
                types.Tool(
                    name="bonus_tool",
                    description="Dynamically added upstream tool",
                    input_schema={"type": "object", "properties": {}},
                )
            )
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(rctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = dict(params.arguments or {})
        ctrl.received_calls.append((params.name, args))

        if params.name == "list_events":
            if args.get("trigger_error"):
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text="upstream_bad_range")],
                    structured_content={"error_detail": "invalid_range"},
                    is_error=True,
                )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="2 events")],
                structured_content={"count": 2, "echo_args": args},
                is_error=False,
            )

        if params.name == "suggest_time" and args.get("elicit"):
            elicit_res = await rctx.session.elicit_form(
                message="Pick a slot for the meeting?",
                requested_schema={
                    "type": "object",
                    "properties": {"slot": {"type": "string"}},
                    "required": ["slot"],
                },
                related_request_id=rctx.request_id,
            )
            if elicit_res.action == "accept" and isinstance(elicit_res.content, dict):
                chosen = elicit_res.content.get("slot", "none")
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"chosen:{chosen}")],
                    structured_content={"action": "accept", "slot": chosen},
                )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"elicit_{elicit_res.action}")],
                structured_content={"action": elicit_res.action},
                is_error=False,
            )

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"ok:{params.name}")],
            structured_content={"tool": params.name, "args": args},
        )

    async def on_initialized(rctx: Any, params: Any) -> None:
        conn: Connection = rctx.session._connection
        ctrl.connections.add(conn)
        conn.exit_stack.callback(ctrl.connections.discard, conn)

    server = FakeUpstreamServer(
        "fake-google-calendar-mcp",
        version="0.1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    server.add_notification_handler("notifications/initialized", types.NotificationParams, on_initialized)

    mgr = StreamableHTTPSessionManager(
        app=server,
        json_response=False,
        stateless=False,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        session_idle_timeout=300,
    )

    class SimpleBearerGate:
        def __init__(self, inner: ASGIApp) -> None:
            self.inner = inner

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http":
                auth_val = ""
                for k, v in scope.get("headers", []):
                    if k.lower() == b"authorization":
                        auth_val = v.decode("latin-1")
                ctrl.seen_auth_headers.append(auth_val)
                token = auth_val[7:].strip() if auth_val.lower().startswith("bearer ") else ""
                if token not in ctrl.valid_tokens:
                    body = json.dumps({"error": "unauthorized"}).encode()
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 401,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
            await self.inner(scope, receive, send)

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/mcp/v1", endpoint=SimpleBearerGate(StreamableHTTPASGIApp(mgr)), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lambda a: mgr.run(),
    )


class FakeTokenSource:
    """In-memory UpstreamTokenSource for testing 401 refresh and persistent 401."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = list(tokens)
        self._index = 0
        self.refresh_calls = 0
        self.has_cred = True

    def has_credential(self) -> bool:
        return self.has_cred

    @property
    def reconnect_hint(self) -> str:
        return "Reconnect Google at /connect/google."

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        if force_refresh:
            self.refresh_calls += 1
            if self._index + 1 < len(self._tokens):
                self._index += 1
        return self._tokens[self._index]


async def test_m5_e2e_mcp_upstream_proxy_full_behavior(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    client_a = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="agent-a")
    client_b = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="agent-b")

    # Start in-process fake upstream MCP server accepting "tok-fresh" (NOT "tok-stale")
    upstream_ctrl = FakeUpstreamControl(valid_tokens={"tok-fresh"})
    upstream_app = _build_fake_upstream_app(upstream_ctrl)
    upstream_port = free_port()
    upstream_url = f"http://127.0.0.1:{upstream_port}/mcp/v1"

    token_src = FakeTokenSource(["tok-stale", "tok-fresh"])

    zos_port = free_port()
    zos_base = f"http://127.0.0.1:{zos_port}"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=zos_base,
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash("owner-pw"),
        google_client_id="fake-cid",
        google_client_secret="fake-csecret",
    )

    async with run_asgi_server(upstream_app, port=upstream_port):
        # Create ZOS app with gcal configured to backend=mcp pointing at our fake upstream
        app = create_app(
            cfg,
            run_migrations_on_startup=False,
            gcal_backend="mcp",
            gcal_mcp_url=upstream_url,
        )

        # Swap gcal's token source to our FakeTokenSource so we test 401->refresh->retry cleanly
        async with run_asgi_server(app, port=zos_port):
            gcal_provider = app.state.zos.registry.get("gcal")
            assert isinstance(gcal_provider, McpUpstreamProvider)
            gcal_provider.token_source = token_src  # type: ignore[assignment]

            # 1. Lazy connect triggers 401 on tok-stale -> refreshes once -> succeeds on tok-fresh!
            connected = await gcal_provider.ensure_connected(broadcast_on_change=False)
            assert connected is True
            assert token_src.refresh_calls == 1
            assert gcal_provider.is_online() is True

            # Verify /healthz reports gcal via backend=mcp online=True
            async with httpx.AsyncClient() as web:
                hz = (await web.get(f"{zos_base}/healthz")).json()
                gcal_hz = next(p for p in hz["providers"] if p["namespace"] == "gcal")
                assert gcal_hz == {"namespace": "gcal", "kind": "upstream", "online": True, "backend": "mcp"}

                # Also verify /connect/google requests ALL unioned Calendar MCP + Gmail scopes
                r_conn = await web.post(
                    f"{zos_base}/connect/google",
                    data={"passphrase": "owner-pw"},
                    follow_redirects=False,
                )
                assert r_conn.status_code == 302
                qs = parse_qs(urlparse(r_conn.headers["location"]).query)
                assert set(qs["scope"][0].split()) == set(GOOGLE_SCOPES)
                assert qs["access_type"] == ["offline"]
                assert qs["prompt"] == ["consent"]

            list_changed_ev = anyio.Event()

            async def on_msg_a(msg: IncomingMessage) -> None:
                if isinstance(msg, types.ToolListChangedNotification):
                    list_changed_ev.set()

            async def elicit_a(ctx: ClientRequestContext, params: types.ElicitRequestParams):
                # Upstream elicitation asks for "slot"; ZOS policy elicitation asks for "approve"
                schema_props = getattr(params, "requested_schema", {}).get("properties", {})
                if "slot" in schema_props:
                    return types.ElicitResult(action="accept", content={"slot": "10:00"})
                return types.ElicitResult(action="accept", content={"approve": True})

            http_a = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_a['token']}"})
            http_b = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_b['token']}"})

            async with (
                http_a,
                http_b,
                streamable_http_client(f"{zos_base}/mcp", http_client=http_a) as (ra, wa),
                ClientSession(ra, wa, elicitation_callback=elicit_a, message_handler=on_msg_a) as sess_a,
                streamable_http_client(f"{zos_base}/mcp", http_client=http_b) as (rb, wb),
                ClientSession(rb, wb, elicitation_callback=None) as sess_b,
            ):
                await sess_a.initialize()
                await sess_b.initialize()
                await asyncio.sleep(0.2)

                # 2. Verify list + prefixing + ZOS policy tiers override upstream annotations!
                tools = {t.name: t for t in (await sess_a.list_tools()).tools}
                for read_tool in (
                    "gcal_list_events",
                    "gcal_get_event",
                    "gcal_list_calendars",
                    "gcal_suggest_time",
                    "gcal_search_events",
                ):
                    assert read_tool in tools
                    assert tools[read_tool].annotations.read_only_hint is True
                    assert tools[read_tool].annotations.destructive_hint is False
                    assert tools[read_tool].annotations.open_world_hint is False

                for ext_tool in (
                    "gcal_create_event",
                    "gcal_update_event",
                    "gcal_delete_event",
                    "gcal_respond_to_event",
                ):
                    assert ext_tool in tools
                    assert tools[ext_tool].annotations.open_world_hint is True
                    assert tools[ext_tool].annotations.read_only_hint is False
                # MCP spec default for destructiveHint is True on write/external unless upstream explicitly says False
                assert tools["gcal_create_event"].annotations.destructive_hint is False
                assert tools["gcal_update_event"].annotations.destructive_hint is True
                assert tools["gcal_delete_event"].annotations.destructive_hint is True
                assert tools["gcal_respond_to_event"].annotations.destructive_hint is True

                # Unknown upstream tool defaults to EXTERNAL + destructiveHint=True even though upstream claimed readOnlyHint=True!
                assert "gcal_mystery_preview_tool" in tools
                assert tools["gcal_mystery_preview_tool"].annotations.open_world_hint is True
                assert tools["gcal_mystery_preview_tool"].annotations.read_only_hint is False
                assert tools["gcal_mystery_preview_tool"].annotations.destructive_hint is True
                # Client B (no elicitation) is denied on unknown tool (proving unknown -> external -> require_approval)
                unk_b = await sess_b.call_tool("gcal_mystery_preview_tool", {"x": "1"})
                assert unk_b.is_error
                assert "zos policy set" in unk_b.content[0].text

                # 3. Call passthrough: prefix stripped, args verbatim, structuredContent & isError preserved!
                res_ok = await sess_a.call_tool("gcal_list_events", {"time_min": "2026-11-24T00:00:00Z"})
                assert not res_ok.is_error
                assert res_ok.content[0].text == "2 events"
                assert res_ok.structured_content == {"count": 2, "echo_args": {"time_min": "2026-11-24T00:00:00Z"}}
                assert ("list_events", {"time_min": "2026-11-24T00:00:00Z"}) in upstream_ctrl.received_calls

                res_err = await sess_a.call_tool("gcal_list_events", {"trigger_error": True})
                assert res_err.is_error is True
                assert res_err.content[0].text == "upstream_bad_range"
                assert res_err.structured_content == {"error_detail": "invalid_range"}

                # 4. Upstream notifications/tools/list_changed -> refreshes cache & broadcasts downstream!
                list_changed_ev = anyio.Event()
                await upstream_ctrl.toggle_bonus_tool(True)
                with anyio.fail_after(5):
                    await list_changed_ev.wait()
                tools_after = {t.name for t in (await sess_a.list_tools()).tools}
                assert "gcal_bonus_tool" in tools_after

                # 5. Upstream elicitation passthrough: Client A accepts, Client B auto-declines; both audited!
                elicit_ok = await sess_a.call_tool("gcal_suggest_time", {"elicit": True})
                assert not elicit_ok.is_error
                assert elicit_ok.content[0].text == "chosen:10:00"
                assert elicit_ok.structured_content == {"action": "accept", "slot": "10:00"}

                elicit_declined = await sess_b.call_tool("gcal_suggest_time", {"elicit": True})
                assert not elicit_declined.is_error
                assert elicit_declined.content[0].text == "elicit_decline"
                assert elicit_declined.structured_content == {"action": "decline"}

                async with db_pool.connection() as conn:
                    cur = await conn.execute(
                        "SELECT client_id, tool, decision, outcome FROM events "
                        "WHERE type = 'elicitation.passthrough' ORDER BY id ASC"
                    )
                    elicit_events = await cur.fetchall()
                assert len(elicit_events) == 2
                assert (elicit_events[0]["tool"], elicit_events[0]["decision"], elicit_events[0]["outcome"]) == (
                    "gcal_suggest_time",
                    "accept",
                    "ok",
                )
                assert (elicit_events[1]["tool"], elicit_events[1]["decision"], elicit_events[1]["outcome"]) == (
                    "gcal_suggest_time",
                    "decline",
                    "denied",
                )

                # 6. Persistent 401 after refresh -> offline + error naming /connect/google!
                upstream_ctrl.valid_tokens.clear()
                res_401 = await sess_a.call_tool("gcal_list_events", {})
                assert res_401.is_error is True
                assert "/connect/google" in res_401.content[0].text
                assert gcal_provider.is_online() is False


async def test_m5_upstream_down_goes_offline_with_clean_error(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    client_rec = await create_bearer_client(
        db_pool,
        principal_id=init["principal"]["id"],
        space_id=init["space"]["id"],
        name="agent-down",
    )

    dead_port = free_port()  # nothing listening on this port
    zos_port = free_port()
    zos_base = f"http://127.0.0.1:{zos_port}"
    cfg = Settings(database_url=get_test_database_url(), zos_base_url=zos_base)
    app = create_app(
        cfg,
        run_migrations_on_startup=False,
        gcal_backend="mcp",
        gcal_mcp_url=f"http://127.0.0.1:{dead_port}/mcp/v1",
    )

    async with run_asgi_server(app, port=zos_port):
        gcal_provider = app.state.zos.registry.get("gcal")
        assert isinstance(gcal_provider, McpUpstreamProvider)
        gcal_provider.token_source = FakeTokenSource(["tok-1"])  # type: ignore[assignment]
        gcal_provider._connect_timeout_s = 2.0

        ok = await gcal_provider.ensure_connected(broadcast_on_change=False)
        assert ok is False
        assert gcal_provider.is_online() is False

        http_c = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_rec['token']}"})
        async with (
            http_c,
            streamable_http_client(f"{zos_base}/mcp", http_client=http_c) as (r, w),
            ClientSession(r, w) as sess,
        ):
            await sess.initialize()
            tools = {t.name for t in (await sess.list_tools()).tools}
            assert not any(name.startswith("gcal_") for name in tools)

            res = await sess.call_tool("gcal_list_events", {})
            assert res.is_error is True
            assert "OFFLINE" in res.content[0].text
