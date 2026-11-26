"""M9a E2E & integration tests for owner-managed routed remote MCP servers:
- All 3 auth modes (`none`, `bearer`, `oauth`) against in-process fake remote MCP + fake OAuth 2.1 AS
- OAuth DCR, PKCE S256, RFC 8707 resource, RFC 9207 iss, `/connect/<slug>` callback, refresh, revoke -> `needs_reconnect`
- Slug validation, tool name sanitization & collision rejection
- `--default-tier` opt-down + per-tool tier editor on `/admin/policy`
- Hot add / set-token / disconnect / remove broadcasting `notifications/tools/list_changed`
- Offline upstream at startup never breaking startup
- CLI `zos upstream` subcommands (stdin-only token) & `/admin/upstreams` UI + audit events
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import secrets
from typing import Any
from urllib.parse import urlencode

import anyio
import httpx
import httpx2
import mcp_types as types
import psycopg
import pytest
from argon2 import PasswordHasher
from mcp.client.session import ClientSession, IncomingMessage
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from psycopg.rows import dict_row
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send
from typer.testing import CliRunner

from tests.conftest import MIGRATIONS_DIR, get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.cli import app as cli_app
from zos.config import Settings
from zos.db import open_pool, run_migrations
from zos.db.queries import create_bearer_client, create_remote_upstream, init_principal_and_space
from zos.providers.mcp_upstream import ToolNameCollisionError, _convert_upstream_tools
from zos.router.naming import sanitize_upstream_tool_name, validate_upstream_slug
from zos.vault import generate_secret_key


class _FakeRemoteMCPServer(Server[Any]):
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


class _FakeRemoteControl:
    """In-process remote MCP + OAuth 2.1 AS controller supporting none, bearer, and oauth modes."""

    def __init__(
        self,
        *,
        base_url: str,
        auth_mode: str,
        valid_bearer: str = "",
        tools: list[types.Tool] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_mode = auth_mode
        self.valid_tokens: set[str] = {valid_bearer} if valid_bearer else set()
        self.tools = tools or [
            types.Tool(
                name="issues.list",
                description="List issues (has dot requiring sanitization)",
                input_schema={"type": "object", "properties": {"repo": {"type": "string"}}},
                annotations=types.ToolAnnotations(read_only_hint=True),
            ),
            types.Tool(
                name="issues/create",
                description="Create an issue (has slash requiring sanitization)",
                input_schema={"type": "object", "properties": {"title": {"type": "string"}}},
                annotations=types.ToolAnnotations(read_only_hint=True),
            ),
        ]
        self.received_calls: list[tuple[str, dict[str, Any]]] = []
        # OAuth AS state
        self.registered_clients: dict[str, dict[str, Any]] = {}
        self.codes: dict[str, dict[str, Any]] = {}
        self.refresh_tokens: dict[str, dict[str, Any]] = {}
        self.dcr_count = 0
        self.refresh_count = 0
        self.revoke_all_refresh = False


def _build_fake_remote_app(ctrl: _FakeRemoteControl) -> Starlette:
    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list(ctrl.tools))

    async def on_call_tool(rctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = dict(params.arguments or {})
        ctrl.received_calls.append((params.name, args))
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"remote_ok:{params.name}")],
            structured_content={"raw_name": params.name, "args": args},
        )

    server = _FakeRemoteMCPServer(
        "fake-remote-mcp",
        version="1.0.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    mgr = StreamableHTTPSessionManager(
        app=server,
        json_response=False,
        stateless=False,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        session_idle_timeout=300,
    )

    class _AuthGate:
        def __init__(self, inner: ASGIApp) -> None:
            self.inner = inner

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http" and ctrl.auth_mode in ("bearer", "oauth"):
                auth_val = ""
                for k, v in scope.get("headers", []):
                    if k.lower() == b"authorization":
                        auth_val = v.decode("latin-1")
                token = auth_val[7:].strip() if auth_val.lower().startswith("bearer ") else ""
                if token not in ctrl.valid_tokens:
                    body = json.dumps({"error": "invalid_token"}).encode()
                    hdrs = [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ]
                    if ctrl.auth_mode == "oauth":
                        prm_url = f"{ctrl.base_url}/.well-known/oauth-protected-resource"
                        hdrs.append(
                            (
                                b"www-authenticate",
                                f'Bearer error="invalid_token", resource_metadata="{prm_url}"'.encode("latin-1"),
                            )
                        )
                    await send({"type": "http.response.start", "status": 401, "headers": hdrs})
                    await send({"type": "http.response.body", "body": body})
                    return
            await self.inner(scope, receive, send)

    async def prm(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "resource": f"{ctrl.base_url}/mcp",
                "authorization_servers": [ctrl.base_url],
                "scopes_supported": ["mcp"],
                "bearer_methods_supported": ["header"],
            }
        )

    async def asm(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "issuer": ctrl.base_url,
                "authorization_endpoint": f"{ctrl.base_url}/oauth/authorize",
                "token_endpoint": f"{ctrl.base_url}/oauth/token",
                "registration_endpoint": f"{ctrl.base_url}/oauth/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none"],
                "code_challenge_methods_supported": ["S256"],
                "authorization_response_iss_parameter_supported": True,
            }
        )

    async def register(request: Request) -> JSONResponse:
        body = await request.json()
        cid = f"remote_cid_{secrets.token_hex(6)}"
        ctrl.dcr_count += 1
        ctrl.registered_clients[cid] = body
        return JSONResponse(
            {
                "client_id": cid,
                "client_name": body.get("client_name", "ZOS"),
                "redirect_uris": body.get("redirect_uris", []),
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            status_code=201,
        )

    async def authorize(request: Request) -> RedirectResponse:
        qp = dict(request.query_params)
        assert qp.get("code_challenge_method") == "S256"
        assert qp.get("code_challenge")
        assert qp.get("resource") == f"{ctrl.base_url}/mcp"
        cid = qp["client_id"]
        assert cid in ctrl.registered_clients
        redirect_uri = qp["redirect_uri"]
        code = f"code_{secrets.token_urlsafe(12)}"
        ctrl.codes[code] = {
            "client_id": cid,
            "code_challenge": qp["code_challenge"],
            "redirect_uri": redirect_uri,
            "resource": qp.get("resource"),
        }
        out_qp = {"code": code, "iss": ctrl.base_url}
        if qp.get("state"):
            out_qp["state"] = qp["state"]
        return RedirectResponse(f"{redirect_uri}?{urlencode(out_qp)}", status_code=302)

    async def token(request: Request) -> JSONResponse:
        form = await request.form()
        data = {k: str(v) for k, v in form.items()}
        grant = data.get("grant_type")
        if grant == "authorization_code":
            rec = ctrl.codes.pop(data.get("code", ""), None)
            if rec is None or rec["client_id"] != data.get("client_id"):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            verifier = data.get("code_verifier", "")
            digest = hashlib.sha256(verifier.encode("ascii")).digest()
            computed = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
            if computed != rec["code_challenge"]:
                return JSONResponse({"error": "invalid_grant", "error_description": "PKCE mismatch"}, status_code=400)
            at = f"oauth_at_{secrets.token_hex(8)}"
            rt = f"oauth_rt_{secrets.token_hex(8)}"
            ctrl.valid_tokens.add(at)
            ctrl.refresh_tokens[rt] = {"client_id": rec["client_id"], "access_token": at}
            return JSONResponse(
                {
                    "access_token": at,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": rt,
                    "scope": "mcp",
                }
            )
        if grant == "refresh_token":
            rt = data.get("refresh_token", "")
            if ctrl.revoke_all_refresh or rt not in ctrl.refresh_tokens:
                return JSONResponse({"error": "invalid_grant", "error_description": "revoked"}, status_code=400)
            old = ctrl.refresh_tokens.pop(rt)
            ctrl.valid_tokens.discard(old["access_token"])
            new_at = f"oauth_at_ref_{secrets.token_hex(8)}"
            new_rt = f"oauth_rt_ref_{secrets.token_hex(8)}"
            ctrl.valid_tokens.add(new_at)
            ctrl.refresh_tokens[new_rt] = {"client_id": old["client_id"], "access_token": new_at}
            ctrl.refresh_count += 1
            return JSONResponse(
                {
                    "access_token": new_at,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": new_rt,
                    "scope": "mcp",
                }
            )
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/.well-known/oauth-protected-resource", prm, methods=["GET"]),
            Route("/.well-known/oauth-protected-resource/mcp", prm, methods=["GET"]),
            Route("/.well-known/oauth-authorization-server", asm, methods=["GET"]),
            Route("/oauth/register", register, methods=["POST"]),
            Route("/oauth/authorize", authorize, methods=["GET"]),
            Route("/oauth/token", token, methods=["POST"]),
            Route("/mcp", endpoint=_AuthGate(StreamableHTTPASGIApp(mgr)), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lambda a: mgr.run(),
    )


def _extract_csrf(html_body: str) -> str:
    m = re.search(r'<input type="hidden" name="csrf_token" value="([^"]+)">', html_body)
    assert m is not None
    return m.group(1)


def test_slug_validation_sanitization_and_collision_rejection() -> None:
    assert validate_upstream_slug("github") == "github"
    assert validate_upstream_slug("cf2") == "cf2"
    for bad in ("zos", "a", "Upper", "has_under", "has-dash", "9digit", "a" * 17, ""):
        with pytest.raises(ValueError):
            validate_upstream_slug(bad)

    assert sanitize_upstream_tool_name("issues.list") == "issues_list"
    assert sanitize_upstream_tool_name("issues/create") == "issues_create"
    assert sanitize_upstream_tool_name("get-user") == "get-user"

    # Non-colliding tools succeed
    ptools, raw_map = _convert_upstream_tools(
        [
            types.Tool(name="issues.list", description="d", input_schema={"type": "object"}),
            types.Tool(name="issues/create", description="d", input_schema={"type": "object"}),
        ],
        namespace="gh",
    )
    assert [t.name for t in ptools] == ["issues_list", "issues_create"]
    assert raw_map == {"issues_list": "issues.list", "issues_create": "issues/create"}

    # Colliding tools (`foo.bar` and `foo_bar` both -> `gh_foo_bar`) MUST be rejected!
    with pytest.raises(ToolNameCollisionError):
        _convert_upstream_tools(
            [
                types.Tool(name="foo.bar", description="d", input_schema={"type": "object"}),
                types.Tool(name="foo_bar", description="d", input_schema={"type": "object"}),
            ],
            namespace="gh",
        )


async def test_m9a_all_three_auth_modes_tiers_hot_reload_and_oauth_lifecycle(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]
    mcp_client = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="agent-m9a")

    owner_pw = "owner-m9a-pass"
    secret_key = generate_secret_key()

    # Spin up 3 in-process remote MCP servers: none, bearer, oauth
    p_none, p_bearer, p_oauth, p_zos = free_port(), free_port(), free_port(), free_port()
    none_ctrl = _FakeRemoteControl(base_url=f"http://127.0.0.1:{p_none}", auth_mode="none")
    pat_secret = "ghp_SUPER_SECRET_PAT_NEVER_LOGGED_999"
    bearer_ctrl = _FakeRemoteControl(
        base_url=f"http://127.0.0.1:{p_bearer}",
        auth_mode="bearer",
        valid_bearer=pat_secret,
    )
    oauth_ctrl = _FakeRemoteControl(base_url=f"http://127.0.0.1:{p_oauth}", auth_mode="oauth")

    zos_base = f"http://127.0.0.1:{p_zos}"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=zos_base,
        zos_secret_key=secret_key,
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
        approval_wait_seconds=0.05,
    )
    app = create_app(cfg, run_migrations_on_startup=False, upstream_connect_timeout_s=5.0)

    async with (
        run_asgi_server(_build_fake_remote_app(none_ctrl), port=p_none),
        run_asgi_server(_build_fake_remote_app(bearer_ctrl), port=p_bearer),
        run_asgi_server(_build_fake_remote_app(oauth_ctrl), port=p_oauth),
        run_asgi_server(app, port=p_zos),
    ):
        list_changed_ev = anyio.Event()

        async def on_msg(msg: IncomingMessage) -> None:
            if isinstance(msg, types.ToolListChangedNotification):
                list_changed_ev.set()

        http_mcp = httpx2.AsyncClient(headers={"Authorization": f"Bearer {mcp_client['token']}"})
        async with (
            http_mcp,
            streamable_http_client(f"{zos_base}/mcp", http_client=http_mcp) as (r, w),
            ClientSession(r, w, message_handler=on_msg) as sess,
            httpx.AsyncClient(base_url=zos_base, follow_redirects=False) as web,
        ):
            await sess.initialize()
            await asyncio.sleep(0.15)

            # Log into /admin
            assert (await web.post("/admin/login", data={"passphrase": owner_pw})).status_code == 303
            r_up = await web.get("/admin/upstreams")
            assert r_up.status_code == 200
            csrf = _extract_csrf(r_up.text)

            # CSRF enforced on /admin/upstreams/add
            assert (
                await web.post(
                    "/admin/upstreams/add",
                    data={"slug": "open", "url": f"http://127.0.0.1:{p_none}/mcp", "auth_mode": "none"},
                )
            ).status_code == 403

            # Reject reserved slug 'zos' and built-in slug 'gcal' cleanly
            for bad_slug in ("zos", "gcal", "Bad_Slug"):
                r_bad = await web.post(
                    "/admin/upstreams/add",
                    data={
                        "csrf_token": csrf,
                        "slug": bad_slug,
                        "url": f"http://127.0.0.1:{p_none}/mcp",
                        "auth_mode": "none",
                        "default_tier": "external",
                    },
                )
                assert r_bad.status_code == 200
                assert "notice-err" in r_bad.text

            # ── 1. AUTH MODE: NONE (with --default-tier read, plus per-tool tier override to external) ──
            list_changed_ev = anyio.Event()
            r_add_none = await web.post(
                "/admin/upstreams/add",
                data={
                    "csrf_token": csrf,
                    "slug": "docs",
                    "url": f"http://127.0.0.1:{p_none}/mcp",
                    "auth_mode": "none",
                    "default_tier": "read",
                },
            )
            assert r_add_none.status_code == 200
            assert "Added MCP server &#x27;docs&#x27;" in r_add_none.text
            assert "online" in r_add_none.text
            with anyio.fail_after(5):
                await list_changed_ev.wait()

            tools = {t.name: t for t in (await sess.list_tools()).tools}
            assert "docs_issues_list" in tools
            assert "docs_issues_create" in tools
            assert tools["docs_issues_list"].annotations.read_only_hint is True

            # Call sanitized tool -> upstream receives raw 'issues.list'
            res_none = await sess.call_tool("docs_issues_list", {"repo": "zos"})
            assert not res_none.is_error
            assert res_none.content[0].text == "remote_ok:issues.list"
            assert ("issues.list", {"repo": "zos"}) in none_ctrl.received_calls

            # Override per-tool tier on /admin/policy/tier: docs_issues_create -> external!
            r_tier = await web.post(
                "/admin/policy/tier",
                data={"csrf_token": csrf, "pattern": "docs_issues_create", "tier": "external"},
            )
            assert r_tier.status_code == 200
            assert "Tier set: docs_issues_create -&gt; external" in r_tier.text

            tools_after_tier = {t.name: t for t in (await sess.list_tools()).tools}
            assert tools_after_tier["docs_issues_list"].annotations.read_only_hint is True
            assert tools_after_tier["docs_issues_create"].annotations.open_world_hint is True
            # Calling external tool without elicitation handler -> hits async approval gate with link
            res_ext = await sess.call_tool("docs_issues_create", {"title": "hello"})
            assert res_ext.is_error is True
            assert "/admin/approvals/" in res_ext.content[0].text

            # ── 2. AUTH MODE: BEARER (default tier = external, stdin/password token in vault) ──
            r_add_b = await web.post(
                "/admin/upstreams/add",
                data={
                    "csrf_token": csrf,
                    "slug": "gh",
                    "url": f"http://127.0.0.1:{p_bearer}/mcp",
                    "auth_mode": "bearer",
                    "default_tier": "external",
                },
            )
            assert r_add_b.status_code == 200
            assert "not_configured" in r_add_b.text

            list_changed_ev = anyio.Event()
            r_tok_b = await web.post(
                "/admin/upstreams/token",
                data={"csrf_token": csrf, "slug": "gh", "token": pat_secret},
            )
            assert r_tok_b.status_code == 200
            assert pat_secret not in r_tok_b.text
            with anyio.fail_after(5):
                await list_changed_ev.wait()

            tools_b = {t.name: t for t in (await sess.list_tools()).tools}
            assert "gh_issues_list" in tools_b
            # Even though upstream claimed readOnlyHint=True, default_tier=external -> open_world_hint=True!
            assert tools_b["gh_issues_list"].annotations.open_world_hint is True
            assert tools_b["gh_issues_list"].annotations.read_only_hint is False

            # Allow gh_issues_list via policy, then call it
            await web.post(
                "/admin/policy/set",
                data={"csrf_token": csrf, "pattern": "gh_issues_list", "decision": "allow"},
            )
            res_b = await sess.call_tool("gh_issues_list", {"repo": "lowercase"})
            assert not res_b.is_error
            assert res_b.content[0].text == "remote_ok:issues.list"

            # Hot disconnect bearer -> tools vanish + list_changed
            list_changed_ev = anyio.Event()
            r_disc = await web.post("/admin/upstreams/disconnect", data={"csrf_token": csrf, "slug": "gh"})
            assert r_disc.status_code == 200
            with anyio.fail_after(5):
                await list_changed_ev.wait()
            assert "gh_issues_list" not in {t.name for t in (await sess.list_tools()).tools}

            # Hot remove docs -> tools vanish + list_changed
            list_changed_ev = anyio.Event()
            r_rem = await web.post("/admin/upstreams/remove", data={"csrf_token": csrf, "slug": "docs"})
            assert r_rem.status_code == 200
            with anyio.fail_after(5):
                await list_changed_ev.wait()
            assert "docs_issues_list" not in {t.name for t in (await sess.list_tools()).tools}

            # ── 3. AUTH MODE: OAUTH (OAuthClientProvider PRM -> DCR -> PKCE -> /connect/<slug> -> refresh -> revoke) ──
            r_add_o = await web.post(
                "/admin/upstreams/add",
                data={
                    "csrf_token": csrf,
                    "slug": "linear",
                    "url": f"http://127.0.0.1:{p_oauth}/mcp",
                    "auth_mode": "oauth",
                    "default_tier": "read",
                },
            )
            assert r_add_o.status_code == 200
            assert "/connect/linear" in r_add_o.text

            # GET /connect/linear renders mobile-friendly passphrase page
            r_cget = await web.get("/connect/linear")
            assert r_cget.status_code == 200
            assert 'name="viewport"' in r_cget.text

            # Wrong passphrase rejected with 401
            r_cbad = await web.post("/connect/linear", data={"passphrase": "wrong"})
            assert r_cbad.status_code == 401

            # Valid passphrase -> SDK runs PRM + DCR + PKCE -> 302 to fake AS /oauth/authorize
            r_cpost = await web.post("/connect/linear", data={"passphrase": owner_pw})
            assert r_cpost.status_code == 302, r_cpost.text
            auth_redirect = r_cpost.headers["location"]
            assert auth_redirect.startswith(f"http://127.0.0.1:{p_oauth}/oauth/authorize?")
            assert oauth_ctrl.dcr_count == 1

            # Phone browser follows redirect to fake AS -> redirected back to /connect/linear/callback
            async with httpx.AsyncClient(follow_redirects=False) as phone:
                r_as = await phone.get(auth_redirect)
                assert r_as.status_code == 302
                cb_url = r_as.headers["location"]
                assert cb_url.startswith(f"{zos_base}/connect/linear/callback?")

                list_changed_ev = anyio.Event()
                r_cb = await phone.get(cb_url)
                assert r_cb.status_code == 200, r_cb.text
                assert "Connected &#x27;linear&#x27;" in r_cb.text
                with anyio.fail_after(5):
                    await list_changed_ev.wait()

            # Tool call succeeds via OAuth access token
            res_o1 = await sess.call_tool("linear_issues_list", {"repo": "eng"})
            assert not res_o1.is_error
            assert res_o1.content[0].text == "remote_ok:issues.list"

            # Refresh test: invalidate current access token on fake remote -> 401 triggers refresh -> retry succeeds!
            oauth_ctrl.valid_tokens.clear()
            res_o2 = await sess.call_tool("linear_issues_list", {"repo": "eng2"})
            assert not res_o2.is_error
            assert res_o2.content[0].text == "remote_ok:issues.list"
            assert oauth_ctrl.refresh_count == 1

            # Revoke test: revoke refresh tokens + clear access tokens -> needs_reconnect + clear reconnect URL!
            oauth_ctrl.revoke_all_refresh = True
            oauth_ctrl.valid_tokens.clear()
            res_rev = await sess.call_tool("linear_issues_list", {"repo": "eng3"})
            assert res_rev.is_error is True
            assert "/connect/linear" in res_rev.content[0].text

            r_up_after = await web.get("/admin/upstreams")
            assert "needs_reconnect" in r_up_after.text
            assert "Reconnect" in r_up_after.text

            hz = (await web.get("/healthz")).json()
            lin_hz = next(p for p in hz["providers"] if p["namespace"] == "linear")
            assert lin_hz["status"] == "needs_reconnect"
            assert lin_hz["online"] is False

    # Verify zero secrets leaked in providers.config or events
    async with db_pool.connection() as conn:
        cfg_rows = await (await conn.execute("SELECT config::text AS c FROM providers")).fetchall()
        for r in cfg_rows:
            assert pat_secret not in r["c"]
        ev_rows = await (await conn.execute("SELECT type, payload::text AS p FROM events")).fetchall()
        ev_types = {r["type"] for r in ev_rows}
        for r in ev_rows:
            assert pat_secret not in r["p"]
        for expected in (
            "upstream.added",
            "upstream.token_set",
            "upstream.disconnected",
            "upstream.removed",
            "credential.connected",
            "credential.refreshed",
            "provider.needs_reconnect",
            "tier.changed",
        ):
            assert expected in ev_types


async def test_m9a_offline_upstream_does_not_break_startup(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    dead_port = free_port()
    await create_remote_upstream(
        db_pool,
        principal_id=init["principal"]["id"],
        space_id=init["space"]["id"],
        slug="downmcp",
        url=f"http://127.0.0.1:{dead_port}/mcp",
        auth_mode="none",
        default_tier="external",
    )

    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8787",
        zos_secret_key=generate_secret_key(),
    )
    app = create_app(cfg, run_migrations_on_startup=False, upstream_connect_timeout_s=1.5)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787") as client,
    ):
        hz = (await client.get("/healthz")).json()
        assert hz["ok"] is True
        by_ns = {p["namespace"]: p for p in hz["providers"]}
        assert by_ns["downmcp"]["online"] is False
        assert by_ns["downmcp"]["status"] == "offline"
        assert by_ns["downmcp"]["tool_count"] == 0


def test_m9a_cli_upstream_commands_and_stdin_token(monkeypatch) -> None:
    url = get_test_database_url()

    async def _reset() -> None:
        async with open_pool(url) as pool:
            await run_migrations(pool, MIGRATIONS_DIR)
            async with pool.connection() as conn:
                await conn.execute("TRUNCATE principals RESTART IDENTITY CASCADE")
                await conn.commit()

    asyncio.run(_reset())
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("ZOS_SECRET_KEY", generate_secret_key())
    monkeypatch.setenv("ZOS_BASE_URL", "https://zos.example.com")

    runner = CliRunner()
    assert runner.invoke(cli_app, ["init", "--principal", "zubin"]).exit_code == 0

    # Add bearer upstream + set-token via stdin
    r_add = runner.invoke(
        cli_app,
        [
            "upstream",
            "add",
            "github",
            "--url",
            "https://api.githubcopilot.com/mcp/",
            "--auth",
            "bearer",
            "--default-tier",
            "read",
        ],
    )
    assert r_add.exit_code == 0, r_add.output
    assert "slug=github" in r_add.output

    secret_pat = "ghp_cli_stdin_only_token_777"
    r_tok = runner.invoke(cli_app, ["upstream", "set-token", "github"], input=secret_pat + "\n")
    assert r_tok.exit_code == 0, r_tok.output
    assert secret_pat not in r_tok.output

    # Add oauth upstream + connect prints /connect/<slug> URL
    r_add_o = runner.invoke(
        cli_app,
        ["upstream", "add", "notion", "--url", "https://mcp.notion.com/mcp", "--auth", "oauth"],
    )
    assert r_add_o.exit_code == 0, r_add_o.output

    r_conn = runner.invoke(cli_app, ["upstream", "connect", "notion"])
    assert r_conn.exit_code == 0, r_conn.output
    assert r_conn.output.strip() == "https://zos.example.com/connect/notion"

    # List shows both
    r_list = runner.invoke(cli_app, ["upstream", "list"])
    assert r_list.exit_code == 0, r_list.output
    assert "github" in r_list.output and "notion" in r_list.output

    # Reject invalid slug
    assert runner.invoke(
        cli_app, ["upstream", "add", "zos", "--url", "https://example.com/mcp", "--auth", "none"]
    ).exit_code == 2

    # Remove cleans up provider, credential, and <slug>_* default tier
    r_rem = runner.invoke(cli_app, ["upstream", "remove", "github"])
    assert r_rem.exit_code == 0, r_rem.output
    with psycopg.connect(url, row_factory=dict_row) as conn:
        assert conn.execute("SELECT id FROM providers WHERE slug = 'github'").fetchone() is None
        assert conn.execute("SELECT id FROM credentials WHERE account = 'github'").fetchone() is None
        assert conn.execute("SELECT tier FROM tool_tiers WHERE tool_pattern = 'github_*'").fetchone() is None
