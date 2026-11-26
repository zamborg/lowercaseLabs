"""Tests for standalone ZOS Google module (`gmail` & `gcal`, encrypted tokens, OAuth UI, API + MCP)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import socket
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import httpx
import httpx2
import mcp_types as types
import psycopg
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from zos_google.calendar import GoogleCalendarService
from zos_google.caller import (
    HTTP_CALLER_HEADER,
    HTTP_CALLER_SIGNATURE_HEADER,
    MCP_CALLER_META_KEY,
    sign_http_caller,
    sign_mcp_caller,
)
from zos_google.gmail import GmailService
from zos_google.mcp_server import create_mcp_app
from zos_google.oauth import GoogleClient
from zos_google.store import GoogleTokenStore, create_google_pool
from zos_google.ui import create_ui_app

DEFAULT_TEST_DB = "postgresql://zos:zos@127.0.0.1:55432/zos_test"
SECRET = "google-test-secret-0123456789"


def _db_url() -> str:
    return os.environ.get("ZOS_TEST_DATABASE_URL", DEFAULT_TEST_DB)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.asynccontextmanager
async def _temp_store(service: str) -> AsyncIterator[GoogleTokenStore]:
    schema = f"test_g_{service}_{uuid.uuid4().hex[:8]}"
    base = _db_url()
    async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")

    sep = "&" if "?" in base else "?"
    schema_url = f"{base}{sep}options=-csearch_path%3D{schema}"
    pool = create_google_pool(schema_url)
    await pool.open()
    store = GoogleTokenStore(pool, module_secret=SECRET, service=service)
    await store.init_schema()
    try:
        yield store
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_google_module_has_zero_zos_imports() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src" / "zos_google"
    for py_file in src_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "import zos\n" not in text
        assert "from zos." not in text
        assert "from zos " not in text


async def test_gmail_oauth_ui_encrypted_tokens_mcp_tools_refresh_and_backoff() -> None:
    state_box: dict[str, Any] = {
        "valid_access": "at-initial-secret",
        "refresh_count": 0,
        "429_remaining": 1,
        "sent_payloads": [],
        "revoke_refresh": False,
    }

    def fake_google(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://oauth2.googleapis.com/token"):
            body = parse_qs(request.content.decode("utf-8"))
            grant = body.get("grant_type", [""])[0]
            if grant == "authorization_code":
                return httpx.Response(
                    200,
                    json={
                        "access_token": state_box["valid_access"],
                        "refresh_token": "rt-super-secret",
                        "expires_in": 3600,
                        "scope": "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.send",
                    },
                )
            if grant == "refresh_token":
                if state_box["revoke_refresh"]:
                    return httpx.Response(400, json={"error": "invalid_grant"})
                state_box["refresh_count"] += 1
                state_box["valid_access"] = f"at-refreshed-{state_box['refresh_count']}"
                return httpx.Response(
                    200,
                    json={"access_token": state_box["valid_access"], "expires_in": 3600},
                )
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {state_box['valid_access']}":
            return httpx.Response(401, json={"error": {"message": "Invalid Credentials"}})

        if state_box["429_remaining"] > 0 and "/messages" in url:
            state_box["429_remaining"] -= 1
            return httpx.Response(429, json={"error": {"message": "Rate limited"}})

        if url.endswith("/profile"):
            return httpx.Response(200, json={"emailAddress": "owner@example.com"})
        if request.method == "GET" and "/messages?" in url:
            return httpx.Response(200, json={"messages": [{"id": "m1"}]})
        if request.method == "GET" and "/messages/m1" in url:
            if "format=full" in url:
                html_b64 = base64.urlsafe_b64encode(b"<p>Hello <b>world</b></p><script>bad()</script>").decode("ascii")
                return httpx.Response(
                    200,
                    json={
                        "id": "m1",
                        "threadId": "t1",
                        "payload": {
                            "mimeType": "text/html",
                            "headers": [
                                {"name": "From", "value": "alice@example.com"},
                                {"name": "To", "value": "owner@example.com"},
                                {"name": "Subject", "value": "Lunch"},
                                {"name": "Date", "value": "Mon, 1 Jan 2026"},
                                {"name": "Message-ID", "value": "<msg-1@example.com>"},
                            ],
                            "body": {"data": html_b64},
                        },
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "m1",
                    "threadId": "t1",
                    "snippet": "Hello world",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "alice@example.com"},
                            {"name": "To", "value": "owner@example.com"},
                            {"name": "Subject", "value": "Lunch"},
                            {"name": "Date", "value": "Mon, 1 Jan 2026"},
                            {"name": "Message-ID", "value": "<msg-1@example.com>"},
                        ]
                    },
                },
            )
        if request.method == "POST" and url.endswith("/messages/send"):
            payload = json.loads(request.content.decode("utf-8"))
            state_box["sent_payloads"].append(payload)
            return httpx.Response(200, json={"id": "m2", "threadId": payload.get("threadId", "t2")})
        return httpx.Response(404)

    transport = httpx.MockTransport(fake_google)

    async with _temp_store("gmail") as store:
        # Idempotent versioned migrations
        assert await store.init_schema() == []

        gclient = GoogleClient(
            store,
            service="gmail",
            client_id="gid-123",
            client_secret="gsec-top-secret",
            http_client_factory=lambda: httpx.AsyncClient(transport=transport),
            backoff_base_s=0.01,
        )
        gmail_svc = GmailService(gclient)
        ui_app = create_ui_app(gclient, service="gmail", module_secret=SECRET)
        mcp_app = create_mcp_app(gmail_svc, service="gmail", module_secret=SECRET)

        # 1. Owner UI OAuth flow + security assertions
        hdr, sig = sign_http_caller(SECRET, principal_id="p-1", client_id="owner", client_name="owner")
        ui_headers = {
            HTTP_CALLER_HEADER: hdr,
            HTTP_CALLER_SIGNATURE_HEADER: sig,
            "X-Forwarded-Prefix": "/admin/m/gmail",
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "zos.example.com",
        }
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=ui_app), base_url="http://gmail.local") as ui:
            assert (await ui.get("/")).status_code == 403
            r_idx = await ui.get("/", headers=ui_headers)
            assert r_idx.status_code == 200
            assert "Not connected" in r_idx.text
            assert "https://zos.example.com/admin/m/gmail/callback" in r_idx.text

            m_csrf = re.search(r'name="csrf_token"\s+value="([^"]+)"', r_idx.text)
            assert m_csrf is not None
            csrf = m_csrf.group(1)
            cookie_hdrs = {**ui_headers, "Cookie": f"google_csrf={csrf}"}

            # Bad CSRF refused
            assert (
                await ui.post("/connect", headers=cookie_hdrs, data={"csrf_token": "wrong"}, follow_redirects=False)
            ).status_code == 403

            # Connect redirects to Google with narrower Gmail scopes ONLY
            r_conn = await ui.post("/connect", headers=cookie_hdrs, data={"csrf_token": csrf}, follow_redirects=False)
            assert r_conn.status_code == 302
            parsed = urlparse(r_conn.headers["location"])
            qs = parse_qs(parsed.query)
            assert qs["redirect_uri"] == ["https://zos.example.com/admin/m/gmail/callback"]
            scopes = set(qs["scope"][0].split())
            assert "https://www.googleapis.com/auth/gmail.readonly" in scopes
            assert "https://www.googleapis.com/auth/gmail.send" in scopes
            assert not any("calendar" in s for s in scopes)
            oauth_state = qs["state"][0]

            # Callback exchanges code & stores encrypted tokens
            r_cb = await ui.get(
                f"/callback?code=auth-code-1&state={oauth_state}",
                headers=cookie_hdrs,
                follow_redirects=False,
            )
            assert r_cb.status_code == 303
            assert "notice=" in r_cb.headers["location"]

            # Replaying same state fails
            r_replay = await ui.get(
                f"/callback?code=auth-code-1&state={oauth_state}",
                headers=cookie_hdrs,
                follow_redirects=False,
            )
            assert "error=" in r_replay.headers["location"]

            # Status page shows Connected + account, NEVER leaks tokens or client secret
            r_after = await ui.get("/", headers=cookie_hdrs)
            assert r_after.status_code == 200
            assert "Connected" in r_after.text
            assert "owner@example.com" in r_after.text
            for secret_val in ("at-initial-secret", "rt-super-secret", "gsec-top-secret"):
                assert secret_val not in r_after.text

        # Verify ciphertext in Postgres is encrypted (no plaintext tokens on disk)
        async with store.pool.connection() as conn:
            row = await (await conn.execute("SELECT ciphertext FROM google_credentials")).fetchone()
            assert row is not None
            raw_ct = bytes(row["ciphertext"])
            assert b"rt-super-secret" not in raw_ct
            assert b"at-initial-secret" not in raw_ct

        # 2. MCP tools (search + 429 retry, read + HTML strip, send + reply threading, 401 refresh, revoke)
        port = _free_port()
        srv = uvicorn.Server(uvicorn.Config(mcp_app, host="127.0.0.1", port=port, log_level="warning"))
        task = asyncio.create_task(srv.serve())
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient() as hc:
                for _ in range(50):
                    try:
                        if (await hc.get(f"{base}/healthz")).status_code == 200:
                            break
                    except Exception:
                        await asyncio.sleep(0.05)

            def meta_for(tool: str, args: dict[str, Any]) -> types.RequestParamsMeta:
                env = sign_mcp_caller(
                    SECRET,
                    principal_id="p-1",
                    client_id="c-1",
                    client_name="claude",
                    tool=tool,
                    args=args,
                )
                return cast(types.RequestParamsMeta, {MCP_CALLER_META_KEY: env})

            async with (
                streamable_http_client(f"{base}/mcp") as (r, w),
                ClientSession(r, w) as sess,
            ):
                await sess.initialize()
                tools = {t.name for t in (await sess.list_tools()).tools}
                assert tools == {"search", "read", "send"}

                # Unsigned call refused
                r_un = await sess.call_tool("search", {"query": "lunch"})
                assert r_un.is_error and "CALLER_REQUIRED" in r_un.content[0].text

                # Search succeeds after 429 retry
                r_s = await sess.call_tool("search", {"query": "lunch", "limit": 5}, meta=meta_for("search", {"query": "lunch", "limit": 5}))
                assert not r_s.is_error
                s_data = json.loads(r_s.content[0].text)
                assert s_data["count"] == 1 and s_data["messages"][0]["subject"] == "Lunch"

                # Read strips HTML + script tags
                r_rd = await sess.call_tool("read", {"message_id": "m1"}, meta=meta_for("read", {"message_id": "m1"}))
                assert not r_rd.is_error
                rd_data = json.loads(r_rd.content[0].text)
                assert "Hello world" in rd_data["body"] and "bad()" not in rd_data["body"]

                # Send with reply_to_message_id threads properly
                send_args = {"to": "alice@example.com", "subject": "Re: Lunch", "body": "Sounds good", "reply_to_message_id": "m1"}
                r_snd = await sess.call_tool("send", send_args, meta=meta_for("send", send_args))
                assert not r_snd.is_error
                assert json.loads(r_snd.content[0].text)["thread_id"] == "t1"
                raw_sent = base64.urlsafe_b64decode(state_box["sent_payloads"][0]["raw"]).decode("utf-8")
                assert "In-Reply-To: <msg-1@example.com>" in raw_sent

                # Force 401 on current access token -> auto-refresh -> succeeds!
                state_box["valid_access"] = "invalidated-force-refresh"
                r_s2 = await sess.call_tool("search", {"query": "lunch"}, meta=meta_for("search", {"query": "lunch"}))
                assert not r_s2.is_error
                assert state_box["refresh_count"] == 1

                # Revoked refresh token -> deletes stored credential and returns reconnect hint
                state_box["valid_access"] = "expired-again"
                state_box["revoke_refresh"] = True
                r_rev = await sess.call_tool("search", {"query": "lunch"}, meta=meta_for("search", {"query": "lunch"}))
                assert r_rev.is_error
                assert "/admin/m/gmail" in r_rev.content[0].text
                assert (await store.get_connection_status())["connected"] is False
        finally:
            srv.should_exit = True
            await task


async def test_gcal_api_and_mcp_backends_and_ui_switch() -> None:
    # Fake remote Google Calendar MCP server for testing backend='mcp'
    fake_mcp_calls: list[tuple[str, dict[str, Any], str]] = []

    async def on_list(rctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="list_events",
                    description="Remote MCP list_events",
                    input_schema={"type": "object", "properties": {}},
                )
            ]
        )

    async def on_call(rctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        req = getattr(rctx, "request", None)
        auth = req.headers.get("authorization", "") if req is not None else ""
        fake_mcp_calls.append((params.name, dict(params.arguments or {}), auth))
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps({"backend": "remote-mcp"}))]
        )

    remote_server = Server("fake-gcal-mcp", version="0.1.0", on_list_tools=on_list, on_call_tool=on_call)
    remote_sm = StreamableHTTPSessionManager(
        app=remote_server,
        json_response=True,
        stateless=True,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @contextlib.asynccontextmanager
    async def remote_lifespan(app: Starlette) -> AsyncIterator[None]:
        async with remote_sm.run():
            yield

    remote_app = Starlette(
        routes=[
            Route("/healthz", lambda _r: JSONResponse({"ok": True}), methods=["GET"]),
            Route("/mcp/v1", endpoint=StreamableHTTPASGIApp(remote_sm), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=remote_lifespan,
    )
    remote_port = _free_port()
    remote_srv = uvicorn.Server(uvicorn.Config(remote_app, host="127.0.0.1", port=remote_port, log_level="warning"))
    remote_task = asyncio.create_task(remote_srv.serve())

    def fake_google_cal(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/calendarList"):
            return httpx.Response(
                200,
                json={"items": [{"id": "primary", "summary": "Main", "primary": True, "timeZone": "UTC", "accessRole": "owner"}]},
            )
        if url.endswith("/freeBusy"):
            return httpx.Response(200, json={"calendars": {"primary": {"busy": [], "errors": []}}})
        if "/events" in url and request.method == "GET":
            return httpx.Response(
                200,
                json={"items": [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-11-26T10:00:00Z"}, "end": {"dateTime": "2026-11-26T10:30:00Z"}}]},
            )
        return httpx.Response(200, json={})

    try:
        async with httpx.AsyncClient() as hc:
            for _ in range(50):
                try:
                    if (await hc.get(f"http://127.0.0.1:{remote_port}/healthz")).status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.05)

        async with _temp_store("gcal") as store:
            await store.save_tokens(
                account="cal@example.com",
                scopes=list(GoogleClient(store, service="gcal", client_id="id", client_secret="sec").scopes),
                access_token="gcal-access-tok",
                refresh_token="gcal-refresh-tok",
                expires_in=3600,
            )
            gclient = GoogleClient(
                store,
                service="gcal",
                client_id="id",
                client_secret="sec",
                http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake_google_cal)),
            )
            cal_svc = GoogleCalendarService(
                gclient,
                default_backend="api",
                mcp_url=f"http://127.0.0.1:{remote_port}/mcp/v1",
                mcp_http_client_factory=lambda tok: httpx2.AsyncClient(headers={"Authorization": f"Bearer {tok}"}),
            )

            # 1. API backend exposes all 9 tools, enforces explicit timezone, calls REST v3
            api_tools = {t.name for t in await cal_svc.list_tools()}
            assert api_tools == {
                "list_events",
                "get_event",
                "list_calendars",
                "search_events",
                "freebusy",
                "create_event",
                "update_event",
                "delete_event",
                "respond_to_event",
            }
            r_naive = await cal_svc.call_tool(
                "list_events",
                {"time_min": "2026-11-26T00:00:00", "time_max": "2026-11-27T00:00:00Z"},
            )
            assert r_naive.is_error and "timezone" in r_naive.content[0].text.lower()

            r_ev = await cal_svc.call_tool(
                "list_events",
                {"time_min": "2026-11-26T00:00:00Z", "time_max": "2026-11-27T00:00:00Z"},
            )
            assert not r_ev.is_error
            assert json.loads(r_ev.content[0].text)["count"] == 1

            r_fb = await cal_svc.call_tool(
                "freebusy",
                {"time_min": "2026-11-26T00:00:00Z", "time_max": "2026-11-27T00:00:00Z"},
            )
            assert not r_fb.is_error and "primary" in json.loads(r_fb.content[0].text)["calendars"]

            # 2. Owner UI switches backend to 'mcp' (CSRF enforced)
            ui_app = create_ui_app(gclient, service="gcal", module_secret=SECRET, calendar_service=cal_svc)
            hdr, sig = sign_http_caller(SECRET, principal_id="p-1", client_id="owner", client_name="owner")
            ui_hdrs = {
                HTTP_CALLER_HEADER: hdr,
                HTTP_CALLER_SIGNATURE_HEADER: sig,
                "X-Forwarded-Prefix": "/admin/m/gcal",
                "X-Forwarded-Proto": "https",
                "X-Forwarded-Host": "zos.example.com",
            }
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=ui_app), base_url="http://gcal.local") as ui:
                r_idx = await ui.get("/", headers=ui_hdrs)
                assert r_idx.status_code == 200
                assert "Calendar Backend" in r_idx.text
                assert "gcal-access-tok" not in r_idx.text
                assert "gcal-refresh-tok" not in r_idx.text
                csrf = re.search(r'name="csrf_token"\s+value="([^"]+)"', r_idx.text).group(1)

                r_sw = await ui.post(
                    "/backend",
                    headers={**ui_hdrs, "Cookie": f"google_csrf={csrf}"},
                    data={"csrf_token": csrf, "backend": "mcp"},
                    follow_redirects=False,
                )
                assert r_sw.status_code == 303

            assert await cal_svc.get_backend() == "mcp"
            mcp_tools = {t.name for t in await cal_svc.list_tools()}
            assert mcp_tools == {"list_events"}

            r_mcp = await cal_svc.call_tool("list_events", {})
            assert not r_mcp.is_error
            assert json.loads(r_mcp.content[0].text) == {"backend": "remote-mcp"}
            assert fake_mcp_calls == [("list_events", {}, "Bearer gcal-access-tok")]
    finally:
        remote_srv.should_exit = True
        await remote_task
