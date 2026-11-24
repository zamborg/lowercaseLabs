"""Tests for FOLLOWUP items:
1. Audited upstream provider backend switching (`AdminService`, `zos provider set-backend`,
   `/admin` Overview toggle, `provider.backend_changed` audit event, zero-restart hot re-register).
2. `gmail_search` schema and `max_results` / `limit` agreement.
"""

from __future__ import annotations

import asyncio
import json
import re
from uuid import uuid4

import anyio
import httpx
import httpx2
import mcp_types as types
import psycopg
import respx
from argon2 import PasswordHasher
from mcp.client.session import ClientSession, IncomingMessage
from mcp.client.streamable_http import streamable_http_client
from psycopg.rows import dict_row
from typer.testing import CliRunner

from tests.conftest import MIGRATIONS_DIR, get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.auth.context import CallContext
from zos.cli import app as cli_app
from zos.config import Settings
from zos.db import open_pool, run_migrations
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.providers.google import GmailProvider, GoogleCalendarProvider, GoogleServiceHub
from zos.providers.mcp_upstream import McpUpstreamProvider
from zos.router import SessionTracker
from zos.vault import generate_secret_key


def _extract_csrf(html_body: str) -> str:
    m = re.search(r'<input type="hidden" name="csrf_token" value="([^"]+)">', html_body)
    assert m is not None
    return m.group(1)


async def test_provider_set_backend_admin_hot_reload_and_audit(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]
    bearer = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="agent-hot")

    owner_pw = "hot-switch-pw"
    zos_port = free_port()
    zos_base = f"http://127.0.0.1:{zos_port}"
    dead_port = free_port()

    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=zos_base,
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
    )
    app = create_app(
        cfg,
        run_migrations_on_startup=False,
        gcal_mcp_url=f"http://127.0.0.1:{dead_port}/mcp/v1",
    )

    async with run_asgi_server(app, port=zos_port):
        # Attach a Google credential so gcal is active (starts on default backend=mcp)
        await app.state.zos.google_hub.attach_google_tokens(
            account="zubin@example.com",
            scopes=["https://www.googleapis.com/auth/calendar.events"],
            access_token="goog-at",
            refresh_token="goog-rt",
            expires_in=3600,
        )
        assert isinstance(app.state.zos.registry.get("gcal"), McpUpstreamProvider)

        list_changed_ev = anyio.Event()

        async def on_msg(msg: IncomingMessage) -> None:
            if isinstance(msg, types.ToolListChangedNotification):
                list_changed_ev.set()

        http_mcp = httpx2.AsyncClient(headers={"Authorization": f"Bearer {bearer['token']}"})
        async with (
            http_mcp,
            streamable_http_client(f"{zos_base}/mcp", http_client=http_mcp) as (r, w),
            ClientSession(r, w, message_handler=on_msg) as sess,
            httpx.AsyncClient(base_url=zos_base, follow_redirects=False) as web,
        ):
            await sess.initialize()
            await asyncio.sleep(0.15)

            # Log into /admin Overview
            r_login = await web.post("/admin/login", data={"passphrase": owner_pw})
            assert r_login.status_code == 303
            r_ov = await web.get("/admin")
            assert r_ov.status_code == 200
            assert "Switch to api" in r_ov.text
            csrf = _extract_csrf(r_ov.text)

            # Missing CSRF -> 403
            r_no_csrf = await web.post("/admin/providers/backend", data={"slug": "gcal", "backend": "api"})
            assert r_no_csrf.status_code == 403

            # Hot-switch gcal from mcp -> api with NO restart!
            list_changed_ev = anyio.Event()
            r_sw = await web.post(
                "/admin/providers/backend",
                data={"csrf_token": csrf, "slug": "gcal", "backend": "api"},
            )
            assert r_sw.status_code == 200
            assert "Switched provider &#x27;gcal&#x27; backend to &#x27;api&#x27;" in r_sw.text or "backend to" in r_sw.text
            assert "Switch to mcp" in r_sw.text

            # Connected MCP client received notifications/tools/list_changed and sees REST gcal tools immediately
            with anyio.fail_after(5):
                await list_changed_ev.wait()

            assert isinstance(app.state.zos.registry.get("gcal"), GoogleCalendarProvider)
            hz = (await web.get("/healthz")).json()
            gcal_hz = next(p for p in hz["providers"] if p["namespace"] == "gcal")
            assert gcal_hz == {"namespace": "gcal", "kind": "upstream", "online": True, "backend": "api"}

            tools = {t.name for t in (await sess.list_tools()).tools}
            assert {"gcal_list_events", "gcal_create_event", "gcal_update_event", "gcal_delete_event"} <= tools

            # Switch back api -> mcp (hot re-registers McpUpstreamProvider)
            list_changed_ev = anyio.Event()
            r_sw2 = await web.post(
                "/admin/providers/backend",
                data={"csrf_token": csrf, "slug": "gcal", "backend": "mcp"},
            )
            assert r_sw2.status_code == 200
            assert "Switch to api" in r_sw2.text
            with anyio.fail_after(5):
                await list_changed_ev.wait()
            assert isinstance(app.state.zos.registry.get("gcal"), McpUpstreamProvider)

            # Invalid slug rejected cleanly in UI
            r_bad = await web.post(
                "/admin/providers/backend",
                data={"csrf_token": csrf, "slug": "gmail", "backend": "mcp"},
            )
            assert r_bad.status_code == 200
            assert "does not support backend switching" in r_bad.text

    # Verify DB config preserved other keys (url) and audit events recorded both switches
    async with db_pool.connection() as conn:
        prov_row = await (
            await conn.execute("SELECT config FROM providers WHERE slug = 'gcal'")
        ).fetchone()
        assert prov_row is not None
        assert prov_row["config"]["backend"] == "mcp"
        assert "url" in prov_row["config"]

        events = await (
            await conn.execute(
                "SELECT type, outcome, payload FROM events WHERE type = 'provider.backend_changed' ORDER BY id ASC"
            )
        ).fetchall()
    assert len(events) == 2
    assert events[0]["outcome"] == "ok"
    assert events[0]["payload"]["slug"] == "gcal"
    assert events[0]["payload"]["from"] == "mcp"
    assert events[0]["payload"]["to"] == "api"
    assert events[0]["payload"]["hot_reloaded"] is True
    assert events[1]["payload"]["from"] == "api"
    assert events[1]["payload"]["to"] == "mcp"


def test_provider_set_backend_cli_parity(monkeypatch) -> None:
    db_url = get_test_database_url()

    async def _reset() -> None:
        async with open_pool(db_url) as pool:
            await run_migrations(pool, MIGRATIONS_DIR)
            async with pool.connection() as conn:
                await conn.execute("TRUNCATE principals RESTART IDENTITY CASCADE")
                await conn.commit()

    asyncio.run(_reset())
    monkeypatch.setenv("DATABASE_URL", db_url)
    runner = CliRunner()

    r_init = runner.invoke(cli_app, ["init", "--principal", "zubin"])
    assert r_init.exit_code == 0, r_init.output

    r_api = runner.invoke(cli_app, ["provider", "set-backend", "gcal", "api"])
    assert r_api.exit_code == 0, r_api.output
    assert "slug=gcal" in r_api.output and "to=api" in r_api.output

    r_mcp = runner.invoke(cli_app, ["provider", "set-backend", "gcal", "mcp"])
    assert r_mcp.exit_code == 0, r_mcp.output
    assert "from=api to=mcp" in r_mcp.output

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        evs = conn.execute(
            "SELECT type, outcome, payload FROM events WHERE type = 'provider.backend_changed' ORDER BY id ASC"
        ).fetchall()
    assert len(evs) == 2
    assert evs[0]["payload"]["to"] == "api"
    assert evs[1]["payload"]["from"] == "api"
    assert evs[1]["payload"]["to"] == "mcp"

    r_bad_slug = runner.invoke(cli_app, ["provider", "set-backend", "gmail", "mcp"])
    assert r_bad_slug.exit_code == 2

    r_bad_backend = runner.invoke(cli_app, ["provider", "set-backend", "gcal", "soap"])
    assert r_bad_backend.exit_code == 2


async def test_gmail_search_schema_and_max_results_agreement(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8787",
        zos_secret_key=generate_secret_key(),
    )
    hub = GoogleServiceHub(db_pool, cfg, SessionTracker())
    await hub.attach_google_tokens(
        account="zubin@example.com",
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        access_token="goog-at",
        refresh_token="goog-rt",
        expires_in=3600,
    )
    gmail = GmailProvider(hub)

    # 1. Schema agrees with accepted parameters and forbids unknown args
    search_tool = next(t for t in await gmail.list_tools() if t.name == "search")
    schema = search_tool.input_schema
    assert schema["additionalProperties"] is False
    assert "limit" in schema["properties"]
    assert "max_results" in schema["properties"]
    assert schema["properties"]["limit"]["maximum"] == 25
    assert schema["properties"]["max_results"]["maximum"] == 25

    ctx = CallContext(
        principal_id=init["principal"]["id"],
        principal_slug="zubin",
        space_id=init["space"]["id"],
        space_slug="main",
        client_id=uuid4(),
        client_name="test",
        client_kind="bearer",
        trust_ceiling="owner",
        token_id=uuid4(),
        scopes=("mcp",),
    )

    seen_max_results: list[str | None] = []

    with respx.mock(assert_all_called=False) as mock:
        def list_handler(request: httpx.Request) -> httpx.Response:
            seen_max_results.append(request.url.params.get("maxResults"))
            # Intentionally return 10 refs even when caller asks for 5, to verify both query param & slice
            return httpx.Response(
                200,
                json={"messages": [{"id": f"m{i}", "threadId": f"t{i}"} for i in range(10)]},
            )

        mock.get("https://gmail.googleapis.com/gmail/v1/users/me/messages").mock(side_effect=list_handler)
        mock.get(url__regex=r"^https://gmail\.googleapis\.com/gmail/v1/users/me/messages/m\d+(\?|$)").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "m0",
                    "threadId": "t0",
                    "snippet": "hi",
                    "payload": {"headers": [{"name": "Subject", "value": "Test"}]},
                },
            )
        )

        # 2. max_results=5 returns 5 items (NOT 10!) and passes maxResults=5 to Gmail
        r5 = await gmail.call_tool("search", {"query": "in:inbox", "max_results": 5}, ctx)
        assert not r5.is_error
        body5 = json.loads(r5.content[0].text)
        assert body5["count"] == 5
        assert len(body5["messages"]) == 5
        assert seen_max_results[-1] == "5"

        # 3. limit=3 returns 3 items and passes maxResults=3
        r3 = await gmail.call_tool("search", {"query": "in:inbox", "limit": 3}, ctx)
        assert not r3.is_error
        body3 = json.loads(r3.content[0].text)
        assert body3["count"] == 3
        assert len(body3["messages"]) == 3
        assert seen_max_results[-1] == "3"

        # 4. Conflicting limit and max_results -> error
        r_conflict = await gmail.call_tool("search", {"query": "in:inbox", "limit": 3, "max_results": 5}, ctx)
        assert r_conflict.is_error is True

        # 5. Unknown parameter -> error (matches additionalProperties: False)
        r_unknown = await gmail.call_tool("search", {"query": "in:inbox", "page_size": 5}, ctx)
        assert r_unknown.is_error is True
