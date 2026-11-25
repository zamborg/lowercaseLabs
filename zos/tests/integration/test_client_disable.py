"""Integration tests for V1 client soft-disable preserving audit and memory authorship."""

from __future__ import annotations

import json
import re

import httpx
from argon2 import PasswordHasher
from typer.testing import CliRunner

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.admin_service import AdminService
from zos.app import create_app
from zos.auth.bearer import verify_client_bearer_token
from zos.auth.context import CallContext
from zos.cli import app as cli_app
from zos.config import Settings
from zos.db.queries import init_principal_and_space
from zos.providers.memory import MemoryProvider
from zos.vault import generate_secret_key


def _extract_csrf(html_text: str) -> str:
    m = re.search(r'name="csrf_token"\s+value="([^"]+)"', html_text)
    assert m is not None
    return m.group(1)


async def test_client_disable_preserves_audit_and_memory_authorship(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    svc = AdminService(db_pool)

    # Register bearer client and write a memory + audit event as that client
    reg = await svc.register_bearer_client(name="smoke-client", trust_ceiling="owner")
    client_id = reg["client"]["id"]
    raw_token = reg["token"]

    ctx = await verify_client_bearer_token(db_pool, raw_token)
    assert ctx is not None
    assert ctx.client_id == client_id

    mem = MemoryProvider(db_pool)
    put_res = await mem.call_tool(
        "put",
        {"key": "authored-note", "content": "Written by smoke client"},
        ctx,
    )
    assert not put_res.is_error

    # Disable client via AdminService
    disabled = await svc.disable_client(client_identifier="smoke-client")
    assert disabled["id"] == client_id
    assert disabled["revoked_tokens"] == 1

    # 1. Token auth is rejected after disable
    assert await verify_client_bearer_token(db_pool, raw_token) is None

    # 2. Hidden from active client list by default
    active_names = [c["name"] for c in await svc.list_clients()]
    assert "smoke-client" not in active_names

    # 3. Audit trail retains client_id and client_name + includes client.disabled event
    events = await svc.list_events(client_id=client_id)
    ev_types = [e["type"] for e in events["events"]]
    assert "client.disabled" in ev_types
    assert "client.registered" in ev_types
    assert all(e["client_name"] == "smoke-client" for e in events["events"])

    # 4. Memory authorship survives intact!
    other_reg = await svc.register_bearer_client(name="reader-client", trust_ceiling="owner")
    reader_ctx = CallContext(
        principal_id=init["principal"]["id"],
        principal_slug="owner",
        space_id=init["space"]["id"],
        space_slug="main",
        client_id=other_reg["client"]["id"],
        client_name="reader-client",
        client_kind="bearer",
        trust_ceiling="owner",
        token_id=other_reg["token_id"],
        scopes=("mcp",),
    )
    get_res = await mem.call_tool("get", {"key": "authored-note"}, reader_ctx)
    payload = json.loads(get_res.content[0].text)
    assert payload["found"] is True
    assert payload["memory"]["author"] == "smoke-client"

    # 5. Re-registering a new bearer client with the same name succeeds (partial unique index)
    reg2 = await svc.register_bearer_client(name="smoke-client", trust_ceiling="owner")
    assert reg2["client"]["id"] != client_id


async def test_client_disable_via_admin_ui_csrf(db_pool) -> None:
    await init_principal_and_space(db_pool, principal_slug="owner")
    svc = AdminService(db_pool)
    c1 = await svc.register_bearer_client(name="ui-retire")
    await svc.register_bearer_client(name="keep-active")

    owner_pw = "owner-tidy-pw"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=base_url,
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    async with run_asgi_server(app, port=port), httpx.AsyncClient(base_url=base_url) as web:
        r_login = await web.post("/admin/login", data={"passphrase": owner_pw}, follow_redirects=False)
        assert r_login.status_code == 303

        r_clients = await web.get("/admin/clients")
        assert r_clients.status_code == 200
        assert "ui-retire" in r_clients.text
        csrf = _extract_csrf(r_clients.text)

        # Bad CSRF -> 403
        r_bad = await web.post(
            "/admin/clients/disable",
            data={"csrf_token": "wrong", "client_id": str(c1["client"]["id"])},
        )
        assert r_bad.status_code == 403

        # Valid CSRF -> disables client and removes from active list
        r_ok = await web.post(
            "/admin/clients/disable",
            data={"csrf_token": csrf, "client_id": str(c1["client"]["id"])},
        )
        assert r_ok.status_code == 200
        assert "Disabled client &#x27;ui-retire&#x27;" in r_ok.text
        assert "keep-active" in r_ok.text
        assert [c["name"] for c in await svc.list_clients()] == ["keep-active"]


def test_client_disable_via_cli(monkeypatch) -> None:
    import psycopg

    url = get_test_database_url()
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    monkeypatch.setenv("DATABASE_URL", url)

    runner = CliRunner()
    assert runner.invoke(cli_app, ["init", "--principal", "owner"]).exit_code == 0
    assert runner.invoke(cli_app, ["client", "add", "cli-retire", "--bearer"]).exit_code == 0

    res = runner.invoke(cli_app, ["client", "disable", "cli-retire"])
    assert res.exit_code == 0, res.output
    assert "Disabled client name=cli-retire" in res.output
    assert "revoked_tokens=1" in res.output
