"""v3.2 E2E & integration tests for generic devices (zos-device bridge, tier defaulting, slug uniqueness, /admin UI)."""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import os
import re
import sys
from pathlib import Path

import anyio
import httpx2
import mcp_types as types
import pytest
from argon2 import PasswordHasher
from mcp.client.session import ClientSession, IncomingMessage
from mcp.client.streamable_http import streamable_http_client

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.admin_service import AdminService
from zos.app import create_app
from zos.auth.bearer import verify_actuator_bearer_token
from zos.config import Settings
from zos.db.queries import (
    create_bearer_client,
    init_principal_and_space,
    set_tool_tier,
)
from zos.modules import ModuleManager
from zos.modules.host import ManualModuleHost
from zos.vault import generate_secret_key

DEVICE_SRC_DIR = Path(__file__).resolve().parents[2] / "device" / "src"


def _write_fake_stdio_mcp_server(path: Path) -> None:
    """Write a minimal standalone stdio MCP server exposing read-only and non-read-only tools."""
    path.write_text(
        """from __future__ import annotations
import asyncio
import json
from typing import Any
import mcp_types as types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server


async def main() -> None:
    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="ping",
                    description="Read-only health probe.",
                    input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
                    annotations=types.ToolAnnotations(read_only_hint=True),
                ),
                types.Tool(
                    name="poke",
                    description="State-changing local action (readOnlyHint=false).",
                    input_schema={"type": "object", "properties": {"target": {"type": "string"}}},
                    annotations=types.ToolAnnotations(read_only_hint=False),
                ),
                types.Tool(
                    name="raw_act",
                    description="Unannotated local action.",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]
        )

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps({"ok": True, "tool": params.name, "args": params.arguments or {}}),
                )
            ]
        )

    server = Server("fake-stdio-device", version="0.1.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


asyncio.run(main())
""",
        encoding="utf-8",
    )


async def test_v32_device_bridge_e2e_tiers_overrides_and_disconnect(db_pool, tmp_path: Path) -> None:
    """Real zos-device bridge relays a fake stdio MCP server into ZOS with tier defaulting and dynamic visibility."""
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    owner_client = await create_bearer_client(
        db_pool, principal_id=p_id, space_id=s_id, name="owner-agent", trust_ceiling="owner"
    )
    untrusted_client = await create_bearer_client(
        db_pool, principal_id=p_id, space_id=s_id, name="untrusted-agent", trust_ceiling="untrusted"
    )

    svc = AdminService(db_pool)
    dev = await svc.add_device(slug="lab")

    fake_server = tmp_path / "fake_device_server.py"
    _write_fake_stdio_mcp_server(fake_server)

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    mcp_url = f"{base_url}/mcp"
    ws_url = f"ws://127.0.0.1:{port}/actuator"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=base_url,
        zos_secret_key=generate_secret_key(),
        approval_wait_seconds=0,
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    list_changed_ev = anyio.Event()

    async def on_message(msg: IncomingMessage) -> None:
        if isinstance(msg, types.ToolListChangedNotification):
            list_changed_ev.set()

    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{DEVICE_SRC_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(DEVICE_SRC_DIR)

    async with run_asgi_server(app, port=port):
        owner_http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {owner_client['token']}"})
        untrusted_http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {untrusted_client['token']}"})
        async with (
            owner_http,
            untrusted_http,
            streamable_http_client(mcp_url, http_client=owner_http) as (r_o, w_o),
            ClientSession(r_o, w_o, message_handler=on_message) as owner_sess,
            streamable_http_client(mcp_url, http_client=untrusted_http) as (r_u, w_u),
            ClientSession(r_u, w_u) as untrusted_sess,
        ):
            await owner_sess.initialize()
            await untrusted_sess.initialize()
            await asyncio.sleep(0.15)

            # 1. Before bridge connects: lab_* tools are absent
            before_names = {t.name for t in (await owner_sess.list_tools()).tools}
            assert not any(n.startswith("lab_") for n in before_names)

            # 2. Launch real zos-device bridge subprocess
            list_changed_ev = anyio.Event()
            bridge_proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "zos_device",
                "--url",
                ws_url,
                "--token",
                dev["token"],
                "--",
                sys.executable,
                str(fake_server),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                with anyio.fail_after(10):
                    await list_changed_ev.wait()

                # 3. Tools appear with default device tier rules:
                #    readOnlyHint=True -> read; otherwise -> external
                owner_tools = {t.name: t for t in (await owner_sess.list_tools()).tools}
                assert {"lab_ping", "lab_poke", "lab_raw_act"}.issubset(owner_tools)
                assert owner_tools["lab_ping"].annotations.read_only_hint is True
                assert owner_tools["lab_ping"].annotations.open_world_hint is False
                assert owner_tools["lab_poke"].annotations.read_only_hint is False
                assert owner_tools["lab_poke"].annotations.open_world_hint is True
                assert owner_tools["lab_raw_act"].annotations.read_only_hint is False
                assert owner_tools["lab_raw_act"].annotations.open_world_hint is True

                # Untrusted client sees read-only lab_ping, NOT external lab_poke / lab_raw_act
                untrusted_names = {t.name for t in (await untrusted_sess.list_tools()).tools}
                assert "lab_ping" in untrusted_names
                assert "lab_poke" not in untrusted_names
                assert "lab_raw_act" not in untrusted_names

                # Calling read-only lab_ping succeeds directly through the bridge!
                ping_res = await owner_sess.call_tool("lab_ping", {"msg": "hello"})
                assert not ping_res.is_error
                payload = json.loads(ping_res.content[0].text)
                assert payload == {"ok": True, "tool": "ping", "args": {"msg": "hello"}}

                # Calling external lab_poke without approval returns an approval-required error
                poke_res = await owner_sess.call_tool("lab_poke", {"target": "btn"})
                assert poke_res.is_error
                assert "approval" in poke_res.content[0].text.lower()

                # 4. Explicit owner tier overrides ALWAYS win in both directions
                await set_tool_tier(db_pool, principal_id=p_id, tool_pattern="lab_poke", tier="read")
                await set_tool_tier(db_pool, principal_id=p_id, tool_pattern="lab_ping", tier="external")

                untrusted_after_override = {t.name for t in (await untrusted_sess.list_tools()).tools}
                assert "lab_poke" in untrusted_after_override
                assert "lab_ping" not in untrusted_after_override

                poke_allowed = await owner_sess.call_tool("lab_poke", {"target": "ok"})
                assert not poke_allowed.is_error
                assert json.loads(poke_allowed.content[0].text)["tool"] == "poke"

                # 5. Stop bridge -> tools disappear immediately via tools/list_changed
                list_changed_ev = anyio.Event()
                bridge_proc.terminate()
                await asyncio.wait_for(bridge_proc.wait(), timeout=5.0)

                with anyio.fail_after(10):
                    await list_changed_ev.wait()

                after_names = {t.name for t in (await owner_sess.list_tools()).tools}
                assert not any(n.startswith("lab_") for n in after_names)
            finally:
                if bridge_proc.returncode is None:
                    bridge_proc.kill()
                    with contextlib.suppress(Exception):
                        await bridge_proc.wait()


async def test_v32_slug_uniqueness_across_devices_connectors_and_modules(db_pool) -> None:
    """Slugs are unique across all provider kinds; 'mac' is no longer reserved, 'zos' still is."""
    await init_principal_and_space(db_pool, principal_slug="owner")
    svc = AdminService(db_pool)

    # 1. 'zos' remains reserved
    with pytest.raises(ValueError, match="reserved"):
        await svc.add_device(slug="zos")

    # 2. 'mac' is no longer reserved: can be registered as a device
    await svc.add_device(slug="mac")

    # Duplicate device slug fails
    with pytest.raises(ValueError, match="already in use"):
        await svc.add_device(slug="mac")

    # Connector colliding with device 'mac' fails
    with pytest.raises(ValueError, match="already in use"):
        await svc.add_remote_upstream(slug="mac", url="https://example.com/mcp", auth_mode="none")

    # 3. Device colliding with an existing connector fails
    await svc.add_remote_upstream(slug="myconn", url="https://example.com/mcp", auth_mode="none")
    with pytest.raises(ValueError, match="already in use"):
        await svc.add_device(slug="myconn")

    # 4. Module install colliding with an existing device fails (and vice versa)
    await svc.add_device(slug="inbox")
    cfg = Settings(database_url=get_test_database_url(), zos_secret_key=generate_secret_key())
    mgr = ModuleManager(db_pool, cfg, host=ManualModuleHost())
    with pytest.raises(ValueError, match="already in use"):
        await mgr.install("inbox", wait_ready=False)

    # Install 'memory' module -> adding device 'memory' fails
    try:
        await mgr.install("memory", wait_ready=False)
        with pytest.raises(ValueError, match="already in use"):
            await svc.add_device(slug="memory")
    finally:
        with contextlib.suppress(Exception):
            await mgr.uninstall("memory", purge=True)


def _extract_csrf(html_text: str) -> str:
    m = re.search(r'name="csrf_token"\s+value="([^"]+)"', html_text)
    assert m is not None, "CSRF token not found in HTML"
    return m.group(1)


async def test_v32_admin_devices_ui_lifecycle_and_audit(db_pool) -> None:
    """Admin /admin/modules 'Runs on your machine' supports add, rotate, disable, enable, remove with audit."""
    await init_principal_and_space(db_pool, principal_slug="owner")
    pw = "test-owner-pass-v32"
    pw_hash = PasswordHasher().hash(pw)

    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="https://my-zos.exe.xyz",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=pw_hash,
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="https://my-zos.exe.xyz",
            follow_redirects=False,
        ) as client,
    ):
        # Sign in
        r_login = await client.post("/admin/login", data={"passphrase": pw})
        assert r_login.status_code == 303

        # 1. GET /admin/modules shows 'Runs on your machine'
        r_mod = await client.get("/admin/modules")
        assert r_mod.status_code == 200
        assert "Runs on your machine" in r_mod.text
        csrf = _extract_csrf(r_mod.text)

        # 2. Add device -> shows one-time token, zos-device command, and config.toml
        r_add = await client.post(
            "/admin/devices/add",
            data={"csrf_token": csrf, "slug": "studio-mac"},
        )
        assert r_add.status_code == 200
        assert "studio-mac" in r_add.text
        m_tok = re.search(r"zos_a_[A-Za-z0-9_-]+", r_add.text)
        assert m_tok is not None
        tok1 = m_tok.group(0)
        assert f"zos-device --url wss://my-zos.exe.xyz/actuator --token {tok1}" in html.unescape(r_add.text)
        assert 'url = "wss://my-zos.exe.xyz/actuator"' in html.unescape(r_add.text)
        assert await verify_actuator_bearer_token(db_pool, tok1) is not None

        # 3. Rotate token -> revokes old token, mints new token
        r_rot = await client.post(
            "/admin/devices/rotate",
            data={"csrf_token": csrf, "slug": "studio-mac"},
        )
        assert r_rot.status_code == 200
        m_tok2 = re.search(r"zos_a_[A-Za-z0-9_-]+", r_rot.text)
        assert m_tok2 is not None
        tok2 = m_tok2.group(0)
        assert tok2 != tok1
        assert await verify_actuator_bearer_token(db_pool, tok1) is None
        assert await verify_actuator_bearer_token(db_pool, tok2) is not None

        # 4. Disable -> keeps row, hides tools / rejects token while disabled
        r_dis = await client.post(
            "/admin/devices/disable",
            data={"csrf_token": csrf, "slug": "studio-mac"},
        )
        assert r_dis.status_code == 200
        assert "Disabled device" in r_dis.text
        assert await verify_actuator_bearer_token(db_pool, tok2) is None

        # 5. Enable -> accepts token again
        r_en = await client.post(
            "/admin/devices/enable",
            data={"csrf_token": csrf, "slug": "studio-mac"},
        )
        assert r_en.status_code == 200
        assert "Enabled device" in r_en.text
        assert await verify_actuator_bearer_token(db_pool, tok2) is not None

        # 6. Remove -> revokes tokens and deletes provider row
        r_rem = await client.post(
            "/admin/devices/remove",
            data={"csrf_token": csrf, "slug": "studio-mac"},
        )
        assert r_rem.status_code == 200
        assert "Removed device" in r_rem.text
        assert await verify_actuator_bearer_token(db_pool, tok2) is None

    # Verify audit events were recorded for every device action
    async with db_pool.connection() as conn:
        cur = await conn.execute(
            "SELECT type, payload FROM events WHERE type LIKE 'device.%' ORDER BY id ASC"
        )
        ev_types = [row["type"] for row in await cur.fetchall()]
    assert ev_types == [
        "device.added",
        "device.token_rotated",
        "device.disabled",
        "device.enabled",
        "device.removed",
    ]
