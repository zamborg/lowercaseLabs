"""M9b tests for connector catalog validation, malformed-entry skipping, manifest tier application on
connect & list_changed (preserving owner-set tiers), --as multi-instance install, uninstall cleanup,
CLI zos catalog, and /admin/upstreams install CSRF.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import anyio
import httpx
import httpx2
import mcp_types as types
import psycopg
from argon2 import PasswordHasher
from mcp.client.session import ClientSession, IncomingMessage
from mcp.client.streamable_http import streamable_http_client
from mcp.server.connection import Connection
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from psycopg.rows import dict_row
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from typer.testing import CliRunner

from tests.conftest import MIGRATIONS_DIR, get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.catalog import default_catalog_dir, load_catalog, load_manifest_file
from zos.cli import app as cli_app
from zos.config import Settings
from zos.db import open_pool, run_migrations
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.vault import generate_secret_key


class _FakeCatServer(Server[Any]):
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


class _FakeCatControl:
    def __init__(self) -> None:
        self.include_new_tool = False
        self.connections: set[Connection] = set()

    async def enable_new_tool(self) -> None:
        self.include_new_tool = True
        for conn in list(self.connections):
            await conn.send_tool_list_changed()


def _build_fake_catalog_upstream(ctrl: _FakeCatControl) -> Starlette:
    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        tools = [
            types.Tool(
                name="list_items",
                description="List items",
                input_schema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="create_item",
                description="Create item",
                input_schema={"type": "object", "properties": {}},
            ),
        ]
        if ctrl.include_new_tool:
            tools.append(
                types.Tool(
                    name="get_stats",
                    description="Get stats (added dynamically)",
                    input_schema={"type": "object", "properties": {}},
                )
            )
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(rctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        return types.CallToolResult(content=[types.TextContent(type="text", text=f"ok:{params.name}")])

    async def on_initialized(rctx: Any, params: Any) -> None:
        conn: Connection = rctx.session._connection
        ctrl.connections.add(conn)
        conn.exit_stack.callback(ctrl.connections.discard, conn)

    server = _FakeCatServer(
        "fake-cat-upstream",
        version="1.0.0",
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

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/mcp", endpoint=StreamableHTTPASGIApp(mgr), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lambda a: mgr.run(),
    )


def _extract_csrf(html_body: str) -> str:
    m = re.search(r'<input type="hidden" name="csrf_token" value="([^"]+)">', html_body)
    assert m is not None
    return m.group(1)


def test_all_checked_in_manifests_validate() -> None:
    cat_dir = default_catalog_dir()
    toml_files = sorted(cat_dir.glob("*.toml"))
    assert len(toml_files) >= 15
    for path in toml_files:
        manifest = load_manifest_file(path)
        assert manifest.manifest_version == 1
        assert manifest.slug == path.stem
        assert manifest.runtime.kind == "remote"

    loaded = load_catalog(cat_dir)
    assert len(loaded) == len(toml_files)
    slugs = {m.slug for m in loaded}
    for required in ("github", "linear", "notion", "cloudflare", "cfdocs", "slack", "atlassian", "sentry", "stripe", "asana"):
        assert required in slugs


def test_malformed_manifest_skipped_without_breaking_load(tmp_path: Path) -> None:
    (tmp_path / "good.toml").write_text(
        'manifest_version = 1\n'
        'slug = "good"\n'
        'name = "Good"\n'
        'description = "Valid entry"\n'
        'docs_url = "https://example.com/docs"\n'
        '[runtime]\nkind = "remote"\nurl = "https://example.com/mcp"\n'
        '[auth]\nmode = "none"\n'
        '[tiers]\ndefault = "external"\n"*_list*" = "read"\n',
        encoding="utf-8",
    )
    (tmp_path / "broken_toml.toml").write_text("not valid [[[ toml", encoding="utf-8")
    (tmp_path / "bad_slug.toml").write_text(
        'manifest_version = 1\nslug = "zos"\nname = "Bad"\ndescription = "x"\ndocs_url = "https://example.com"\n'
        '[runtime]\nkind = "remote"\nurl = "https://example.com/mcp"\n[auth]\nmode = "none"\n',
        encoding="utf-8",
    )
    (tmp_path / "bad_version.toml").write_text(
        'manifest_version = 99\nslug = "badv"\nname = "BadV"\ndescription = "x"\ndocs_url = "https://example.com"\n'
        '[runtime]\nkind = "remote"\nurl = "https://example.com/mcp"\n[auth]\nmode = "none"\n',
        encoding="utf-8",
    )

    loaded = load_catalog(tmp_path)
    assert [m.slug for m in loaded] == ["good"]


async def test_catalog_install_tiers_list_changed_owner_override_as_and_csrf(db_pool, tmp_path: Path) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]
    mcp_client = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="agent-cat")

    p_fake, p_zos = free_port(), free_port()
    ctrl = _FakeCatControl()
    fake_url = f"http://127.0.0.1:{p_fake}/mcp"

    # Temporary catalog pointing at our in-process fake upstream
    (tmp_path / "demo.toml").write_text(
        'manifest_version = 1\n'
        'slug = "demo"\n'
        'name = "Demo MCP"\n'
        'description = "In-process demo connector"\n'
        'docs_url = "https://example.com/demo"\n'
        f'[runtime]\nkind = "remote"\nurl = "{fake_url}"\n'
        '[auth]\nmode = "none"\n'
        '[tiers]\n'
        'default = "external"\n'
        '"*_list*" = "read"\n'
        '"*_get*" = "read"\n',
        encoding="utf-8",
    )

    owner_pw = "cat-owner-pw"
    zos_base = f"http://127.0.0.1:{p_zos}"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=zos_base,
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
        catalog_dir=tmp_path,
    )
    app = create_app(cfg, run_migrations_on_startup=False, upstream_connect_timeout_s=5.0)

    async with (
        run_asgi_server(_build_fake_catalog_upstream(ctrl), port=p_fake),
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

            assert (await web.post("/admin/login", data={"passphrase": owner_pw})).status_code == 303
            r_up = await web.get("/admin/upstreams")
            assert r_up.status_code == 200
            assert "Available connectors" in r_up.text
            assert "Demo MCP" in r_up.text
            csrf = _extract_csrf(r_up.text)

            # 1. Admin install CSRF check
            r_no_csrf = await web.post("/admin/upstreams/install", data={"catalog_slug": "demo"})
            assert r_no_csrf.status_code == 403
            r_bad_csrf = await web.post(
                "/admin/upstreams/install",
                data={"csrf_token": "wrong", "catalog_slug": "demo"},
            )
            assert r_bad_csrf.status_code == 403

            # 2. Install `demo` -> connects, applies manifest tiers (`demo_list_items` -> read, `demo_create_item` -> external)
            list_changed_ev = anyio.Event()
            r_inst = await web.post(
                "/admin/upstreams/install",
                data={"csrf_token": csrf, "catalog_slug": "demo"},
            )
            assert r_inst.status_code == 200
            assert "installed" in r_inst.text
            with anyio.fail_after(5):
                await list_changed_ev.wait()

            tools1 = {t.name: t for t in (await sess.list_tools()).tools}
            assert "demo_list_items" in tools1 and "demo_create_item" in tools1
            assert tools1["demo_list_items"].annotations.read_only_hint is True
            assert tools1["demo_create_item"].annotations.open_world_hint is True

            # 3. Owner sets `demo_list_items` -> `external` by hand; survives `list_changed` while new `demo_get_stats` gets `read`!
            r_hand = await web.post(
                "/admin/policy/tier",
                data={"csrf_token": csrf, "pattern": "demo_list_items", "tier": "external"},
            )
            assert r_hand.status_code == 200

            list_changed_ev = anyio.Event()
            await ctrl.enable_new_tool()
            with anyio.fail_after(5):
                await list_changed_ev.wait()

            tools2 = {t.name: t for t in (await sess.list_tools()).tools}
            assert "demo_get_stats" in tools2
            # Owner-set tier survived list_changed!
            assert tools2["demo_list_items"].annotations.open_world_hint is True
            assert tools2["demo_list_items"].annotations.read_only_hint is False
            # New tool resolved from manifest pattern `*_get*` -> read!
            assert tools2["demo_get_stats"].annotations.read_only_hint is True

            # 4. Install second instance with `--as workdemo` via Admin UI
            list_changed_ev = anyio.Event()
            r_inst2 = await web.post(
                "/admin/upstreams/install",
                data={"csrf_token": csrf, "catalog_slug": "demo", "as_slug": "workdemo"},
            )
            assert r_inst2.status_code == 200
            with anyio.fail_after(5):
                await list_changed_ev.wait()

            tools3 = {t.name: t for t in (await sess.list_tools()).tools}
            assert "workdemo_list_items" in tools3
            assert tools3["workdemo_list_items"].annotations.read_only_hint is True
            assert tools3["workdemo_get_stats"].annotations.read_only_hint is True

            # 5. Uninstall `demo` drops manifest-managed tier (`demo_get_stats`), preserves owner-set tier (`demo_list_items`)
            r_rem = await web.post("/admin/upstreams/remove", data={"csrf_token": csrf, "slug": "demo"})
            assert r_rem.status_code == 200

            async with db_pool.connection() as conn:
                rows = await (
                    await conn.execute(
                        "SELECT tool_pattern, tier FROM tool_tiers WHERE tool_pattern LIKE 'demo_%'"
                    )
                ).fetchall()
                tiers_left = {r["tool_pattern"]: r["tier"] for r in rows}
                assert "demo_get_stats" not in tiers_left
                assert tiers_left.get("demo_list_items") == "external"

                cat_evs = await (
                    await conn.execute("SELECT payload FROM events WHERE type = 'catalog.installed' ORDER BY id ASC")
                ).fetchall()
                assert len(cat_evs) == 2
                assert cat_evs[0]["payload"]["slug"] == "demo"
                assert cat_evs[1]["payload"]["slug"] == "workdemo"


def test_cli_catalog_list_and_install_with_as(monkeypatch) -> None:
    url = get_test_database_url()

    async def _reset() -> None:
        async with open_pool(url) as pool:
            await run_migrations(pool, MIGRATIONS_DIR)
            async with pool.connection() as conn:
                await conn.execute("TRUNCATE principals RESTART IDENTITY CASCADE")
                await conn.commit()

    asyncio.run(_reset())
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("ZOS_BASE_URL", "https://zos.example.com")

    runner = CliRunner()
    assert runner.invoke(cli_app, ["init", "--principal", "zubin"]).exit_code == 0

    r_list = runner.invoke(cli_app, ["catalog", "list"])
    assert r_list.exit_code == 0, r_list.output
    assert "linear" in r_list.output and "github" in r_list.output and "cfdocs" in r_list.output

    r_i1 = runner.invoke(cli_app, ["catalog", "install", "linear"])
    assert r_i1.exit_code == 0, r_i1.output
    assert "https://zos.example.com/connect/linear" in r_i1.output

    r_i2 = runner.invoke(cli_app, ["catalog", "install", "linear", "--as", "worklinear"])
    assert r_i2.exit_code == 0, r_i2.output
    assert "slug=worklinear" in r_i2.output
    assert "https://zos.example.com/connect/worklinear" in r_i2.output

    with psycopg.connect(url, row_factory=dict_row) as conn:
        slugs = [r["slug"] for r in conn.execute("SELECT slug FROM providers WHERE kind = 'upstream' ORDER BY slug").fetchall()]
    assert slugs == ["linear", "worklinear"]
