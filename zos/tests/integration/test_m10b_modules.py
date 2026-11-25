"""Integration tests for M10b: Postgres role isolation, MCP caller signing, ModuleManager lifecycle, and /admin/modules."""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

import httpx
import mcp_types as types
import psycopg
import pytest
from argon2 import PasswordHasher
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from zos_module_sdk.caller import CallerVerificationError, verify_mcp_caller

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.admin_service import AdminService
from zos.app import create_app
from zos.auth.bearer import verify_client_bearer_token
from zos.catalog import ModuleManifest
from zos.config import Settings
from zos.db.queries import init_principal_and_space
from zos.modules import InstalledModule, ManualModuleHost, ModuleManager, ModuleRuntimeStatus
from zos.vault import generate_secret_key

DIGEST = "b" * 64


def _write_demo_manifest(modules_dir: Path, slug: str = "echo") -> None:
    modules_dir.mkdir(parents=True, exist_ok=True)
    (modules_dir / f"{slug}.toml").write_text(
        f"""
manifest_version = 2
slug = "{slug}"
name = "Echo Module"
version = "0.1.0"
description = "Test echo module"

[runtime]
kind = "container"
image = "ghcr.io/example/{slug}@sha256:{DIGEST}"
mcp_port = 8081
mcp_path = "/mcp"
ui_port = 8082

[storage]
postgres = true

[network]
egress = false

[ui]
title = "Echo"

[tiers]
default = "write"
"{slug}_ping" = "read"
""",
        encoding="utf-8",
    )


class _FakeDockerHost:
    """In-memory ModuleHost stand-in that records lifecycle calls and points at a local test MCP server."""

    kind = "docker"

    def __init__(self, mcp_url: str, ui_url: str | None = None) -> None:
        self.mcp_url = mcp_url
        self.ui_url = ui_url
        self.last_env: dict[str, str] = {}
        self.stopped: list[str] = []
        self.started: list[str] = []
        self.uninstalled: list[tuple[str, bool]] = []

    async def install(
        self,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str],
    ) -> InstalledModule:
        self.last_env = dict(env)
        return InstalledModule(
            mcp_url=self.mcp_url,
            ui_url=self.ui_url,
            image_pin=manifest.runtime.image or "sha256:fake",
            host="docker",
        )

    async def stop(self, slug: str) -> None:
        self.stopped.append(slug)

    async def start(self, slug: str) -> None:
        self.started.append(slug)

    async def uninstall(self, slug: str, *, purge: bool = False) -> None:
        self.uninstalled.append((slug, purge))

    async def status(self, slug: str) -> ModuleRuntimeStatus:
        return ModuleRuntimeStatus(host="docker", state="running")

    async def upgrade(
        self,
        slug: str,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str],
    ) -> InstalledModule:
        return await self.install(manifest, config, env=env)


def _create_fake_module_mcp_app(secret_holder: dict[str, str]) -> Starlette:
    """Real Streamable-HTTP MCP server that verifies `_meta['zos/caller']` on `tools/call`."""
    observed: list[dict[str, Any]] = []

    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        observed.append({"method": "tools/list", "meta": getattr(params, "meta", None)})
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="ping",
                    description="Ping returning verified caller",
                    input_schema={
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                    },
                ),
                types.Tool(
                    name="write_note",
                    description="Write note",
                    input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                ),
            ]
        )

    async def on_call_tool(rctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = dict(params.arguments or {})
        meta = params.meta
        observed.append({"method": "tools/call", "name": params.name, "meta": meta})
        try:
            caller = verify_mcp_caller(
                meta,
                secret=secret_holder["secret"],
                tool=params.name,
                args=args,
            )
        except CallerVerificationError as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"CALLER_ERR:{exc}")],
                is_error=True,
            )
        if caller is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="UNSIGNED_SYSTEM_CALL")],
                is_error=True,
            )
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "tool": params.name,
                            "client_name": caller.client_name,
                            "client_id": caller.client_id,
                            "principal_id": caller.principal_id,
                            "args": args,
                        }
                    ),
                )
            ]
        )

    srv = Server("fake-mod", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    sm = StreamableHTTPSessionManager(
        app=srv,
        json_response=True,
        stateless=True,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with sm.run():
            yield

    app = Starlette(
        routes=[
            Route("/healthz", lambda _req: PlainTextResponse("ok"), methods=["GET"]),
            Route("/mcp", endpoint=StreamableHTTPASGIApp(sm), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )
    app.state.observed = observed
    return app


async def test_module_postgres_role_isolation_and_purge(db_pool: Any, tmp_path: Path) -> None:
    db_url = get_test_database_url()
    await init_principal_and_space(db_pool, principal_slug="owner")
    modules_dir = tmp_path / "modules"
    _write_demo_manifest(modules_dir, slug="iso")

    cfg = Settings(
        database_url=db_url,
        zos_secret_key=generate_secret_key(),
        modules_catalog_dir=modules_dir,
    )
    fake_host = _FakeDockerHost(mcp_url="http://127.0.0.1:19999/mcp")
    mgr = ModuleManager(db_pool, cfg, host=fake_host, modules_dir=modules_dir)

    try:
        await mgr.install("iso", wait_ready=False)
        mod_dsn = fake_host.last_env["ZOS_MODULE_DATABASE_URL"]
        assert "mod_iso:" in mod_dsn

        # Connect AS the module role `mod_iso`
        async with await psycopg.AsyncConnection.connect(mod_dsn, autocommit=True) as mod_conn:
            # 1. Can create & query tables in its own schema `mod_iso`
            await mod_conn.execute("CREATE TABLE items (id INT PRIMARY KEY, body TEXT)")
            await mod_conn.execute("INSERT INTO items (id, body) VALUES (1, 'hello')")
            cur = await mod_conn.execute("SELECT body FROM items WHERE id = 1")
            row = await cur.fetchone()
            assert row == ("hello",)

            # 2. CANNOT read core ZOS tables!
            for core_table in ("credentials", "events", "memories", "tokens"):
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    await mod_conn.execute(f"SELECT COUNT(*) FROM public.{core_table}")

        # Uninstall WITHOUT purge keeps schema `mod_iso` and data intact
        await mgr.uninstall("iso", purge=False)
        async with db_pool.connection() as conn:
            cur = await conn.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'mod_iso'"
            )
            assert await cur.fetchone() is not None

        # Re-install and uninstall WITH purge drops schema and role cleanly
        await mgr.install("iso", wait_ready=False)
        await mgr.uninstall("iso", purge=True)
        async with db_pool.connection() as conn:
            cur = await conn.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'mod_iso'"
            )
            assert await cur.fetchone() is None
            cur = await conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'mod_iso'")
            assert await cur.fetchone() is None
    finally:
        # Ensure clean teardown even on assertion failure
        await mgr.storage.purge_storage(
            principal_id=(await init_principal_and_space(db_pool, principal_slug="owner"))["principal"]["id"],
            slug="iso",
        )


async def test_module_lifecycle_signed_caller_and_admin_ui(db_pool: Any, tmp_path: Path) -> None:
    db_url = get_test_database_url()
    await init_principal_and_space(db_pool, principal_slug="owner")
    svc = AdminService(db_pool)
    reg = await svc.register_bearer_client(name="alice-agent", trust_ceiling="owner")
    alice_ctx = await verify_client_bearer_token(db_pool, reg["token"])
    assert alice_ctx is not None

    modules_dir = tmp_path / "catalog_modules"
    _write_demo_manifest(modules_dir, slug="echo")

    secret_holder = {"secret": ""}
    mod_port = free_port()
    mod_app = _create_fake_module_mcp_app(secret_holder)

    owner_pw = "owner-m10b-passphrase"
    cfg = Settings(
        database_url=db_url,
        zos_base_url=f"http://127.0.0.1:{free_port()}",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
        modules_catalog_dir=modules_dir,
    )
    fake_host = _FakeDockerHost(
        mcp_url=f"http://127.0.0.1:{mod_port}/mcp",
        ui_url=f"http://127.0.0.1:{mod_port}",
    )
    app = create_app(cfg, module_host=fake_host)

    try:
        async with run_asgi_server(mod_app, port=mod_port), app.router.lifespan_context(app):
            rt = app.state.zos
            # Pre-seed a manual owner tier override to verify manifest doesn't overwrite it
            await svc.set_tool_tier(pattern="echo_write_note", tier="external")

            # Install via /admin/modules POST (also tests CSRF & auth!)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=cfg.zos_base_url) as web:
                r_unauth = await web.get("/admin/modules", follow_redirects=False)
                assert r_unauth.status_code == 303

                r_login = await web.post(
                    "/admin/login",
                    data={"passphrase": owner_pw},
                    follow_redirects=False,
                )
                assert r_login.status_code == 303

                r_page = await web.get("/admin/modules")
                assert r_page.status_code == 200
                assert "Echo Module" in r_page.text
                csrf_m = re.search(r'name="csrf_token"\s+value="([^"]+)"', r_page.text)
                assert csrf_m is not None
                csrf = csrf_m.group(1)

                r_install = await web.post(
                    "/admin/modules/install",
                    data={"csrf_token": csrf, "slug": "echo"},
                )
                assert r_install.status_code == 200
                assert "Installed module" in r_install.text

            # Give fake module the provisioned secret (same as container env `ZOS_MODULE_SECRET`)
            secret_holder["secret"] = fake_host.last_env["ZOS_MODULE_SECRET"]
            assert secret_holder["secret"]

            # Tool list contains echo_ping & echo_write_note; tools/list was unsigned system call
            tools = await rt.router.list_tools(alice_ctx)
            names = [t.name for t in tools]
            assert "echo_ping" in names
            assert "echo_write_note" in names
            list_obs = [o for o in mod_app.state.observed if o["method"] == "tools/list"]
            assert list_obs and list_obs[0]["meta"] is None

            # Manifest tier applied to echo_ping (read) without overwriting owner tier on echo_write_note (external)
            ping_tool = next(t for t in tools if t.name == "echo_ping")
            note_tool = next(t for t in tools if t.name == "echo_write_note")
            assert ping_tool.annotations is not None and ping_tool.annotations.read_only_hint is True
            assert note_tool.annotations is not None and note_tool.annotations.open_world_hint is True

            # Calling echo_ping forwards signed `_meta["zos/caller"]` matching alice-agent!
            call_res = await rt.router.call_tool("echo_ping", {"msg": "hello-m10b"}, alice_ctx)
            assert not call_res.is_error
            body = json.loads(call_res.content[0].text)
            assert body["tool"] == "ping"
            assert body["client_name"] == "alice-agent"
            assert body["client_id"] == str(alice_ctx.client_id)
            assert body["args"] == {"msg": "hello-m10b"}

            # Disable stops container and hides tools
            await rt.module_manager.disable("echo")
            assert fake_host.stopped == ["echo"]
            names_after_disable = [t.name for t in await rt.router.list_tools(alice_ctx)]
            assert "echo_ping" not in names_after_disable

            # Enable starts container and restores tools
            await rt.module_manager.enable("echo")
            assert fake_host.started == ["echo"]
            names_after_enable = [t.name for t in await rt.router.list_tools(alice_ctx)]
            assert "echo_ping" in names_after_enable

            # Uninstall removes container & upstream
            await rt.module_manager.uninstall("echo", purge=True)
            assert fake_host.uninstalled == [("echo", True)]
            names_after_uninstall = [t.name for t in await rt.router.list_tools(alice_ctx)]
            assert "echo_ping" not in names_after_uninstall

            # Verify audit events recorded for full lifecycle
            async with rt.pool.connection() as conn:
                ev_types = [
                    r["type"]
                    for r in await (
                        await conn.execute(
                            "SELECT type FROM events WHERE type LIKE 'module.%' ORDER BY id ASC"
                        )
                    ).fetchall()
                ]
            assert ev_types == [
                "module.installed",
                "module.disabled",
                "module.enabled",
                "module.uninstalled",
            ]
    finally:
        await ModuleManager(db_pool, cfg, host=fake_host, modules_dir=modules_dir).storage.purge_storage(
            principal_id=alice_ctx.principal_id,
            slug="echo",
        )


async def test_manual_module_host_install_and_attach(db_pool: Any, tmp_path: Path) -> None:
    db_url = get_test_database_url()
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    modules_dir = tmp_path / "manual_modules"
    _write_demo_manifest(modules_dir, slug="manmod")

    cfg = Settings(
        database_url=db_url,
        zos_secret_key=generate_secret_key(),
        module_host="manual",
        modules_catalog_dir=modules_dir,
    )
    mgr = ModuleManager(db_pool, cfg, host=ManualModuleHost(), modules_dir=modules_dir)
    try:
        res = await mgr.install("manmod", wait_ready=False)
        assert res["module"]["status"] == "pending_attach"
        assert res["instructions"] and "zos module attach manmod" in res["instructions"]

        attached = await mgr.attach(
            "manmod",
            mcp_url="http://127.0.0.1:18081/mcp",
            ui_url="http://127.0.0.1:18082",
        )
        assert attached["status"] == "installed"
        assert attached["mcp_url"] == "http://127.0.0.1:18081/mcp"
        assert attached["ui_url"] == "http://127.0.0.1:18082"
        assert attached["upstream_id"] is not None
    finally:
        await mgr.uninstall("manmod", purge=True)
        await mgr.storage.purge_storage(principal_id=init["principal"]["id"], slug="manmod")
