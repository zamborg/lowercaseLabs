"""Integration test: module tools enforce visibility + allowance chain and appear in /admin/policy."""

from __future__ import annotations

import contextlib
import json
from typing import Any

import httpx
import mcp_types as types
from argon2 import PasswordHasher
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.admin_service import AdminService
from zos.app import create_app
from zos.auth.bearer import verify_client_bearer_token
from zos.catalog import ModuleManifest
from zos.config import Settings
from zos.db.queries import init_principal_and_space
from zos.modules import InstalledModule, ModuleManager, ModuleRuntimeStatus
from zos.vault import generate_secret_key


class _InboxStubHost:
    kind = "docker"

    def __init__(self, mcp_url: str, ui_url: str) -> None:
        self.mcp_url = mcp_url
        self.ui_url = ui_url

    async def install(self, manifest: ModuleManifest, config: dict[str, str], *, env: dict[str, str]) -> InstalledModule:
        return InstalledModule(
            mcp_url=self.mcp_url,
            ui_url=self.ui_url,
            image_pin="sha256:inbox-stub",
            host="docker",
        )

    async def stop(self, slug: str) -> None:
        return None

    async def start(self, slug: str) -> None:
        return None

    async def uninstall(self, slug: str, *, purge: bool = False) -> None:
        return None

    async def status(self, slug: str) -> ModuleRuntimeStatus:
        return ModuleRuntimeStatus(host="docker", state="running")

    async def upgrade(self, slug: str, manifest: ModuleManifest, config: dict[str, str], *, env: dict[str, str]) -> InstalledModule:
        return await self.install(manifest, config, env=env)


def _fake_inbox_mcp_server() -> Starlette:
    tool_names = ("send", "list", "read", "archive", "directory")

    async def on_list_tools(rctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=n,
                    description=f"inbox {n}",
                    input_schema={"type": "object", "properties": {}},
                )
                for n in tool_names
            ]
        )

    async def on_call_tool(rctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps({"ok": True, "tool": params.name}))]
        )

    srv = Server("fake-inbox", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
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

    return Starlette(
        routes=[
            Route("/healthz", lambda _req: PlainTextResponse("ok"), methods=["GET"]),
            Route("/mcp", endpoint=StreamableHTTPASGIApp(sm), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )


async def test_inbox_module_tools_follow_policy_visibility_and_admin_policy(db_pool: Any) -> None:
    """Per-client deny on `inbox_*` hides all inbox tools from that client, and `/admin/policy` shows them."""
    db_url = get_test_database_url()
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    svc = AdminService(db_pool)

    reg_ok = await svc.register_bearer_client(name="allowed-agent", trust_ceiling="owner")
    reg_deny = await svc.register_bearer_client(name="denied-agent", trust_ceiling="owner")
    ok_ctx = await verify_client_bearer_token(db_pool, reg_ok["token"])
    deny_ctx = await verify_client_bearer_token(db_pool, reg_deny["token"])
    assert ok_ctx is not None and deny_ctx is not None

    # Deny all inbox_* tools for denied-agent
    await svc.set_policy(pattern="inbox_*", decision="deny", client_identifier="denied-agent")

    port = free_port()
    owner_pw = "owner-policy-pass"
    cfg = Settings(
        database_url=db_url,
        zos_base_url=f"http://127.0.0.1:{free_port()}",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
    )
    stub_host = _InboxStubHost(
        mcp_url=f"http://127.0.0.1:{port}/mcp",
        ui_url=f"http://127.0.0.1:{port}",
    )
    app = create_app(cfg, module_host=stub_host)

    try:
        async with run_asgi_server(_fake_inbox_mcp_server(), port=port), app.router.lifespan_context(app):
            rt = app.state.zos
            await rt.module_manager.install("inbox")

            # 1. allowed-agent sees all 5 inbox_* tools and can call inbox_list
            ok_names = {t.name for t in await rt.router.list_tools(ok_ctx)}
            assert {"inbox_send", "inbox_list", "inbox_read", "inbox_archive", "inbox_directory"} <= ok_names
            r_ok = await rt.router.call_tool("inbox_list", {}, ok_ctx)
            assert not r_ok.is_error

            # 2. denied-agent sees ZERO inbox_* tools, and calling inbox_list is rejected by visibility
            deny_names = {t.name for t in await rt.router.list_tools(deny_ctx)}
            assert not any(n.startswith("inbox_") for n in deny_names)
            r_deny = await rt.router.call_tool("inbox_list", {}, deny_ctx)
            assert r_deny.is_error
            assert "Permission denied" in r_deny.content[0].text

            async with rt.pool.connection() as conn:
                ev = await (
                    await conn.execute(
                        """
                        SELECT decision, outcome, payload
                        FROM events
                        WHERE type = 'tool.called' AND client_id = %s AND tool = 'inbox_list'
                        ORDER BY id DESC LIMIT 1
                        """,
                        (deny_ctx.client_id,),
                    )
                ).fetchone()
            assert ev is not None
            assert ev["decision"] == "deny"
            assert ev["outcome"] == "denied"
            assert ev["payload"]["engine"] == "visibility"

            # 3. /admin/policy lists all 5 inbox_* tools with their manifest tiers and overrides
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=cfg.zos_base_url) as web:
                await web.post("/admin/login", data={"passphrase": owner_pw}, follow_redirects=False)
                r_pol = await web.get("/admin/policy")
                assert r_pol.status_code == 200
                for tname in ("inbox_send", "inbox_list", "inbox_read", "inbox_archive", "inbox_directory"):
                    assert tname in r_pol.text
    finally:
        await ModuleManager(db_pool, cfg, host=stub_host).storage.purge_storage(
            principal_id=init["principal"]["id"],
            slug="inbox",
        )
