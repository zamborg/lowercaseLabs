"""Integration & unit tests for M10c: `/admin/m/<slug>` reverse proxy, cookie/header isolation, and nav."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from argon2 import PasswordHasher
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route
from zos_module_sdk.caller import verify_http_caller

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.catalog import ModuleManifest
from zos.config import Settings
from zos.db.queries import init_principal_and_space
from zos.modules import InstalledModule, ModuleManager, ModuleRuntimeStatus
from zos.vault import generate_secret_key
from zos.web.module_proxy import (
    filter_and_scope_set_cookie,
    rewrite_location_header,
    strip_admin_cookies,
)

DIGEST = "c" * 64


def test_proxy_header_and_cookie_helpers() -> None:
    assert strip_admin_cookies("zos_admin_session=secret; inbox_csrf=abc123") == "inbox_csrf=abc123"
    assert strip_admin_cookies("zos_admin_session=secret") is None
    assert strip_admin_cookies("a=1; ZOS_ADMIN_SESSION=secret; b=2") == "a=1; b=2"

    assert (
        rewrite_location_header("/threads/42?x=1", prefix="/admin/m/inbox", upstream_base_url="http://zos-mod-inbox:8082")
        == "/admin/m/inbox/threads/42?x=1"
    )
    assert (
        rewrite_location_header(
            "http://zos-mod-inbox:8082/compose",
            prefix="/admin/m/inbox",
            upstream_base_url="http://zos-mod-inbox:8082",
        )
        == "/admin/m/inbox/compose"
    )
    assert (
        rewrite_location_header(
            "/admin/m/inbox/already",
            prefix="/admin/m/inbox",
            upstream_base_url="http://zos-mod-inbox:8082",
        )
        == "/admin/m/inbox/already"
    )
    assert (
        rewrite_location_header(
            "https://example.com/external",
            prefix="/admin/m/inbox",
            upstream_base_url="http://zos-mod-inbox:8082",
        )
        == "https://example.com/external"
    )

    assert filter_and_scope_set_cookie("zos_admin_session=evil; Path=/", prefix="/admin/m/inbox") is None
    assert filter_and_scope_set_cookie("zos_admin_foo=evil; Path=/", prefix="/admin/m/inbox") is None
    scoped = filter_and_scope_set_cookie("inbox_csrf=tok123; Path=/; HttpOnly", prefix="/admin/m/inbox")
    assert scoped is not None
    assert "Path=/admin/m/inbox" in scoped
    assert "inbox_csrf=tok123" in scoped
    scoped_default = filter_and_scope_set_cookie("inbox_csrf=tok456", prefix="/admin/m/inbox", secure_cookie=True)
    assert scoped_default == "inbox_csrf=tok456; Path=/admin/m/inbox; Secure"


class _StubUIHost:
    kind = "docker"

    def __init__(self, *, mcp_url: str, ui_url: str | None) -> None:
        self.mcp_url = mcp_url
        self.ui_url = ui_url
        self.last_env: dict[str, str] = {}

    async def install(self, manifest: ModuleManifest, config: dict[str, str], *, env: dict[str, str]) -> InstalledModule:
        self.last_env = dict(env)
        return InstalledModule(
            mcp_url=self.mcp_url,
            ui_url=self.ui_url,
            image_pin=manifest.runtime.image or "sha256:stub",
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


def _create_fake_ui_app(secret_holder: dict[str, str]) -> Starlette:
    async def index(request: Request) -> JSONResponse:
        caller = verify_http_caller(dict(request.headers), secret=secret_holder["secret"])
        resp = JSONResponse(
            {
                "path": request.url.path,
                "query": request.url.query,
                "prefix": request.headers.get("x-forwarded-prefix"),
                "cookie": request.headers.get("cookie"),
                "authorization": request.headers.get("authorization"),
                "caller_name": caller.client_name,
                "caller_principal": caller.principal_id,
            }
        )
        # Attempt to set BOTH a legitimate module cookie and a forbidden admin-named cookie
        resp.headers.append("Set-Cookie", "mod_csrf=good123; Path=/; HttpOnly")
        resp.headers.append("Set-Cookie", "zos_admin_session=hijack; Path=/")
        return resp

    async def bounce(request: Request) -> RedirectResponse:
        return RedirectResponse("/threads/99?tab=all", status_code=303)

    return Starlette(
        routes=[
            Route("/healthz", lambda _req: PlainTextResponse("ok"), methods=["GET"]),
            Route("/", index, methods=["GET", "POST"]),
            Route("/bounce", bounce, methods=["POST"]),
            Route("/{sub:path}", index, methods=["GET", "POST"]),
        ]
    )


async def test_module_ui_proxy_end_to_end(db_pool: Any, tmp_path: Path) -> None:
    db_url = get_test_database_url()
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    modules_dir = tmp_path / "mods"
    modules_dir.mkdir(parents=True, exist_ok=True)
    (modules_dir / "board.toml").write_text(
        f"""
manifest_version = 2
slug = "board"
name = "Board"
version = "0.1.0"
description = "Test UI module"

[runtime]
kind = "container"
image = "ghcr.io/example/board@sha256:{DIGEST}"
mcp_port = 8081
ui_port = 8082

[ui]
title = "Board UI"
""",
        encoding="utf-8",
    )

    secret_holder = {"secret": ""}
    ui_port = free_port()
    ui_app = _create_fake_ui_app(secret_holder)

    owner_pw = "owner-m10c-secret"
    cfg = Settings(
        database_url=db_url,
        zos_base_url=f"http://127.0.0.1:{free_port()}",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
        modules_catalog_dir=modules_dir,
    )
    fake_host = _StubUIHost(
        mcp_url="http://127.0.0.1:19998/mcp",
        ui_url=f"http://127.0.0.1:{ui_port}",
    )
    app = create_app(cfg, module_host=fake_host)

    try:
        async with run_asgi_server(ui_app, port=ui_port), app.router.lifespan_context(app):
            rt = app.state.zos
            await rt.module_manager.install("board", wait_ready=False)
            secret_holder["secret"] = fake_host.last_env["ZOS_MODULE_SECRET"]

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=cfg.zos_base_url) as client:
                # 1. Unauthenticated request redirects to /admin/login?next=...
                r_unauth = await client.get("/admin/m/board/items?a=1", follow_redirects=False)
                assert r_unauth.status_code == 303
                assert "/admin/login?next=/admin/m/board/items" in r_unauth.headers["location"]

                # Login as owner
                r_login = await client.post(
                    "/admin/login",
                    data={"passphrase": owner_pw},
                    follow_redirects=False,
                )
                assert r_login.status_code == 303

                # 2. Overview & nav include link to Board UI
                r_ov = await client.get("/admin")
                assert r_ov.status_code == 200
                assert "/admin/m/board/" in r_ov.text
                assert "Board UI" in r_ov.text

                # 3. Unknown module returns 404
                r_404 = await client.get("/admin/m/nonexistent/")
                assert r_404.status_code == 404

                # 4. Proxy strips zos_admin_session & Authorization, preserves module cookie,
                #    sets X-Forwarded-Prefix, signs owner caller, filters/scopes Set-Cookie
                admin_cookie = client.cookies.get("zos_admin_session")
                assert admin_cookie
                r_prox = await client.get(
                    "/admin/m/board/sub/page?k=v",
                    headers={
                        "Authorization": "Bearer secret-should-be-stripped",
                        "Cookie": f"zos_admin_session={admin_cookie}; mod_csrf=kept99",
                        "X-ZOS-Caller": "spoofed",
                        "X-ZOS-Caller-Signature": "spoofed",
                    },
                )
                assert r_prox.status_code == 200
                data = json.loads(r_prox.text)
                assert data["path"] == "/sub/page"
                assert data["query"] == "k=v"
                assert data["prefix"] == "/admin/m/board"
                assert data["authorization"] is None
                assert data["cookie"] == "mod_csrf=kept99"
                assert "zos_admin_session" not in (data["cookie"] or "")
                assert data["caller_name"] == "owner"
                assert data["caller_principal"] == str(init["principal"]["id"])

                set_cookies = r_prox.headers.get_list("set-cookie")
                assert any("mod_csrf=good123" in c and "Path=/admin/m/board" in c for c in set_cookies)
                assert not any("zos_admin_session" in c.lower() for c in set_cookies)

                # 5. Root path without trailing slash also renders 200
                r_root = await client.get("/admin/m/board")
                assert r_root.status_code == 200
                assert json.loads(r_root.text)["path"] == "/"

                # 6. Location header rewritten under /admin/m/board
                r_redir = await client.post("/admin/m/board/bounce", follow_redirects=False)
                assert r_redir.status_code == 303
                assert r_redir.headers["location"] == "/admin/m/board/threads/99?tab=all"

                # 7. Disabled module returns 404 and disappears from Overview
                await rt.module_manager.disable("board")
                r_dis = await client.get("/admin/m/board/")
                assert r_dis.status_code == 404
                r_ov2 = await client.get("/admin")
                assert "/admin/m/board/" not in r_ov2.text
    finally:
        await ModuleManager(db_pool, cfg, host=fake_host, modules_dir=modules_dir).storage.purge_storage(
            principal_id=init["principal"]["id"],
            slug="board",
        )
