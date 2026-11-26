"""Integration test for M11 (3-4): module upgrade preserves data, secret, upstream_id, and overrides."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
import psycopg
from argon2 import PasswordHasher

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port
from zos.admin_service import AdminService
from zos.app import create_app
from zos.catalog import ModuleManifest
from zos.config import Settings
from zos.db.queries import init_principal_and_space
from zos.modules import InstalledModule, ModuleManager, ModuleRuntimeStatus
from zos.vault import generate_secret_key

DIGEST_V1 = "1" * 64
DIGEST_V2 = "2" * 64


class _UpgradeFakeHost:
    kind = "docker"

    def __init__(self) -> None:
        self.install_envs: list[dict[str, str]] = []
        self.upgrade_versions: list[str] = []

    async def install(
        self, manifest: ModuleManifest, config: dict[str, str], *, env: dict[str, str]
    ) -> InstalledModule:
        self.install_envs.append(dict(env))
        return InstalledModule(
            mcp_url="http://127.0.0.1:19991/mcp",
            ui_url="http://127.0.0.1:19992",
            image_pin=manifest.runtime.image or "sha256:v1",
            host="docker",
        )

    async def upgrade(
        self, slug: str, manifest: ModuleManifest, config: dict[str, str], *, env: dict[str, str]
    ) -> InstalledModule:
        self.upgrade_versions.append(manifest.version)
        return await self.install(manifest, config, env=env)

    async def stop(self, slug: str) -> None:
        return None

    async def start(self, slug: str) -> None:
        return None

    async def uninstall(self, slug: str, *, purge: bool = False) -> None:
        return None

    async def status(self, slug: str) -> ModuleRuntimeStatus:
        return ModuleRuntimeStatus(host="docker", state="running")


def _write_inbox_manifest(modules_dir: Path, *, version: str, digest: str) -> None:
    modules_dir.mkdir(parents=True, exist_ok=True)
    (modules_dir / "inbox.toml").write_text(
        f"""
manifest_version = 2
slug = "inbox"
name = "Inbox"
version = "{version}"
description = "Internal agent mailboxes"

[runtime]
kind = "container"
image = "ghcr.io/example/zos-inbox@sha256:{digest}"
mcp_port = 8081
mcp_path = "/mcp"
ui_port = 8082

[storage]
postgres = true

[network]
egress = false

[ui]
title = "Inbox"

[tiers]
default = "write"
"inbox_list" = "read"
"inbox_read" = "read"
"inbox_directory" = "read"
"inbox_send" = "write"
"inbox_archive" = "write"
""",
        encoding="utf-8",
    )


async def test_module_upgrade_preserves_data_secret_overrides_and_audits(
    db_pool: Any, tmp_path: Path
) -> None:
    db_url = get_test_database_url()
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    principal_id = init["principal"]["id"]
    svc = AdminService(db_pool)
    reg = await svc.register_bearer_client(name="upgrade-agent", trust_ceiling="owner")
    client_id = reg["client"]["id"]

    modules_dir = tmp_path / "catalog_modules"
    _write_inbox_manifest(modules_dir, version="0.1.0", digest=DIGEST_V1)

    owner_pw = "owner-upgrade-pass"
    cfg = Settings(
        database_url=db_url,
        zos_base_url=f"http://127.0.0.1:{free_port()}",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
        modules_catalog_dir=modules_dir,
    )
    fake_host = _UpgradeFakeHost()
    app = create_app(cfg, module_host=fake_host)

    try:
        async with app.router.lifespan_context(app):
            rt = app.state.zos
            inst = await rt.module_manager.install("inbox", wait_ready=False)
            orig_upstream_id = inst["module"]["upstream_id"]
            orig_secret = fake_host.install_envs[0]["ZOS_MODULE_SECRET"]
            orig_dsn = fake_host.install_envs[0]["ZOS_MODULE_DATABASE_URL"]

            # 1. Manifest tiers seeded immediately on install (read for list/read/directory, write for send/archive)
            async with rt.pool.connection() as conn:
                tiers = {
                    r["tool_pattern"]: r["tier"]
                    for r in await (
                        await conn.execute(
                            "SELECT tool_pattern, tier FROM tool_tiers WHERE principal_id = %s AND tool_pattern LIKE 'inbox%%'",
                            (principal_id,),
                        )
                    ).fetchall()
                }
            assert tiers["inbox_list"] == "read"
            assert tiers["inbox_read"] == "read"
            assert tiers["inbox_directory"] == "read"
            assert tiers["inbox_send"] == "write"
            assert tiers["inbox_archive"] == "write"

            # 2. Write data into mod_inbox schema
            async with await psycopg.AsyncConnection.connect(orig_dsn, autocommit=True) as mod_conn:
                await mod_conn.execute("CREATE TABLE notes (id INT PRIMARY KEY, body TEXT)")
                await mod_conn.execute("INSERT INTO notes (id, body) VALUES (1, 'survives upgrade')")

            # 3. Owner edits a tier override and a client policy override
            await svc.set_tool_tier(pattern="inbox_send", tier="external")
            await svc.set_policy(pattern="inbox_archive", decision="deny", client_identifier="upgrade-agent")

            # 4. Bump catalog manifest to v0.2.0 and click Upgrade in /admin/modules
            _write_inbox_manifest(modules_dir, version="0.2.0", digest=DIGEST_V2)

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=cfg.zos_base_url) as web:
                await web.post("/admin/login", data={"passphrase": owner_pw}, follow_redirects=False)
                r_page = await web.get("/admin/modules")
                assert r_page.status_code == 200
                csrf = re.search(r'name="csrf_token"\s+value="([^"]+)"', r_page.text).group(1)

                r_up = await web.post(
                    "/admin/modules/upgrade",
                    data={"csrf_token": csrf, "slug": "inbox"},
                )
                assert r_up.status_code == 200
                assert "Upgraded module" in r_up.text
                assert "New tools appear in clients after they refresh/reconnect their ZOS connector." in r_up.text

            # 5. Verify secret, upstream_id, data, owner overrides, and module.upgraded audit all survived!
            assert fake_host.upgrade_versions == ["0.2.0"]
            assert fake_host.install_envs[-1]["ZOS_MODULE_SECRET"] == orig_secret

            mod_after = await rt.module_manager.get_module("inbox")
            assert mod_after is not None
            assert mod_after["version"] == "0.2.0"
            assert mod_after["image_pin"] == f"ghcr.io/example/zos-inbox@sha256:{DIGEST_V2}"
            assert mod_after["upstream_id"] == orig_upstream_id

            async with await psycopg.AsyncConnection.connect(orig_dsn, autocommit=True) as mod_conn:
                cur = await mod_conn.execute("SELECT body FROM notes WHERE id = 1")
                assert await cur.fetchone() == ("survives upgrade",)

            async with rt.pool.connection() as conn:
                t_send = await (
                    await conn.execute(
                        "SELECT tier FROM tool_tiers WHERE principal_id = %s AND tool_pattern = 'inbox_send'",
                        (principal_id,),
                    )
                ).fetchone()
                assert t_send is not None and t_send["tier"] == "external"

                p_arch = await (
                    await conn.execute(
                        "SELECT decision FROM client_policies WHERE principal_id = %s AND client_id = %s AND tool_pattern = 'inbox_archive'",
                        (principal_id, client_id),
                    )
                ).fetchone()
                assert p_arch is not None and p_arch["decision"] == "deny"

                ev = await (
                    await conn.execute(
                        "SELECT payload FROM events WHERE type = 'module.upgraded' ORDER BY id DESC LIMIT 1"
                    )
                ).fetchone()
                assert ev is not None
                assert ev["payload"]["slug"] == "inbox"
                assert ev["payload"]["version"] == "0.2.0"
    finally:
        await ModuleManager(db_pool, cfg, host=fake_host, modules_dir=modules_dir).storage.purge_storage(
            principal_id=principal_id,
            slug="inbox",
        )
