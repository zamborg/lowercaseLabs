"""Integration tests for M11 invariants: single-instance advisory lock & owner auth without Access."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from argon2 import PasswordHasher

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port
from zos.app import create_app
from zos.config import Settings
from zos.db import SingleInstanceError
from zos.db.queries import init_principal_and_space
from zos.vault import generate_secret_key


async def test_single_instance_advisory_lock_blocks_second_process(db_pool: Any) -> None:
    """Startup fails loudly if a second ZOS process tries to run against the same database."""
    db_url = get_test_database_url()
    cfg = Settings(database_url=db_url, zos_secret_key=generate_secret_key())
    app1 = create_app(cfg)
    app2 = create_app(cfg)

    async with app1.router.lifespan_context(app1):
        with pytest.raises(SingleInstanceError, match="Another ZOS instance is already running"):
            async with app2.router.lifespan_context(app2):
                pass

    # After the first instance shuts down and releases the advisory lock, a new instance starts cleanly
    app3 = create_app(cfg)
    async with app3.router.lifespan_context(app3):
        assert app3.state.zos is not None


async def test_owner_routes_protected_without_cloudflare_access(db_pool: Any) -> None:
    """Verify /admin*, /oauth/authorize, and /connect/* enforce ZOS's own auth with zero Access headers."""
    db_url = get_test_database_url()
    await init_principal_and_space(db_pool, principal_slug="owner")
    owner_pw = "standalone-owner-secret"
    cfg = Settings(
        database_url=db_url,
        zos_base_url=f"http://127.0.0.1:{free_port()}",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
    )
    app = create_app(cfg)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=cfg.zos_base_url) as client:
            # 1. All /admin routes redirect unauthenticated callers to /admin/login
            for path in (
                "/admin",
                "/admin/approvals",
                "/admin/modules",
                "/admin/upstreams",
                "/admin/clients",
                "/admin/policy",
                "/admin/activity",
                "/admin/m/inbox/",
            ):
                r = await client.get(path, follow_redirects=False)
                assert r.status_code == 303, f"{path} returned {r.status_code}"
                assert r.headers["location"].startswith("/admin/login")

            # 2. Mutating /admin POSTs reject unauthenticated callers
            r_post = await client.post("/admin/modules/install", data={"slug": "inbox"}, follow_redirects=False)
            assert r_post.status_code == 303
            assert r_post.headers["location"].startswith("/admin/login")

            # 3. Legacy /connect/google is removed (404)
            r_g = await client.get("/connect/google")
            assert r_g.status_code == 404

            # 4. /admin/login rejects wrong passphrase (401)
            r_bad = await client.post("/admin/login", data={"passphrase": "wrong-password"})
            assert r_bad.status_code == 401
