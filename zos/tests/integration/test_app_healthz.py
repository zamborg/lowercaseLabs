"""Integration smoke test for M1.9 ASGI app (/healthz and /mcp auth gate)."""

from __future__ import annotations

import httpx

from tests.conftest import get_test_database_url
from zos.app import create_app
from zos.config import Settings


async def test_app_healthz_and_mcp_401(db_pool) -> None:
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8080",
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8080") as client:
            r = await client.get("/healthz")
            assert r.status_code == 200
            body = r.json()
            assert body["ok"] is True
            assert any(p["namespace"] == "memory" and p["online"] for p in body["providers"])

            r_mcp = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
            assert r_mcp.status_code == 401
            assert 'resource_metadata="http://localhost:8080/.well-known/oauth-protected-resource"' in (
                r_mcp.headers["www-authenticate"]
            )
