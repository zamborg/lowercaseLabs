"""Integration tests for M1.3 bearer authentication middleware and CallContext."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from zos.auth import BearerAuthASGIMiddleware, CallContext, current_call_context
from zos.db.queries import create_bearer_client, init_principal_and_space, revoke_token


def _make_protected_app(db_pool, base_url: str = "http://localhost:8080") -> Starlette:
    async def echo_ctx(request: Request) -> JSONResponse:
        ctx: CallContext = request.scope["zos_call_context"]
        assert current_call_context.get() == ctx
        return JSONResponse(
            {
                "principal": ctx.principal_slug,
                "space": ctx.space_slug,
                "client": ctx.client_name,
                "trust_ceiling": ctx.trust_ceiling,
            }
        )

    inner = Starlette(routes=[Route("/mcp", echo_ctx)])
    return BearerAuthASGIMiddleware(inner, db_pool, base_url)  # type: ignore[return-value]


async def test_bearer_middleware_valid_revoked_and_expired(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    created = await create_bearer_client(
        db_pool, principal_id=p_id, space_id=s_id, name="codex", trust_ceiling="owner"
    )
    raw_token = created["token"]
    token_id = created["token_id"]

    app = _make_protected_app(db_pool, base_url="http://localhost:8080")
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8080") as client:
        # Missing header -> 401 + spec-compliant WWW-Authenticate
        r_missing = await client.get("/mcp")
        assert r_missing.status_code == 401
        www = r_missing.headers["www-authenticate"]
        assert 'resource_metadata="http://localhost:8080/.well-known/oauth-protected-resource"' in www

        # Wrong token -> 401
        r_bad = await client.get("/mcp", headers={"Authorization": "Bearer zos_b_wrong"})
        assert r_bad.status_code == 401
        assert "resource_metadata=" in r_bad.headers["www-authenticate"]

        # Valid token -> 200 with immutable CallContext
        r_ok = await client.get("/mcp", headers={"Authorization": f"Bearer {raw_token}"})
        assert r_ok.status_code == 200
        assert r_ok.json() == {
            "principal": "zubin",
            "space": "main",
            "client": "codex",
            "trust_ceiling": "owner",
        }

        # Expired token -> 401
        async with db_pool.connection() as conn:
            await conn.execute(
                "UPDATE tokens SET expires_at = %s WHERE id = %s",
                (datetime.now(UTC) - timedelta(minutes=1), token_id),
            )
            await conn.commit()

        r_exp = await client.get("/mcp", headers={"Authorization": f"Bearer {raw_token}"})
        assert r_exp.status_code == 401
        assert "resource_metadata=" in r_exp.headers["www-authenticate"]

        # Clear expiry, then revoke -> 401
        async with db_pool.connection() as conn:
            await conn.execute("UPDATE tokens SET expires_at = NULL WHERE id = %s", (token_id,))
            await conn.commit()

        assert await revoke_token(db_pool, token_id=token_id) == 1
        r_rev = await client.get("/mcp", headers={"Authorization": f"Bearer {raw_token}"})
        assert r_rev.status_code == 401
        assert "resource_metadata=" in r_rev.headers["www-authenticate"]
