"""Bearer token verification and ASGI authentication middleware."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool
from starlette.types import ASGIApp, Receive, Scope, Send

from zos.auth.context import ActuatorAuthContext, CallContext, current_call_context
from zos.db.queries import hash_secret


def _normalize_resource(url: str) -> str:
    return url.strip().rstrip("/")


def _resource_matches(token_resource: str | None, base_url: str) -> bool:
    if not token_resource:
        return True
    norm_tok = _normalize_resource(token_resource)
    norm_base = _normalize_resource(base_url)
    return norm_tok in (norm_base, f"{norm_base}/mcp")


async def verify_client_bearer_token(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    raw_token: str,
    *,
    expected_resource: str | None = None,
) -> CallContext | None:
    """Verify a client bearer or OAuth access token and return immutable CallContext."""
    if not raw_token:
        return None
    digest = hash_secret(raw_token)
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT
                t.id AS token_id,
                t.kind AS token_kind,
                t.scopes AS token_scopes,
                t.resource AS token_resource,
                t.expires_at AS token_expires_at,
                t.revoked_at AS token_revoked_at,
                c.id AS client_id,
                c.name AS client_name,
                c.kind AS client_kind,
                c.trust_ceiling AS trust_ceiling,
                c.disabled_at AS client_disabled_at,
                p.id AS principal_id,
                p.slug AS principal_slug,
                s.id AS space_id,
                s.slug AS space_slug
            FROM tokens t
            JOIN clients c ON c.id = t.client_id
            JOIN principals p ON p.id = t.principal_id
            JOIN spaces s ON s.id = c.default_space_id
            WHERE t.token_hash = %s
              AND t.kind IN ('bearer', 'access')
            """,
            (digest,),
        )
        row = await cur.fetchone()
        if row is None:
            return None

        if row["token_revoked_at"] is not None or row["client_disabled_at"] is not None:
            return None

        expires_at = row["token_expires_at"]
        if expires_at is not None and expires_at <= datetime.now(UTC):
            return None

        if expected_resource and not _resource_matches(row["token_resource"], expected_resource):
            return None

        await conn.execute("UPDATE tokens SET last_used_at = now() WHERE id = %s", (row["token_id"],))
        await conn.execute("UPDATE clients SET last_seen_at = now() WHERE id = %s", (row["client_id"],))
        await conn.commit()

    return CallContext(
        principal_id=row["principal_id"],
        principal_slug=row["principal_slug"],
        space_id=row["space_id"],
        space_slug=row["space_slug"],
        client_id=row["client_id"],
        client_name=row["client_name"],
        client_kind=row["client_kind"],
        trust_ceiling=row["trust_ceiling"],
        token_id=row["token_id"],
        scopes=tuple(row["token_scopes"] or ()),
    )


async def verify_actuator_bearer_token(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    raw_token: str,
) -> ActuatorAuthContext | None:
    """Verify an actuator token bound to a provider row."""
    if not raw_token:
        return None
    digest = hash_secret(raw_token)
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT
                t.id AS token_id,
                t.expires_at AS token_expires_at,
                t.revoked_at AS token_revoked_at,
                pr.id AS provider_id,
                pr.slug AS provider_slug,
                pr.principal_id AS principal_id,
                pr.space_id AS space_id,
                pr.enabled AS provider_enabled
            FROM tokens t
            JOIN providers pr ON pr.id = t.provider_id
            WHERE t.token_hash = %s
              AND t.kind = 'actuator'
            """,
            (digest,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        if row["token_revoked_at"] is not None or not row["provider_enabled"]:
            return None
        expires_at = row["token_expires_at"]
        if expires_at is not None and expires_at <= datetime.now(UTC):
            return None

        await conn.execute("UPDATE tokens SET last_used_at = now() WHERE id = %s", (row["token_id"],))
        await conn.execute("UPDATE providers SET last_seen_at = now() WHERE id = %s", (row["provider_id"],))
        await conn.commit()

    return ActuatorAuthContext(
        principal_id=row["principal_id"],
        space_id=row["space_id"],
        provider_id=row["provider_id"],
        provider_slug=row["provider_slug"],
        token_id=row["token_id"],
    )


def build_www_authenticate(base_url: str, *, error: str = "invalid_token", description: str = "Authentication required") -> str:
    """Build RFC 9728 / MCP spec-compliant WWW-Authenticate Bearer header."""
    resource_metadata_url = f"{base_url.rstrip('/')}/.well-known/oauth-protected-resource"
    return (
        f'Bearer error="{error}", '
        f'error_description="{description}", '
        f'resource_metadata="{resource_metadata_url}"'
    )


def extract_bearer_token(headers: list[tuple[bytes, bytes]]) -> str | None:
    for key, value in headers:
        if key.lower() == b"authorization":
            decoded = value.decode("latin-1").strip()
            if decoded.lower().startswith("bearer "):
                token = decoded[7:].strip()
                return token or None
    return None


class BearerAuthASGIMiddleware:
    """ASGI wrapper protecting an MCP endpoint with Bearer / OAuth access token auth."""

    def __init__(
        self,
        app: ASGIApp,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        base_url: str,
    ) -> None:
        self.app = app
        self.pool = pool
        self.base_url = base_url.rstrip("/")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_token = extract_bearer_token(scope.get("headers", []))
        if not raw_token:
            await self._send_401(send, description="Missing bearer token")
            return

        call_ctx = await verify_client_bearer_token(
            self.pool,
            raw_token,
            expected_resource=self.base_url,
        )
        if call_ctx is None:
            await self._send_401(send, description="Invalid, expired, or revoked bearer token")
            return

        scope["zos_call_context"] = call_ctx
        # Also populate Starlette/MCP AuthenticatedUser so StreamableHTTPSessionManager
        # binds stateful MCP sessions to the authenticated client.
        scope["user"] = AuthenticatedUser(
            AccessToken(
                token=str(call_ctx.token_id),
                client_id=str(call_ctx.client_id),
                scopes=list(call_ctx.scopes),
                subject=str(call_ctx.principal_id),
            )
        )

        token_var = current_call_context.set(call_ctx)
        try:
            await self.app(scope, receive, send)
        finally:
            current_call_context.reset(token_var)

    async def _send_401(self, send: Send, *, description: str) -> None:
        header = build_www_authenticate(self.base_url, error="invalid_token", description=description)
        body = json.dumps({"error": "invalid_token", "error_description": description}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"www-authenticate", header.encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
