"""OAuth 2.1 single-user Authorization Server for MCP clients (RFC 8414, 9728, 7591, 7636, 8707, 9207)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode, urlparse

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from zos.audit import record_event
from zos.auth.rate_limit import (
    GlobalPassphraseRateLimiter,
    GlobalSlidingWindowLimiter,
    record_auth_failure,
)
from zos.config import Settings
from zos.db.queries import generate_token, get_principal_and_space, hash_secret
from zos.web.pages import render_authorize_page

AUTH_CODE_TTL_SECONDS = 60
ACCESS_TOKEN_TTL_SECONDS = 3600  # 1 hour
REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 3600  # 30 days

_password_hasher = PasswordHasher()


def _oauth_error(error: str, description: str, *, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        {"error": error, "error_description": description},
        status_code=status_code,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def _resource_is_valid(resource: str | None, base_url: str) -> bool:
    if not resource:
        return True
    norm_res = _normalize_url(resource)
    norm_base = _normalize_url(base_url)
    return norm_res in (norm_base, f"{norm_base}/mcp")


def _verify_pkce_s256(code_verifier: str, expected_challenge: str) -> bool:
    if not (43 <= len(code_verifier) <= 128):
        return False
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return hmac.compare_digest(computed, expected_challenge)


def _append_query(uri: str, params: dict[str, str]) -> str:
    sep = "&" if "?" in uri else "?"
    return f"{uri}{sep}{urlencode(params)}"


def verify_owner_passphrase(passphrase: str, expected_hash: str) -> bool:
    if not passphrase or not expected_hash:
        return False
    try:
        return _password_hasher.verify(expected_hash, passphrase)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def create_oauth_routes(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    settings: Settings,
    *,
    passphrase_limiter: GlobalPassphraseRateLimiter | None = None,
    dcr_limiter: GlobalSlidingWindowLimiter | None = None,
) -> list[Route]:
    """Create discovery, DCR, authorize, and token routes for MCP OAuth 2.1."""
    base_url = _normalize_url(settings.zos_base_url)
    pw_limiter = passphrase_limiter or GlobalPassphraseRateLimiter()
    _dcr_limiter = dcr_limiter  # wired in M7.3

    def _cors_json(data: dict[str, Any], *, status_code: int = 200) -> JSONResponse:
        return JSONResponse(
            data,
            status_code=status_code,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, MCP-Protocol-Version",
                "Cache-Control": "no-store",
            },
        )

    def _cors_preflight(request: Request) -> Response:
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, MCP-Protocol-Version",
            },
        )

    # 2.1 Protected Resource Metadata (RFC 9728)
    async def protected_resource_metadata(request: Request) -> Response:
        if request.method == "OPTIONS":
            return _cors_preflight(request)
        return _cors_json(
            {
                "resource": base_url,
                "authorization_servers": [base_url],
                "scopes_supported": ["mcp"],
                "bearer_methods_supported": ["header"],
            }
        )

    # 2.1 Authorization Server Metadata (RFC 8414)
    async def authorization_server_metadata(request: Request) -> Response:
        if request.method == "OPTIONS":
            return _cors_preflight(request)
        return _cors_json(
            {
                "issuer": base_url,
                "authorization_endpoint": f"{base_url}/oauth/authorize",
                "token_endpoint": f"{base_url}/oauth/token",
                "registration_endpoint": f"{base_url}/oauth/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none"],
                "code_challenge_methods_supported": ["S256"],
                "scopes_supported": ["mcp"],
                "authorization_response_iss_parameter_supported": True,
            }
        )

    # 2.2 Dynamic Client Registration (RFC 7591)
    async def register_client(request: Request) -> Response:
        if request.method == "OPTIONS":
            return _cors_preflight(request)
        try:
            body = await request.json()
        except Exception:
            return _oauth_error("invalid_client_metadata", "Request body must be valid JSON")

        if not isinstance(body, dict):
            return _oauth_error("invalid_client_metadata", "Request body must be a JSON object")

        raw_uris = body.get("redirect_uris")
        if not isinstance(raw_uris, list) or not raw_uris or not all(isinstance(u, str) and u.strip() for u in raw_uris):
            return _oauth_error("invalid_redirect_uri", "redirect_uris must be a non-empty list of URIs")

        redirect_uris: list[str] = []
        for uri in raw_uris:
            parsed = urlparse(uri)
            if parsed.scheme not in ("http", "https") and not parsed.scheme:
                return _oauth_error("invalid_redirect_uri", f"Invalid redirect_uri: {uri}")
            if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
                return _oauth_error("invalid_redirect_uri", "Non-loopback redirect_uri must use HTTPS")
            redirect_uris.append(uri)

        auth_method = body.get("token_endpoint_auth_method", "none")
        if auth_method != "none":
            return _oauth_error(
                "invalid_client_metadata",
                "Only public clients with token_endpoint_auth_method='none' are supported",
            )

        client_name = str(body.get("client_name") or "").strip()
        oauth_client_id = f"zos_c_{secrets.token_urlsafe(16)}"
        if not client_name:
            client_name = oauth_client_id

        try:
            principal, space = await get_principal_and_space(pool)
        except ValueError as exc:
            return _oauth_error("server_error", str(exc), status_code=503)

        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO clients (
                    principal_id, default_space_id, name, kind, oauth_client_id, redirect_uris
                )
                VALUES (%s, %s, %s, 'oauth', %s, %s)
                RETURNING id
                """,
                (principal["id"], space["id"], client_name, oauth_client_id, redirect_uris),
            )
            row = await cur.fetchone()
            await conn.commit()
        assert row is not None

        await record_event(
            pool,
            principal_id=principal["id"],
            space_id=space["id"],
            client_id=row["id"],
            event_type="client.registered",
            payload={"name": client_name, "kind": "oauth", "oauth_client_id": oauth_client_id},
        )

        issued_at = int(time.time())
        return _cors_json(
            {
                "client_id": oauth_client_id,
                "client_id_issued_at": issued_at,
                "client_name": client_name,
                "redirect_uris": redirect_uris,
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            status_code=201,
        )

    async def _load_oauth_client(oauth_client_id: str) -> dict[str, Any] | None:
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, principal_id, default_space_id, name, oauth_client_id, redirect_uris
                FROM clients
                WHERE oauth_client_id = %s AND kind = 'oauth'
                """,
                (oauth_client_id,),
            )
            row = await cur.fetchone()
        return dict(row) if row else None

    # 2.3 Authorization Endpoint
    async def authorize(request: Request) -> Response:
        if request.method == "GET":
            params = dict(request.query_params)
        else:
            form = await request.form()
            params = {k: str(v) for k, v in form.items()}

        response_type = params.get("response_type", "")
        oauth_client_id = params.get("client_id", "")
        redirect_uri = params.get("redirect_uri", "")
        state = params.get("state", "")
        scope_str = params.get("scope", "mcp").strip() or "mcp"
        scopes = [s for s in scope_str.split() if s]
        code_challenge = params.get("code_challenge", "")
        code_challenge_method = params.get("code_challenge_method", "")
        resource = params.get("resource", "").strip()

        client = await _load_oauth_client(oauth_client_id)
        if client is None:
            return HTMLResponse("<h1>Invalid client_id</h1>", status_code=400)
        if not redirect_uri or redirect_uri not in (client["redirect_uris"] or []):
            return HTMLResponse("<h1>Invalid redirect_uri</h1>", status_code=400)

        def _redirect_error(code: str, desc: str) -> RedirectResponse:
            qp = {"error": code, "error_description": desc, "iss": base_url}
            if state:
                qp["state"] = state
            return RedirectResponse(_append_query(redirect_uri, qp), status_code=302)

        if response_type != "code":
            return _redirect_error("unsupported_response_type", "response_type must be 'code'")
        if code_challenge_method != "S256" or not code_challenge:
            return _redirect_error("invalid_request", "PKCE S256 code_challenge is required")
        if resource and not _resource_is_valid(resource, base_url):
            return _redirect_error("invalid_target", "Requested resource does not match this ZOS server")

        hidden = {
            "response_type": response_type,
            "client_id": oauth_client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
        }
        if state:
            hidden["state"] = state
        if resource:
            hidden["resource"] = resource

        html_headers = {
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "frame-ancestors 'none'",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        }

        if request.method == "GET":
            html_body = render_authorize_page(
                client_name=client["name"],
                redirect_uri=redirect_uri,
                scopes=scopes,
                hidden_fields=hidden,
            )
            return HTMLResponse(html_body, headers=html_headers)

        # POST submission
        decision = params.get("decision", "approve")
        if decision == "deny":
            return _redirect_error("access_denied", "Owner denied authorization")

        retry_after = pw_limiter.retry_after_seconds()
        if retry_after is not None:
            html_body = render_authorize_page(
                client_name=client["name"],
                redirect_uri=redirect_uri,
                scopes=scopes,
                hidden_fields=hidden,
                error=f"Too many failed attempts. Try again in {retry_after}s.",
            )
            return HTMLResponse(
                html_body,
                status_code=429,
                headers={**html_headers, "Retry-After": str(retry_after)},
            )

        if not settings.zos_owner_password_hash:
            return HTMLResponse(
                "<h1>Owner passphrase not configured (ZOS_OWNER_PASSWORD_HASH)</h1>",
                status_code=503,
                headers=html_headers,
            )

        passphrase = params.get("passphrase", "")
        if not verify_owner_passphrase(passphrase, settings.zos_owner_password_hash):
            pw_limiter.record_failure()
            await record_auth_failure(
                pool,
                route="/oauth/authorize",
                reason="invalid_passphrase",
                client_id=client["id"],
            )
            html_body = render_authorize_page(
                client_name=client["name"],
                redirect_uri=redirect_uri,
                scopes=scopes,
                hidden_fields=hidden,
                error="Invalid owner passphrase.",
            )
            return HTMLResponse(html_body, status_code=401, headers=html_headers)

        pw_limiter.record_success()

        raw_code = secrets.token_urlsafe(32)
        code_hash = hash_secret(raw_code)
        expires_at = datetime.now(UTC) + timedelta(seconds=AUTH_CODE_TTL_SECONDS)
        bound_resource = _normalize_url(resource) if resource else base_url

        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO oauth_codes (
                    code_hash, client_id, code_challenge, redirect_uri, scopes, resource, expires_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    code_hash,
                    client["id"],
                    code_challenge,
                    redirect_uri,
                    scopes,
                    bound_resource,
                    expires_at,
                ),
            )
            await conn.commit()

        await record_event(
            pool,
            principal_id=client["principal_id"],
            space_id=client["default_space_id"],
            client_id=client["id"],
            event_type="client.authorized",
            payload={"scopes": scopes, "resource": bound_resource},
        )

        qp = {"code": raw_code, "iss": base_url}
        if state:
            qp["state"] = state
        return RedirectResponse(_append_query(redirect_uri, qp), status_code=302)

    # 2.4 Token Endpoint (authorization_code + refresh_token rotation & reuse revocation)
    async def token_endpoint(request: Request) -> Response:
        if request.method == "OPTIONS":
            return _cors_preflight(request)

        form = await request.form()
        data = {k: str(v) for k, v in form.items()}
        grant_type = data.get("grant_type", "")

        if grant_type == "authorization_code":
            return await _handle_authorization_code_grant(data)
        if grant_type == "refresh_token":
            return await _handle_refresh_token_grant(data)
        return _oauth_error("unsupported_grant_type", f"Unsupported grant_type: {grant_type}")

    async def _issue_token_pair(
        *,
        principal_id: Any,
        space_id: Any,
        client_id: Any,
        scopes: list[str],
        resource: str,
        family_id: uuid.UUID,
    ) -> JSONResponse:
        now = datetime.now(UTC)
        access_raw, access_hash = generate_token("zos_at")
        refresh_raw, refresh_hash = generate_token("zos_rt")
        access_exp = now + timedelta(seconds=ACCESS_TOKEN_TTL_SECONDS)
        refresh_exp = now + timedelta(seconds=REFRESH_TOKEN_TTL_SECONDS)

        async with pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                INSERT INTO tokens (
                    principal_id, client_id, kind, token_hash, family_id, scopes, resource, expires_at
                )
                VALUES (%s, %s, 'access', %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (principal_id, client_id, access_hash, family_id, scopes, resource, access_exp),
            )
            at_row = await cur.fetchone()
            cur = await conn.execute(
                """
                INSERT INTO tokens (
                    principal_id, client_id, kind, token_hash, family_id, scopes, resource, expires_at
                )
                VALUES (%s, %s, 'refresh', %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (principal_id, client_id, refresh_hash, family_id, scopes, resource, refresh_exp),
            )
            rt_row = await cur.fetchone()
        assert at_row is not None and rt_row is not None

        await record_event(
            pool,
            principal_id=principal_id,
            space_id=space_id,
            client_id=client_id,
            event_type="token.issued",
            payload={
                "access_token_id": str(at_row["id"]),
                "refresh_token_id": str(rt_row["id"]),
                "family_id": str(family_id),
                "resource": resource,
            },
        )

        return _cors_json(
            {
                "access_token": access_raw,
                "token_type": "Bearer",
                "expires_in": ACCESS_TOKEN_TTL_SECONDS,
                "refresh_token": refresh_raw,
                "scope": " ".join(scopes),
            }
        )

    async def _handle_authorization_code_grant(data: dict[str, str]) -> Response:
        code = data.get("code", "")
        redirect_uri = data.get("redirect_uri", "")
        oauth_client_id = data.get("client_id", "")
        code_verifier = data.get("code_verifier", "")
        resource = data.get("resource", "").strip()

        if not code or not oauth_client_id or not code_verifier or not redirect_uri:
            return _oauth_error("invalid_request", "Missing required token exchange parameters")

        client = await _load_oauth_client(oauth_client_id)
        if client is None:
            return _oauth_error("invalid_client", "Unknown client_id", status_code=401)

        code_hash = hash_secret(code)
        async with pool.connection() as conn:
            # Atomic single-use claim of valid non-expired code
            cur = await conn.execute(
                """
                UPDATE oauth_codes
                SET used_at = now()
                WHERE code_hash = %s
                  AND client_id = %s
                  AND used_at IS NULL
                  AND expires_at > now()
                RETURNING code_challenge, redirect_uri, scopes, resource
                """,
                (code_hash, client["id"]),
            )
            code_row = await cur.fetchone()
            await conn.commit()

        if code_row is None:
            return _oauth_error("invalid_grant", "Authorization code is invalid, expired, or already used")

        if code_row["redirect_uri"] != redirect_uri:
            return _oauth_error("invalid_grant", "redirect_uri mismatch")

        if not _verify_pkce_s256(code_verifier, code_row["code_challenge"]):
            return _oauth_error("invalid_grant", "Invalid PKCE code_verifier")

        bound_resource = code_row["resource"] or base_url
        if resource and not _resource_is_valid(resource, base_url):
            return _oauth_error("invalid_target", "Requested resource does not match this ZOS server")

        return await _issue_token_pair(
            principal_id=client["principal_id"],
            space_id=client["default_space_id"],
            client_id=client["id"],
            scopes=list(code_row["scopes"] or ["mcp"]),
            resource=_normalize_url(resource) if resource else bound_resource,
            family_id=uuid.uuid4(),
        )

    async def _handle_refresh_token_grant(data: dict[str, str]) -> Response:
        refresh_token = data.get("refresh_token", "")
        oauth_client_id = data.get("client_id", "")
        resource = data.get("resource", "").strip()

        if not refresh_token or not oauth_client_id:
            return _oauth_error("invalid_request", "Missing refresh_token or client_id")

        client = await _load_oauth_client(oauth_client_id)
        if client is None:
            return _oauth_error("invalid_client", "Unknown client_id", status_code=401)

        if resource and not _resource_is_valid(resource, base_url):
            return _oauth_error("invalid_target", "Requested resource does not match this ZOS server")

        rt_hash = hash_secret(refresh_token)
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, family_id, scopes, resource, expires_at, rotated_at, revoked_at
                FROM tokens
                WHERE token_hash = %s
                  AND client_id = %s
                  AND kind = 'refresh'
                """,
                (rt_hash, client["id"]),
            )
            row = await cur.fetchone()

            if row is None:
                return _oauth_error("invalid_grant", "Invalid refresh token")

            # Reuse detection: presenting an already-rotated refresh token revokes the ENTIRE family!
            if row["rotated_at"] is not None:
                family_id = row["family_id"]
                if family_id is not None:
                    await conn.execute(
                        """
                        UPDATE tokens SET revoked_at = now()
                        WHERE family_id = %s AND revoked_at IS NULL
                        """,
                        (family_id,),
                    )
                    await conn.commit()
                await record_event(
                    pool,
                    principal_id=client["principal_id"],
                    space_id=client["default_space_id"],
                    client_id=client["id"],
                    event_type="token.revoked",
                    outcome="denied",
                    payload={"reason": "refresh_token_reuse", "family_id": str(family_id)},
                )
                return _oauth_error("invalid_grant", "Refresh token reuse detected; token family revoked")

            if row["revoked_at"] is not None:
                return _oauth_error("invalid_grant", "Refresh token has been revoked")

            if row["expires_at"] is not None and row["expires_at"] <= datetime.now(UTC):
                return _oauth_error("invalid_grant", "Refresh token has expired")

            # Atomically mark this refresh token rotated
            cur = await conn.execute(
                """
                UPDATE tokens
                SET rotated_at = now(), last_used_at = now()
                WHERE id = %s AND rotated_at IS NULL AND revoked_at IS NULL
                RETURNING id
                """,
                (row["id"],),
            )
            claimed = await cur.fetchone()
            await conn.commit()

        if claimed is None:
            return _oauth_error("invalid_grant", "Refresh token already rotated")

        family_id = row["family_id"] or uuid.uuid4()
        bound_resource = _normalize_url(resource) if resource else (row["resource"] or base_url)
        return await _issue_token_pair(
            principal_id=client["principal_id"],
            space_id=client["default_space_id"],
            client_id=client["id"],
            scopes=list(row["scopes"] or ["mcp"]),
            resource=bound_resource,
            family_id=family_id,
        )

    return [
        Route(
            "/.well-known/oauth-protected-resource",
            protected_resource_metadata,
            methods=["GET", "OPTIONS"],
        ),
        Route(
            "/.well-known/oauth-protected-resource/mcp",
            protected_resource_metadata,
            methods=["GET", "OPTIONS"],
        ),
        Route(
            "/.well-known/oauth-authorization-server",
            authorization_server_metadata,
            methods=["GET", "OPTIONS"],
        ),
        Route("/oauth/register", register_client, methods=["POST", "OPTIONS"]),
        Route("/oauth/authorize", authorize, methods=["GET", "POST"]),
        Route("/oauth/token", token_endpoint, methods=["POST", "OPTIONS"]),
    ]
