"""Shared Google OAuth connect flow, credential lifecycle, and HTTP API client with 429 backoff."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import httpx
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from zos.audit import record_event
from zos.auth.oauth import verify_owner_passphrase
from zos.config import Settings
from zos.db.queries import get_principal_and_space
from zos.router import SessionTracker
from zos.vault import CredentialRevokedError, CredentialStore, CredentialVault
from zos.web.pages import render_google_connect_page

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_PROFILE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/profile"

GOOGLE_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
)

STATE_TTL_SECONDS = 600


class GoogleAPIError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class GoogleServiceHub:
    """Owns the shared Google credential used by both `gcal` and `gmail` providers."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        settings: Settings,
        sessions: SessionTracker,
        *,
        backoff_base_s: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        self.pool = pool
        self.settings = settings
        self.sessions = sessions
        self.backoff_base_s = backoff_base_s
        self.max_retries = max_retries

        self._store: CredentialStore | None = (
            CredentialStore(pool, CredentialVault(settings.zos_secret_key))
            if settings.zos_secret_key
            else None
        )
        self._credential_id: UUID | None = None
        self._principal_id: UUID | None = None
        self._space_id: UUID | None = None
        self._gcal_provider_id: UUID | None = None
        self._gmail_provider_id: UUID | None = None
        self._pending_states: dict[str, float] = {}

    @property
    def is_connected(self) -> bool:
        return self._store is not None and self._credential_id is not None

    @property
    def gcal_provider_id(self) -> UUID | None:
        return self._gcal_provider_id

    @property
    def gmail_provider_id(self) -> UUID | None:
        return self._gmail_provider_id

    async def load_existing_connection(self) -> None:
        """Load persisted Google credential and gcal/gmail provider IDs on startup."""
        if self._store is None:
            return
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, principal_id
                FROM credentials
                WHERE issuer = 'google'
                ORDER BY updated_at DESC
                LIMIT 1
                """
            )
            cred = await cur.fetchone()
            if cred is None:
                return
            self._credential_id = cred["id"]
            self._principal_id = cred["principal_id"]

            cur = await conn.execute(
                "SELECT id FROM spaces WHERE principal_id = %s ORDER BY created_at ASC LIMIT 1",
                (self._principal_id,),
            )
            sp = await cur.fetchone()
            if sp is not None:
                self._space_id = sp["id"]

            cur = await conn.execute(
                """
                SELECT id, slug FROM providers
                WHERE principal_id = %s AND slug IN ('gcal', 'gmail') AND enabled = true
                """,
                (self._principal_id,),
            )
            for row in await cur.fetchall():
                if row["slug"] == "gcal":
                    self._gcal_provider_id = row["id"]
                elif row["slug"] == "gmail":
                    self._gmail_provider_id = row["id"]

    async def attach_google_tokens(
        self,
        *,
        account: str,
        scopes: list[str],
        access_token: str,
        refresh_token: str,
        expires_in: int,
    ) -> UUID:
        """Persist encrypted Google credential + gcal & gmail provider rows, bring online, broadcast."""
        if self._store is None:
            raise RuntimeError("ZOS_SECRET_KEY is not configured; cannot store Google credentials")

        principal, space = await get_principal_and_space(self.pool)
        p_id: UUID = principal["id"]
        s_id: UUID = space["id"]

        cred_id = await self._store.save_google_credential(
            principal_id=p_id,
            account=account,
            scopes=scopes,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
        )

        async with self.pool.connection() as conn, conn.transaction():
            ids: dict[str, UUID] = {}
            for slug in ("gcal", "gmail"):
                cur = await conn.execute(
                    """
                    INSERT INTO providers (
                        principal_id, space_id, slug, kind, credential_id, config, enabled, last_seen_at
                    )
                    VALUES (%s, %s, %s, 'upstream', %s, %s, true, now())
                    ON CONFLICT (principal_id, slug) DO UPDATE
                    SET credential_id = EXCLUDED.credential_id,
                        enabled = true,
                        last_seen_at = now()
                    RETURNING id
                    """,
                    (p_id, s_id, slug, cred_id, Jsonb({})),
                )
                row = await cur.fetchone()
                assert row is not None
                ids[slug] = row["id"]

        self._credential_id = cred_id
        self._principal_id = p_id
        self._space_id = s_id
        self._gcal_provider_id = ids["gcal"]
        self._gmail_provider_id = ids["gmail"]

        await record_event(
            self.pool,
            principal_id=p_id,
            space_id=s_id,
            event_type="credential.connected",
            payload={"issuer": "google", "credential_id": str(cred_id), "scopes": scopes},
        )
        for slug in ("gcal", "gmail"):
            await record_event(
                self.pool,
                principal_id=p_id,
                space_id=s_id,
                provider_id=ids[slug],
                event_type="provider.connected",
                payload={"slug": slug, "issuer": "google"},
            )
        await self.sessions.broadcast_tools_changed()
        return cred_id

    async def mark_revoked(self, reason: str = "invalid_grant") -> None:
        """Mark Google credential offline, delete revoked credential row, and emit list_changed."""
        cred_id = self._credential_id
        p_id = self._principal_id
        s_id = self._space_id
        self._credential_id = None

        if cred_id is not None and self._store is not None:
            await self._store.delete_credential(cred_id)
        if p_id is not None and s_id is not None:
            await record_event(
                self.pool,
                principal_id=p_id,
                space_id=s_id,
                event_type="provider.disconnected",
                outcome="error",
                payload={"issuer": "google", "reason": reason},
            )
        await self.sessions.broadcast_tools_changed()

    async def _refresh_google_token(self, refresh_token: str) -> tuple[str, int, str | None]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": self.settings.google_client_id,
                    "client_secret": self.settings.google_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
        if resp.status_code in (400, 401):
            err_code = ""
            with contextlib.suppress(Exception):
                err_code = str(resp.json().get("error", ""))
            if err_code in ("invalid_grant", "unauthorized_client"):
                await self.mark_revoked(err_code)
                raise CredentialRevokedError(f"Google refresh token revoked ({err_code})")
        if resp.status_code != 200:
            raise GoogleAPIError(resp.status_code, f"Google token refresh failed ({resp.status_code})")
        data = resp.json()
        return (
            str(data["access_token"]),
            int(data.get("expires_in", 3600)),
            data.get("refresh_token"),
        )

    async def api_request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | list[tuple[str, str]] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an authenticated Google REST request with refresh-on-401 and 429 exponential backoff."""
        if not self.is_connected or self._store is None or self._credential_id is None:
            raise CredentialRevokedError("Google is not connected. Visit /connect/google to connect.")

        cred_id = self._credential_id
        space_id = self._space_id
        if space_id is None:
            _, sp = await get_principal_and_space(self.pool)
            space_id = sp["id"]
            self._space_id = space_id

        access_token = await self._store.get_valid_access_token(
            cred_id,
            space_id=space_id,
            refresh_fn=self._refresh_google_token,
        )

        refreshed_after_401 = False
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(self.max_retries + 1):
                resp = await client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if resp.status_code == 401 and not refreshed_after_401:
                    refreshed_after_401 = True
                    access_token = await self._store.get_valid_access_token(
                        cred_id,
                        space_id=space_id,
                        refresh_fn=self._refresh_google_token,
                        force_refresh=True,
                    )
                    continue

                if resp.status_code in (429, 503) and attempt < self.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = min(float(retry_after), 10.0)
                    else:
                        delay = self.backoff_base_s * (2**attempt)
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code == 204:
                    return {}
                if 200 <= resp.status_code < 300:
                    return resp.json() if resp.content else {}

                # Extract safe error message without leaking tokens
                err_msg = f"Google API {method} {url} failed with status {resp.status_code}"
                try:
                    err_json = resp.json()
                    if isinstance(err_json, dict) and "error" in err_json:
                        detail = err_json["error"]
                        if isinstance(detail, dict) and detail.get("message"):
                            err_msg = f"{err_msg}: {detail['message']}"
                except Exception:
                    pass
                raise GoogleAPIError(resp.status_code, err_msg)

        raise GoogleAPIError(429, "Google API rate limit retries exhausted")

    def routes(self) -> list[Route]:
        base_url = self.settings.zos_base_url.rstrip("/")
        callback_uri = f"{base_url}/connect/google/callback"

        html_headers = {
            "Cache-Control": "no-store",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "frame-ancestors 'none'",
        }

        async def connect_google(request: Request) -> Response:
            if request.method == "GET":
                status = (
                    "Google is currently connected."
                    if self.is_connected
                    else None
                )
                return HTMLResponse(
                    render_google_connect_page(status_message=status),
                    headers=html_headers,
                )

            if not self.settings.zos_owner_password_hash:
                return HTMLResponse(
                    "<h1>ZOS_OWNER_PASSWORD_HASH is not configured</h1>",
                    status_code=503,
                    headers=html_headers,
                )
            if not self.settings.google_client_id or not self.settings.google_client_secret:
                return HTMLResponse(
                    "<h1>GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not configured</h1>",
                    status_code=503,
                    headers=html_headers,
                )
            if self._store is None:
                return HTMLResponse(
                    "<h1>ZOS_SECRET_KEY is not configured</h1>",
                    status_code=503,
                    headers=html_headers,
                )

            form = await request.form()
            passphrase = str(form.get("passphrase", ""))
            if not verify_owner_passphrase(passphrase, self.settings.zos_owner_password_hash):
                return HTMLResponse(
                    render_google_connect_page(error="Invalid owner passphrase."),
                    status_code=401,
                    headers=html_headers,
                )

            now = time.monotonic()
            # Prune expired pending states
            self._pending_states = {k: exp for k, exp in self._pending_states.items() if exp > now}
            state = secrets.token_urlsafe(24)
            self._pending_states[state] = now + STATE_TTL_SECONDS

            params = {
                "client_id": self.settings.google_client_id,
                "redirect_uri": callback_uri,
                "response_type": "code",
                "scope": " ".join(GOOGLE_SCOPES),
                "access_type": "offline",
                "prompt": "consent",
                "state": state,
            }
            return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=302)

        async def connect_google_callback(request: Request) -> Response:
            error = request.query_params.get("error")
            if error:
                return HTMLResponse(f"<h1>Google authorization failed: {error}</h1>", status_code=400)

            state = request.query_params.get("state", "")
            code = request.query_params.get("code", "")
            exp = self._pending_states.pop(state, None)
            if exp is None or exp < time.monotonic():
                return HTMLResponse("<h1>Invalid or expired OAuth state</h1>", status_code=400)
            if not code:
                return HTMLResponse("<h1>Missing authorization code</h1>", status_code=400)

            async with httpx.AsyncClient(timeout=20.0) as client:
                token_resp = await client.post(
                    GOOGLE_TOKEN_URL,
                    data={
                        "client_id": self.settings.google_client_id,
                        "client_secret": self.settings.google_client_secret,
                        "code": code,
                        "redirect_uri": callback_uri,
                        "grant_type": "authorization_code",
                    },
                )
                if token_resp.status_code != 200:
                    return HTMLResponse("<h1>Failed to exchange Google OAuth code</h1>", status_code=502)
                token_data = token_resp.json()

                access_token = str(token_data.get("access_token", ""))
                refresh_token = str(token_data.get("refresh_token", ""))
                expires_in = int(token_data.get("expires_in", 3600))
                scopes = sorted(str(token_data.get("scope", "")).split()) or list(GOOGLE_SCOPES)
                if not access_token or not refresh_token:
                    return HTMLResponse(
                        "<h1>Google did not return a refresh_token (ensure prompt=consent & access_type=offline)</h1>",
                        status_code=502,
                    )

                account = "google-account"
                try:
                    prof_resp = await client.get(
                        GMAIL_PROFILE_URL,
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    if prof_resp.status_code == 200:
                        account = str(prof_resp.json().get("emailAddress") or account)
                except Exception:
                    pass

            await self.attach_google_tokens(
                account=account,
                scopes=scopes,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=expires_in,
            )
            return HTMLResponse(
                render_google_connect_page(
                    status_message="Google Calendar and Gmail are now connected and online."
                ),
                status_code=200,
                headers={"Cache-Control": "no-store"},
            )

        return [
            Route("/connect/google", connect_google, methods=["GET", "POST"]),
            Route("/connect/google/callback", connect_google_callback, methods=["GET"]),
        ]
