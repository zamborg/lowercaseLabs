"""Google OAuth client, concurrency-safe token refresh, and REST API helper with 429 backoff."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx

from zos_google.store import GoogleTokenStore

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
GCAL_API_BASE = "https://www.googleapis.com/calendar/v3"
DEFAULT_GCAL_MCP_URL = "https://calendarmcp.googleapis.com/mcp/v1"

SERVICE_SCOPES: dict[str, tuple[str, ...]] = {
    "gmail": (
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
    ),
    "gcal": (
        "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
        "https://www.googleapis.com/auth/calendar.events.freebusy",
        "https://www.googleapis.com/auth/calendar.events.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    ),
}

REFRESH_SKEW_SECONDS = 60


class GoogleNotConnectedError(RuntimeError):
    """Raised when no valid Google credential is stored or the refresh token was revoked."""


class GoogleAPIError(RuntimeError):
    """Raised on non-2xx Google REST API responses."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class GoogleClient:
    """Owns Google OAuth code exchange, encrypted token refresh, and authenticated REST calls."""

    def __init__(
        self,
        store: GoogleTokenStore,
        *,
        service: str,
        client_id: str,
        client_secret: str,
        auth_url: str = GOOGLE_AUTH_URL,
        token_url: str = GOOGLE_TOKEN_URL,
        gmail_api_base: str = GMAIL_API_BASE,
        gcal_api_base: str = GCAL_API_BASE,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        backoff_base_s: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        if service not in SERVICE_SCOPES:
            raise ValueError(f"Unsupported Google service: {service!r}")
        self.store = store
        self.service = service
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.auth_url = auth_url
        self.token_url = token_url
        self.gmail_api_base = gmail_api_base.rstrip("/")
        self.gcal_api_base = gcal_api_base.rstrip("/")
        self._http_client_factory = http_client_factory
        self.backoff_base_s = backoff_base_s
        self.max_retries = max_retries
        self._refresh_lock = asyncio.Lock()

    @property
    def scopes(self) -> tuple[str, ...]:
        return SERVICE_SCOPES[self.service]

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _http(self, *, timeout: float = 30.0) -> httpx.AsyncClient:
        if self._http_client_factory is not None:
            return self._http_client_factory()
        return httpx.AsyncClient(timeout=timeout)

    def build_authorize_url(self, *, redirect_uri: str, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        return f"{self.auth_url}?{urlencode(params)}"

    async def exchange_code(self, *, code: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange an authorization code for tokens and save them encrypted in module storage."""
        if not self.is_configured:
            raise RuntimeError("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are not configured")

        existing = await self.store.load_tokens()
        async with self._http(timeout=20.0) as client:
            resp = await client.post(
                self.token_url,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            if resp.status_code != 200:
                raise GoogleAPIError(resp.status_code, "Failed to exchange Google OAuth code")
            data = resp.json()

            access_token = str(data.get("access_token") or "")
            refresh_token = str(data.get("refresh_token") or "")
            if not refresh_token and existing is not None:
                refresh_token = str(existing.get("refresh_token") or "")
            expires_in = int(data.get("expires_in", 3600))
            scopes = sorted(str(data.get("scope") or "").split()) or list(self.scopes)
            if not access_token or not refresh_token:
                raise RuntimeError(
                    "Google did not return a refresh_token (ensure prompt=consent & access_type=offline)"
                )

            account = await self._fetch_account_label(client, access_token)

        await self.store.save_tokens(
            account=account,
            scopes=scopes,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
        )
        return {"account": account, "scopes": scopes}

    async def _fetch_account_label(self, client: httpx.AsyncClient, access_token: str) -> str:
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            if self.service == "gmail":
                prof = await client.get(f"{self.gmail_api_base}/profile", headers=headers)
                if prof.status_code == 200:
                    email = str(prof.json().get("emailAddress") or "").strip()
                    if email:
                        return email
            else:
                cal = await client.get(
                    f"{self.gcal_api_base}/users/me/calendarList/primary",
                    headers=headers,
                )
                if cal.status_code == 200:
                    cid = str(cal.json().get("id") or "").strip()
                    if cid:
                        return cid
        except Exception:
            pass
        return "google-account"

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        """Return a valid access token, refreshing under a lock when expired or forced."""
        rec = await self.store.load_tokens()
        if rec is None or not rec.get("refresh_token"):
            raise GoogleNotConnectedError(
                f"Google ({self.service}) is not connected. Connect at /admin/m/{self.service}."
            )

        now = datetime.now(UTC)
        exp = rec.get("access_expires_at")
        if (
            not force_refresh
            and rec.get("access_token")
            and isinstance(exp, datetime)
            and exp > now + timedelta(seconds=REFRESH_SKEW_SECONDS)
        ):
            return str(rec["access_token"])

        async with self._refresh_lock:
            # Re-read under lock to avoid duplicate refreshes across concurrent calls
            rec2 = await self.store.load_tokens()
            if rec2 is None or not rec2.get("refresh_token"):
                raise GoogleNotConnectedError(
                    f"Google ({self.service}) is not connected. Connect at /admin/m/{self.service}."
                )
            exp2 = rec2.get("access_expires_at")
            if (
                not force_refresh
                and rec2.get("access_token")
                and isinstance(exp2, datetime)
                and exp2 > datetime.now(UTC) + timedelta(seconds=REFRESH_SKEW_SECONDS)
            ):
                return str(rec2["access_token"])

            async with self._http(timeout=20.0) as client:
                resp = await client.post(
                    self.token_url,
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "refresh_token": rec2["refresh_token"],
                        "grant_type": "refresh_token",
                    },
                )
            if resp.status_code in (400, 401):
                err_code = ""
                with contextlib.suppress(Exception):
                    err_code = str(resp.json().get("error") or "")
                if err_code in ("invalid_grant", "unauthorized_client"):
                    await self.store.delete_tokens()
                    raise GoogleNotConnectedError(
                        f"Google ({self.service}) refresh token revoked ({err_code}). "
                        f"Reconnect at /admin/m/{self.service}."
                    )
            if resp.status_code != 200:
                raise GoogleAPIError(resp.status_code, f"Google token refresh failed ({resp.status_code})")

            data = resp.json()
            new_access = str(data["access_token"])
            new_expires = int(data.get("expires_in", 3600))
            new_refresh = str(data.get("refresh_token") or rec2["refresh_token"])
            await self.store.save_tokens(
                account=str(rec2["account"]),
                scopes=list(rec2["scopes"]),
                access_token=new_access,
                refresh_token=new_refresh,
                expires_in=new_expires,
            )
            return new_access

    async def api_request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | list[tuple[str, str]] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an authenticated Google REST request with 401 refresh-retry and 429 backoff."""
        access_token = await self.get_access_token()
        refreshed_after_401 = False

        async with self._http(timeout=30.0) as client:
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
                    access_token = await self.get_access_token(force_refresh=True)
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
