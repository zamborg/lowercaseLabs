"""Owner-managed remote MCP upstream lifecycle, vault token sources, and /connect/<slug> OAuth flow."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import httpx2
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.auth.oauth2 import AuthorizationCodeResult
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool
from pydantic import AnyUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from zos.audit import record_event
from zos.auth.oauth import verify_owner_passphrase
from zos.auth.rate_limit import GlobalPassphraseRateLimiter, record_auth_failure
from zos.config import Settings
from zos.db.queries import (
    attach_upstream_credential,
    get_remote_upstream,
    list_remote_upstreams,
)
from zos.providers.mcp_upstream import (
    DEFAULT_CONNECT_TIMEOUT_S,
    ElicitationAuditFn,
    HttpClientFactory,
    McpUpstreamProvider,
)
from zos.router import ProviderRegistry, SessionTracker
from zos.vault import (
    REFRESH_SKEW_SECONDS,
    CredentialRevokedError,
    CredentialStore,
    CredentialVault,
)
from zos.web.pages import render_upstream_connect_page

logger = logging.getLogger(__name__)

CONNECT_FLOW_TTL_SECONDS = 600
CONNECT_REDIRECT_TIMEOUT_S = 20.0
CONNECT_CALLBACK_TIMEOUT_S = 20.0

ISSUER_MCP_BEARER = "mcp_bearer"
ISSUER_MCP_OAUTH = "mcp_oauth"

_HTML_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class VaultBearerTokenSource:
    """UpstreamTokenSource for static PAT / bearer tokens stored in the AEAD vault."""

    def __init__(
        self,
        *,
        slug: str,
        store: CredentialStore | None,
        credential_id: UUID | None,
        reconnect_url: str,
    ) -> None:
        self.slug = slug
        self.store = store
        self._credential_id = credential_id
        self._reconnect_url = reconnect_url
        self._needs_reconnect = False

    @property
    def credential_id(self) -> UUID | None:
        return self._credential_id

    @property
    def needs_reconnect(self) -> bool:
        return self._needs_reconnect

    def clear_needs_reconnect(self) -> None:
        self._needs_reconnect = False

    def set_credential_id(self, credential_id: UUID | None) -> None:
        self._credential_id = credential_id
        if credential_id is not None:
            self._needs_reconnect = False

    def has_credential(self) -> bool:
        return self.store is not None and self._credential_id is not None

    @property
    def reconnect_hint(self) -> str:
        return f"Update token for '{self.slug}' at {self._reconnect_url} or via `zos upstream set-token {self.slug}`."

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        if not self.has_credential() or self.store is None or self._credential_id is None:
            raise CredentialRevokedError(f"Upstream '{self.slug}' has no bearer token configured. {self.reconnect_hint}")
        if force_refresh:
            # Static bearer token cannot be refreshed; 401 means it is invalid or revoked.
            self._needs_reconnect = True
            raise CredentialRevokedError(
                f"Upstream '{self.slug}' rejected bearer token (401). {self.reconnect_hint}"
            )
        loaded = await self.store.load_credential(self._credential_id)
        if loaded is None:
            self._needs_reconnect = True
            raise CredentialRevokedError(f"Credential for '{self.slug}' not found. {self.reconnect_hint}")
        _row, payload = loaded
        token = str(payload.get("access_token") or "")
        if not token:
            self._needs_reconnect = True
            raise CredentialRevokedError(f"Empty bearer token for '{self.slug}'. {self.reconnect_hint}")
        return token


class VaultOAuthTokenStorage(TokenStorage):
    """`mcp.client.auth.TokenStorage` persisted inside ZOS's AEAD-encrypted `credentials` table."""

    def __init__(
        self,
        *,
        store: CredentialStore,
        principal_id: UUID,
        slug: str,
        credential_id: UUID | None = None,
    ) -> None:
        self.store = store
        self.principal_id = principal_id
        self.slug = slug
        self.credential_id = credential_id
        self._lock = asyncio.Lock()

    async def _load_state(self) -> tuple[UUID | None, dict[str, Any]]:
        if self.credential_id is not None:
            loaded = await self.store.load_credential(self.credential_id)
            if loaded is not None:
                row, payload = loaded
                return row["id"], dict(payload)
        found = await self.store.find_credential(
            principal_id=self.principal_id,
            issuer=ISSUER_MCP_OAUTH,
            account=self.slug,
        )
        if found is not None:
            row, payload = found
            self.credential_id = row["id"]
            return row["id"], dict(payload)
        return None, {}

    async def _save_state(
        self,
        payload: dict[str, Any],
        *,
        scopes: list[str] | None = None,
        access_expires_at: datetime | None = None,
    ) -> UUID:
        cred_id = await self.store.save_credential(
            principal_id=self.principal_id,
            issuer=ISSUER_MCP_OAUTH,
            account=self.slug,
            payload=payload,
            scopes=scopes,
            access_expires_at=access_expires_at,
        )
        self.credential_id = cred_id
        return cred_id

    async def get_tokens(self) -> OAuthToken | None:
        async with self._lock:
            _cid, state = await self._load_state()
        raw = state.get("tokens")
        if not isinstance(raw, dict):
            return None
        return OAuthToken.model_validate(raw)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        async with self._lock:
            _cid, state = await self._load_state()
            state["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
            state["needs_reconnect"] = False
            expires_at: datetime | None = None
            if tokens.expires_in is not None:
                expires_at = datetime.now(UTC) + timedelta(seconds=int(tokens.expires_in))
                state["expires_at"] = expires_at.isoformat()
            else:
                state.pop("expires_at", None)
            scopes = tokens.scope.split() if tokens.scope else []
            await self._save_state(state, scopes=scopes, access_expires_at=expires_at)

    async def clear_tokens(self, *, needs_reconnect: bool = False) -> None:
        async with self._lock:
            _cid, state = await self._load_state()
            if not state and self.credential_id is None:
                return
            state["tokens"] = None
            state.pop("expires_at", None)
            state["needs_reconnect"] = needs_reconnect
            await self._save_state(state, scopes=[], access_expires_at=None)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        async with self._lock:
            _cid, state = await self._load_state()
        raw = state.get("client_info")
        if not isinstance(raw, dict):
            return None
        return OAuthClientInformationFull.model_validate(raw)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        async with self._lock:
            _cid, state = await self._load_state()
            state["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
            exp_raw = state.get("expires_at")
            exp_dt = datetime.fromisoformat(exp_raw) if isinstance(exp_raw, str) else None
            toks = state.get("tokens") or {}
            scopes = str(toks.get("scope") or "").split() if isinstance(toks, dict) else []
            await self._save_state(state, scopes=scopes, access_expires_at=exp_dt)

    async def save_discovery_metadata(
        self,
        *,
        oauth_metadata: OAuthMetadata | None,
        protected_resource_metadata: ProtectedResourceMetadata | None,
    ) -> None:
        async with self._lock:
            _cid, state = await self._load_state()
            if oauth_metadata is not None:
                state["oauth_metadata"] = oauth_metadata.model_dump(mode="json", exclude_none=True)
            if protected_resource_metadata is not None:
                state["protected_resource_metadata"] = protected_resource_metadata.model_dump(
                    mode="json", exclude_none=True
                )
            exp_raw = state.get("expires_at")
            exp_dt = datetime.fromisoformat(exp_raw) if isinstance(exp_raw, str) else None
            toks = state.get("tokens") or {}
            scopes = str(toks.get("scope") or "").split() if isinstance(toks, dict) else []
            await self._save_state(state, scopes=scopes, access_expires_at=exp_dt)

    async def load_full_state(self) -> dict[str, Any]:
        async with self._lock:
            _cid, state = await self._load_state()
        return state


class VaultOAuthTokenSource:
    """UpstreamTokenSource backed by AEAD vault + `OAuthClientProvider` refresh logic."""

    def __init__(
        self,
        *,
        slug: str,
        server_url: str,
        callback_url: str,
        connect_url: str,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        store: CredentialStore | None,
        principal_id: UUID,
        space_id: UUID,
        provider_id: UUID,
        credential_id: UUID | None = None,
        http_client_factory: HttpClientFactory | None = None,
        on_needs_reconnect: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.slug = slug
        self.server_url = server_url
        self.callback_url = callback_url
        self.connect_url = connect_url
        self.pool = pool
        self.store = store
        self.principal_id = principal_id
        self.space_id = space_id
        self.provider_id = provider_id
        self._credential_id = credential_id
        self._http_client_factory = http_client_factory
        self._on_needs_reconnect = on_needs_reconnect

        self._needs_reconnect = False
        self._has_token = credential_id is not None
        self._refresh_lock = asyncio.Lock()

        self.storage: VaultOAuthTokenStorage | None = (
            VaultOAuthTokenStorage(
                store=store,
                principal_id=principal_id,
                slug=slug,
                credential_id=credential_id,
            )
            if store is not None
            else None
        )

    @property
    def needs_reconnect(self) -> bool:
        return self._needs_reconnect

    def clear_needs_reconnect(self) -> None:
        self._needs_reconnect = False

    def set_connected(self, credential_id: UUID | None) -> None:
        self._credential_id = credential_id
        self._has_token = credential_id is not None
        self._needs_reconnect = False
        if self.storage is not None:
            self.storage.credential_id = credential_id

    async def sync_state_from_vault(self) -> None:
        if self.storage is None or self._credential_id is None:
            self._has_token = False
            return
        state = await self.storage.load_full_state()
        self._needs_reconnect = bool(state.get("needs_reconnect", False))
        toks = state.get("tokens")
        self._has_token = isinstance(toks, dict) and bool(toks.get("access_token"))

    def has_credential(self) -> bool:
        return self.storage is not None and self._credential_id is not None and self._has_token and not self._needs_reconnect

    @property
    def reconnect_hint(self) -> str:
        return f"Reconnect '{self.slug}' at {self.connect_url}."

    def _client_metadata(self) -> OAuthClientMetadata:
        return OAuthClientMetadata(
            client_name=f"ZOS ({self.slug})",
            redirect_uris=[AnyUrl(self.callback_url)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )

    def _make_http_client(self, auth: httpx2.Auth | None = None) -> httpx2.AsyncClient:
        if self._http_client_factory is not None:
            return self._http_client_factory(auth)
        return httpx2.AsyncClient(auth=auth, timeout=httpx2.Timeout(20.0))

    async def _transition_needs_reconnect(self, reason: str) -> None:
        self._needs_reconnect = True
        self._has_token = False
        if self.storage is not None:
            with contextlib.suppress(Exception):
                await self.storage.clear_tokens(needs_reconnect=True)
        await record_event(
            self.pool,
            principal_id=self.principal_id,
            space_id=self.space_id,
            provider_id=self.provider_id,
            event_type="provider.needs_reconnect",
            outcome="error",
            payload={"slug": self.slug, "reason": reason},
        )
        if self._on_needs_reconnect is not None:
            with contextlib.suppress(Exception):
                await self._on_needs_reconnect(reason)

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        if self.storage is None or self._credential_id is None:
            raise CredentialRevokedError(f"Upstream '{self.slug}' is not connected. {self.reconnect_hint}")

        # Fast path when fresh
        if not force_refresh:
            tok = await self._read_if_fresh()
            if tok is not None:
                return tok

        async with self._refresh_lock:
            if not force_refresh:
                tok = await self._read_if_fresh()
                if tok is not None:
                    return tok

            state = await self.storage.load_full_state()
            if state.get("needs_reconnect"):
                self._needs_reconnect = True
                self._has_token = False
                raise CredentialRevokedError(
                    f"Upstream '{self.slug}' needs reconnection. {self.reconnect_hint}"
                )

            raw_tokens = state.get("tokens")
            raw_client = state.get("client_info")
            if not isinstance(raw_tokens, dict) or not raw_tokens.get("access_token"):
                await self._transition_needs_reconnect("missing_token")
                raise CredentialRevokedError(
                    f"Upstream '{self.slug}' has no valid OAuth token. {self.reconnect_hint}"
                )

            tokens = OAuthToken.model_validate(raw_tokens)
            if not tokens.refresh_token:
                await self._transition_needs_reconnect("expired_without_refresh_token")
                raise CredentialRevokedError(
                    f"Upstream '{self.slug}' token expired and cannot be refreshed. {self.reconnect_hint}"
                )

            if not isinstance(raw_client, dict):
                await self._transition_needs_reconnect("missing_client_info")
                raise CredentialRevokedError(
                    f"Upstream '{self.slug}' OAuth client registration missing. {self.reconnect_hint}"
                )

            client_info = OAuthClientInformationFull.model_validate(raw_client)
            oauth_meta = (
                OAuthMetadata.model_validate(state["oauth_metadata"])
                if isinstance(state.get("oauth_metadata"), dict)
                else None
            )
            prm = (
                ProtectedResourceMetadata.model_validate(state["protected_resource_metadata"])
                if isinstance(state.get("protected_resource_metadata"), dict)
                else None
            )

            oauth_provider = OAuthClientProvider(
                server_url=self.server_url,
                client_metadata=self._client_metadata(),
                storage=self.storage,
            )
            oauth_provider.context.client_info = client_info
            oauth_provider.context.current_tokens = tokens
            oauth_provider.context.oauth_metadata = oauth_meta
            oauth_provider.context.protected_resource_metadata = prm

            refresh_req = await oauth_provider._refresh_token()
            async with self._make_http_client(None) as http:
                refresh_resp = await http.send(refresh_req)
                ok = await oauth_provider._handle_refresh_response(refresh_resp)

            if not ok or oauth_provider.context.current_tokens is None:
                await self._transition_needs_reconnect("refresh_rejected")
                raise CredentialRevokedError(
                    f"Upstream '{self.slug}' OAuth refresh failed or was revoked. {self.reconnect_hint}"
                )

            self._needs_reconnect = False
            self._has_token = True
            await record_event(
                self.pool,
                principal_id=self.principal_id,
                space_id=self.space_id,
                provider_id=self.provider_id,
                event_type="credential.refreshed",
                payload={"credential_id": str(self._credential_id), "issuer": ISSUER_MCP_OAUTH, "slug": self.slug},
            )
            return oauth_provider.context.current_tokens.access_token

    async def _read_if_fresh(self) -> str | None:
        if self.storage is None:
            return None
        state = await self.storage.load_full_state()
        if state.get("needs_reconnect"):
            self._needs_reconnect = True
            self._has_token = False
            raise CredentialRevokedError(f"Upstream '{self.slug}' needs reconnection. {self.reconnect_hint}")
        raw_tokens = state.get("tokens")
        if not isinstance(raw_tokens, dict) or not raw_tokens.get("access_token"):
            return None
        exp_raw = state.get("expires_at")
        if isinstance(exp_raw, str):
            exp_dt = datetime.fromisoformat(exp_raw)
            if exp_dt <= datetime.now(UTC) + timedelta(seconds=REFRESH_SKEW_SECONDS):
                return None
        return str(raw_tokens["access_token"])


@dataclass
class _PendingOAuthFlow:
    slug: str
    expires_at: float
    auth_url_future: asyncio.Future[str]
    callback_future: asyncio.Future[AuthorizationCodeResult]
    task: asyncio.Task[None]
    storage: VaultOAuthTokenStorage
    oauth_provider: OAuthClientProvider
    completed_ok: bool = False
    error: str | None = None


@dataclass
class _ManagedUpstream:
    slug: str
    provider_id: UUID
    principal_id: UUID
    space_id: UUID
    url: str
    auth_mode: str
    default_tier: str
    provider: McpUpstreamProvider
    token_source: VaultBearerTokenSource | VaultOAuthTokenSource | None = None
    credential_id: UUID | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class UpstreamManager:
    """Manages generic remote MCP upstreams (`runtime='remote'`): startup, hot-reload, and /connect/<slug>."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        settings: Settings,
        registry: ProviderRegistry,
        sessions: SessionTracker,
        *,
        passphrase_limiter: GlobalPassphraseRateLimiter | None = None,
        audit_elicitation: ElicitationAuditFn | None = None,
        http_client_factory: HttpClientFactory | None = None,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    ) -> None:
        self.pool = pool
        self.settings = settings
        self.registry = registry
        self.sessions = sessions
        self.passphrase_limiter = passphrase_limiter or GlobalPassphraseRateLimiter()
        self._audit_elicitation = audit_elicitation
        self._http_client_factory = http_client_factory
        self._connect_timeout_s = connect_timeout_s

        self._store: CredentialStore | None = (
            CredentialStore(pool, CredentialVault(settings.zos_secret_key))
            if settings.zos_secret_key
            else None
        )
        self._managed: dict[str, _ManagedUpstream] = {}
        self._pending_flows: dict[str, _PendingOAuthFlow] = {}
        self._lock = asyncio.Lock()

    @property
    def store(self) -> CredentialStore | None:
        return self._store

    def get_managed(self, slug: str) -> _ManagedUpstream | None:
        return self._managed.get(slug)

    def get_provider(self, slug: str) -> McpUpstreamProvider | None:
        entry = self._managed.get(slug)
        return entry.provider if entry is not None else None

    def get_runtime_info(self, slug: str) -> dict[str, Any] | None:
        entry = self._managed.get(slug)
        if entry is None:
            return None
        return {
            "status": entry.provider.status,
            "online": entry.provider.is_online(),
            "tool_count": entry.provider.tool_count,
            "last_error": entry.provider.last_error,
            "needs_reconnect": entry.provider.needs_reconnect,
        }

    def _connect_url(self, slug: str) -> str:
        return f"{self.settings.zos_base_url.rstrip('/')}/connect/{slug}"

    def _callback_url(self, slug: str) -> str:
        return f"{self.settings.zos_base_url.rstrip('/')}/connect/{slug}/callback"

    def _admin_upstreams_url(self) -> str:
        return f"{self.settings.zos_base_url.rstrip('/')}/admin/upstreams"

    def _build_entry(self, row: dict[str, Any]) -> _ManagedUpstream:
        slug = str(row["slug"])
        cfg = row.get("config") if isinstance(row.get("config"), dict) else {}
        url = str(cfg.get("url") or "").strip()
        auth_mode = str(cfg.get("auth_mode") or "none").strip().lower()
        default_tier = str(cfg.get("default_tier") or "external").strip().lower()
        provider_id: UUID = row["id"]
        principal_id: UUID = row["principal_id"]
        space_id: UUID = row["space_id"]
        credential_id: UUID | None = row.get("credential_id")

        token_source: VaultBearerTokenSource | VaultOAuthTokenSource | None = None

        async def _on_auth_failed(reason: str) -> None:
            if isinstance(token_source, VaultOAuthTokenSource):
                await token_source._transition_needs_reconnect(reason)
            elif isinstance(token_source, VaultBearerTokenSource):
                token_source._needs_reconnect = True
                await record_event(
                    self.pool,
                    principal_id=principal_id,
                    space_id=space_id,
                    provider_id=provider_id,
                    event_type="provider.needs_reconnect",
                    outcome="error",
                    payload={"slug": slug, "reason": reason},
                )

        if auth_mode == "bearer":
            token_source = VaultBearerTokenSource(
                slug=slug,
                store=self._store,
                credential_id=credential_id,
                reconnect_url=self._admin_upstreams_url(),
            )
        elif auth_mode == "oauth":
            token_source = VaultOAuthTokenSource(
                slug=slug,
                server_url=url,
                callback_url=self._callback_url(slug),
                connect_url=self._connect_url(slug),
                pool=self.pool,
                store=self._store,
                principal_id=principal_id,
                space_id=space_id,
                provider_id=provider_id,
                credential_id=credential_id,
                http_client_factory=self._http_client_factory,
            )

        mcp_provider = McpUpstreamProvider(
            namespace=slug,
            url=url,
            token_source=token_source,
            sessions=self.sessions,
            provider_id_getter=lambda pid=provider_id: pid,
            audit_elicitation=self._audit_elicitation,
            http_client_factory=self._http_client_factory,
            connect_timeout_s=self._connect_timeout_s,
            auth_mode=auth_mode,
            on_auth_failed=_on_auth_failed,
        )
        return _ManagedUpstream(
            slug=slug,
            provider_id=provider_id,
            principal_id=principal_id,
            space_id=space_id,
            url=url,
            auth_mode=auth_mode,
            default_tier=default_tier,
            provider=mcp_provider,
            token_source=token_source,
            credential_id=credential_id,
        )

    async def load_all_on_startup(self) -> None:
        """Load and connect every registered remote MCP upstream without letting one offline break startup."""
        try:
            rows = await list_remote_upstreams(self.pool)
        except Exception as exc:
            logger.warning("Could not load remote MCP upstreams on startup: %s", exc)
            return

        entries: list[_ManagedUpstream] = []
        async with self._lock:
            for row in rows:
                if not row.get("enabled", True):
                    continue
                try:
                    entry = self._build_entry(row)
                    if isinstance(entry.token_source, VaultOAuthTokenSource):
                        await entry.token_source.sync_state_from_vault()
                    self._managed[entry.slug] = entry
                    self.registry.register(entry.provider)
                    entries.append(entry)
                except Exception as exc:
                    logger.warning("Skipping remote upstream '%s' on startup: %s", row.get("slug"), exc)

        if not entries:
            return

        async def _safe_connect(entry: _ManagedUpstream) -> None:
            try:
                if entry.auth_mode == "none" or (
                    entry.token_source is not None and entry.token_source.has_credential()
                ):
                    await entry.provider.ensure_connected(broadcast_on_change=False)
            except Exception as exc:
                logger.warning("Remote upstream '%s' offline at startup: %s", entry.slug, exc)

        await asyncio.gather(*(_safe_connect(e) for e in entries), return_exceptions=True)

    async def hot_register_from_row(self, row: dict[str, Any], *, connect: bool = True) -> _ManagedUpstream:
        """Hot-register (or replace) a remote upstream in the live registry and broadcast list_changed."""
        slug = str(row["slug"])
        async with self._lock:
            old = self._managed.pop(slug, None)
            if old is not None:
                self.registry.unregister(slug)
                with contextlib.suppress(Exception):
                    await old.provider.aclose()

            entry = self._build_entry(row)
            if isinstance(entry.token_source, VaultOAuthTokenSource):
                await entry.token_source.sync_state_from_vault()
            self._managed[slug] = entry
            self.registry.register(entry.provider)

        if connect and (
            entry.auth_mode == "none"
            or (entry.token_source is not None and entry.token_source.has_credential())
        ):
            with contextlib.suppress(Exception):
                await entry.provider.ensure_connected(broadcast_on_change=False)

        await self.sessions.broadcast_tools_changed()
        return entry

    async def hot_update_bearer_token(self, slug: str, credential_id: UUID) -> None:
        """Hot-attach a newly saved bearer credential, reconnect upstream, and broadcast list_changed."""
        entry = self._managed.get(slug)
        if entry is None:
            row = await get_remote_upstream(self.pool, slug=slug)
            if row is not None:
                await self.hot_register_from_row(row, connect=True)
            return

        entry.credential_id = credential_id
        if isinstance(entry.token_source, VaultBearerTokenSource):
            entry.token_source.set_credential_id(credential_id)
        entry.provider.clear_needs_reconnect()
        await entry.provider.disconnect(broadcast_on_change=False)
        with contextlib.suppress(Exception):
            await entry.provider.ensure_connected(broadcast_on_change=False)
        await self.sessions.broadcast_tools_changed()

    async def hot_disconnect(self, slug: str) -> None:
        """Disconnect live upstream session, clear credentials in memory, and broadcast list_changed."""
        await self._cancel_pending_flow(slug)
        entry = self._managed.get(slug)
        if entry is None:
            return
        entry.credential_id = None
        if isinstance(entry.token_source, VaultBearerTokenSource):
            entry.token_source.set_credential_id(None)
        elif isinstance(entry.token_source, VaultOAuthTokenSource):
            entry.token_source.set_connected(None)
        entry.provider.clear_needs_reconnect()
        await entry.provider.disconnect(broadcast_on_change=False)
        await self.sessions.broadcast_tools_changed()

    async def hot_remove(self, slug: str) -> None:
        """Unregister and close a removed remote upstream and broadcast list_changed."""
        await self._cancel_pending_flow(slug)
        async with self._lock:
            old = self._managed.pop(slug, None)
            self.registry.unregister(slug)
        if old is not None:
            with contextlib.suppress(Exception):
                await old.provider.aclose()
        await self.sessions.broadcast_tools_changed()

    async def aclose(self) -> None:
        """Clean shutdown hook called from ASGI lifespan exit."""
        for slug in list(self._pending_flows):
            await self._cancel_pending_flow(slug)
        async with self._lock:
            entries = list(self._managed.values())
            self._managed.clear()
        for entry in entries:
            with contextlib.suppress(Exception):
                await entry.provider.aclose()

    async def _cancel_pending_flow(self, slug: str) -> None:
        flow = self._pending_flows.pop(slug, None)
        if flow is not None and not flow.task.done():
            flow.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(flow.task, timeout=2.0)

    def _prune_expired_flows(self) -> None:
        now = time.monotonic()
        for slug, flow in list(self._pending_flows.items()):
            if flow.expires_at <= now:
                self._pending_flows.pop(slug, None)
                if not flow.task.done():
                    flow.task.cancel()

    async def _start_oauth_connect_flow(self, row: dict[str, Any]) -> str:
        """Drive `OAuthClientProvider` through PRM + DCR + PKCE and return the upstream authorize URL."""
        if self._store is None:
            raise RuntimeError("ZOS_SECRET_KEY is not configured; cannot store OAuth credentials")

        slug = str(row["slug"])
        cfg = row["config"] if isinstance(row.get("config"), dict) else {}
        server_url = str(cfg.get("url") or "").strip()
        principal_id: UUID = row["principal_id"]
        credential_id: UUID | None = row.get("credential_id")

        self._prune_expired_flows()
        await self._cancel_pending_flow(slug)

        storage = VaultOAuthTokenStorage(
            store=self._store,
            principal_id=principal_id,
            slug=slug,
            credential_id=credential_id,
        )
        # Clear any stale/revoked tokens so OAuthClientProvider initiates a fresh interactive grant
        await storage.clear_tokens(needs_reconnect=False)

        loop = asyncio.get_running_loop()
        auth_url_future: asyncio.Future[str] = loop.create_future()
        callback_future: asyncio.Future[AuthorizationCodeResult] = loop.create_future()

        async def redirect_handler(auth_url: str) -> None:
            if not auth_url_future.done():
                auth_url_future.set_result(auth_url)

        async def callback_handler() -> AuthorizationCodeResult:
            return await asyncio.wait_for(callback_future, timeout=CONNECT_FLOW_TTL_SECONDS)

        client_metadata = OAuthClientMetadata(
            client_name=f"ZOS ({slug})",
            redirect_uris=[AnyUrl(self._callback_url(slug))],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
        oauth_provider = OAuthClientProvider(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

        async def _drive_flow() -> None:
            http = (
                self._http_client_factory(oauth_provider)
                if self._http_client_factory is not None
                else httpx2.AsyncClient(auth=oauth_provider, timeout=httpx2.Timeout(20.0))
            )
            try:
                async with http:
                    # Probe upstream MCP endpoint; 401 triggers SDK PRM -> ASM -> DCR -> PKCE flow
                    await http.post(
                        server_url,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json, text/event-stream",
                            "MCP-Protocol-Version": "2025-03-26",
                        },
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2025-03-26",
                                "capabilities": {},
                                "clientInfo": {"name": "zos", "version": "0.1.0"},
                            },
                        },
                    )
                # Persist discovered metadata alongside tokens/client_info for post-restart refresh
                await storage.save_discovery_metadata(
                    oauth_metadata=oauth_provider.context.oauth_metadata,
                    protected_resource_metadata=oauth_provider.context.protected_resource_metadata,
                )
                flow.completed_ok = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # If tokens were already stored before the post-auth probe failed, still treat as connected
                stored = await storage.get_tokens()
                if stored is not None and stored.access_token:
                    await storage.save_discovery_metadata(
                        oauth_metadata=oauth_provider.context.oauth_metadata,
                        protected_resource_metadata=oauth_provider.context.protected_resource_metadata,
                    )
                    flow.completed_ok = True
                else:
                    flow.error = str(exc)
                    if not auth_url_future.done():
                        auth_url_future.set_exception(exc)

        task = asyncio.create_task(_drive_flow(), name=f"zos-oauth-connect-{slug}")
        flow = _PendingOAuthFlow(
            slug=slug,
            expires_at=time.monotonic() + CONNECT_FLOW_TTL_SECONDS,
            auth_url_future=auth_url_future,
            callback_future=callback_future,
            task=task,
            storage=storage,
            oauth_provider=oauth_provider,
        )
        self._pending_flows[slug] = flow

        try:
            auth_url = await asyncio.wait_for(auth_url_future, timeout=CONNECT_REDIRECT_TIMEOUT_S)
        except Exception:
            await self._cancel_pending_flow(slug)
            raise
        return auth_url

    def routes(self) -> list[Route]:
        """Return `/connect/{slug}` and `/connect/{slug}/callback` routes for OAuth remote MCP servers."""

        async def connect_upstream(request: Request) -> Response:
            slug = str(request.path_params.get("slug", "")).strip().lower()
            row = await get_remote_upstream(self.pool, slug=slug)
            if row is None:
                return HTMLResponse(
                    f"<h1>Unknown MCP upstream '{slug}'</h1>",
                    status_code=404,
                    headers=_HTML_HEADERS,
                )
            cfg = row["config"] if isinstance(row.get("config"), dict) else {}
            auth_mode = str(cfg.get("auth_mode") or "none").lower()
            url = str(cfg.get("url") or "")
            if auth_mode != "oauth":
                return HTMLResponse(
                    f"<h1>Upstream '{slug}' uses auth_mode='{auth_mode}', not OAuth.</h1>",
                    status_code=400,
                    headers=_HTML_HEADERS,
                )

            if request.method == "GET":
                entry = self._managed.get(slug)
                status_msg = (
                    f"'{slug}' is currently connected ({entry.provider.tool_count} tool(s))."
                    if entry is not None and entry.provider.is_online()
                    else None
                )
                return HTMLResponse(
                    render_upstream_connect_page(
                        slug=slug,
                        upstream_url=url,
                        status_message=status_msg,
                    ),
                    headers=_HTML_HEADERS,
                )

            retry_after = self.passphrase_limiter.retry_after_seconds()
            if retry_after is not None:
                return HTMLResponse(
                    render_upstream_connect_page(
                        slug=slug,
                        upstream_url=url,
                        error=f"Too many failed attempts. Try again in {retry_after}s.",
                    ),
                    status_code=429,
                    headers={**_HTML_HEADERS, "Retry-After": str(retry_after)},
                )

            if not self.settings.zos_owner_password_hash:
                return HTMLResponse(
                    "<h1>ZOS_OWNER_PASSWORD_HASH is not configured</h1>",
                    status_code=503,
                    headers=_HTML_HEADERS,
                )
            if self._store is None:
                return HTMLResponse(
                    "<h1>ZOS_SECRET_KEY is not configured</h1>",
                    status_code=503,
                    headers=_HTML_HEADERS,
                )

            form = await request.form()
            passphrase = str(form.get("passphrase", ""))
            if not verify_owner_passphrase(passphrase, self.settings.zos_owner_password_hash):
                self.passphrase_limiter.record_failure()
                await record_auth_failure(
                    self.pool,
                    route=f"/connect/{slug}",
                    reason="invalid_passphrase",
                )
                return HTMLResponse(
                    render_upstream_connect_page(
                        slug=slug,
                        upstream_url=url,
                        error="Invalid owner passphrase.",
                    ),
                    status_code=401,
                    headers=_HTML_HEADERS,
                )

            self.passphrase_limiter.record_success()
            try:
                auth_url = await self._start_oauth_connect_flow(row)
            except Exception as exc:
                logger.warning("Failed starting OAuth flow for upstream '%s': %s", slug, exc)
                return HTMLResponse(
                    render_upstream_connect_page(
                        slug=slug,
                        upstream_url=url,
                        error=f"Could not start OAuth flow for '{slug}': {exc}",
                    ),
                    status_code=502,
                    headers=_HTML_HEADERS,
                )

            return RedirectResponse(auth_url, status_code=302, headers=_HTML_HEADERS)

        async def connect_upstream_callback(request: Request) -> Response:
            slug = str(request.path_params.get("slug", "")).strip().lower()
            row = await get_remote_upstream(self.pool, slug=slug)
            if row is None:
                return HTMLResponse(
                    f"<h1>Unknown MCP upstream '{slug}'</h1>",
                    status_code=404,
                    headers=_HTML_HEADERS,
                )
            cfg = row["config"] if isinstance(row.get("config"), dict) else {}
            url = str(cfg.get("url") or "")

            error = request.query_params.get("error")
            if error:
                await self._cancel_pending_flow(slug)
                desc = request.query_params.get("error_description") or error
                return HTMLResponse(
                    render_upstream_connect_page(
                        slug=slug,
                        upstream_url=url,
                        error=f"Upstream authorization failed: {desc}",
                    ),
                    status_code=400,
                    headers=_HTML_HEADERS,
                )

            flow = self._pending_flows.get(slug)
            if flow is None or flow.expires_at <= time.monotonic():
                await self._cancel_pending_flow(slug)
                return HTMLResponse(
                    render_upstream_connect_page(
                        slug=slug,
                        upstream_url=url,
                        error="OAuth session expired or not found. Please start again.",
                    ),
                    status_code=400,
                    headers=_HTML_HEADERS,
                )

            code = request.query_params.get("code", "")
            state = request.query_params.get("state")
            iss = request.query_params.get("iss")
            if not code:
                await self._cancel_pending_flow(slug)
                return HTMLResponse(
                    render_upstream_connect_page(
                        slug=slug,
                        upstream_url=url,
                        error="Missing authorization code.",
                    ),
                    status_code=400,
                    headers=_HTML_HEADERS,
                )

            if not flow.callback_future.done():
                flow.callback_future.set_result(
                    AuthorizationCodeResult(code=code, state=state, iss=iss)
                )

            try:
                await asyncio.wait_for(flow.task, timeout=CONNECT_CALLBACK_TIMEOUT_S)
            except Exception as exc:
                self._pending_flows.pop(slug, None)
                return HTMLResponse(
                    render_upstream_connect_page(
                        slug=slug,
                        upstream_url=url,
                        error=f"OAuth token exchange failed: {exc}",
                    ),
                    status_code=502,
                    headers=_HTML_HEADERS,
                )

            self._pending_flows.pop(slug, None)
            if not flow.completed_ok or flow.storage.credential_id is None:
                return HTMLResponse(
                    render_upstream_connect_page(
                        slug=slug,
                        upstream_url=url,
                        error=f"OAuth token exchange failed: {flow.error or 'no tokens stored'}",
                    ),
                    status_code=502,
                    headers=_HTML_HEADERS,
                )

            cred_id = flow.storage.credential_id
            await attach_upstream_credential(
                self.pool,
                provider_id=row["id"],
                credential_id=cred_id,
            )
            refreshed_row = await get_remote_upstream(self.pool, slug=slug)
            assert refreshed_row is not None
            entry = await self.hot_register_from_row(refreshed_row, connect=True)

            await record_event(
                self.pool,
                principal_id=row["principal_id"],
                space_id=row["space_id"],
                provider_id=row["id"],
                event_type="credential.connected",
                payload={"issuer": ISSUER_MCP_OAUTH, "slug": slug, "credential_id": str(cred_id)},
            )
            await record_event(
                self.pool,
                principal_id=row["principal_id"],
                space_id=row["space_id"],
                provider_id=row["id"],
                event_type="provider.connected",
                payload={"slug": slug, "auth_mode": "oauth", "tool_count": entry.provider.tool_count},
            )

            status_text = (
                f"Connected '{slug}' ({entry.provider.tool_count} tool(s) loaded and live)."
                if entry.provider.is_online()
                else f"Saved OAuth credentials for '{slug}', but upstream is currently unreachable."
            )
            return HTMLResponse(
                render_upstream_connect_page(
                    slug=slug,
                    upstream_url=url,
                    status_message=status_text,
                    connected=True,
                ),
                status_code=200,
                headers=_HTML_HEADERS,
            )

        return [
            Route("/connect/{slug:str}", connect_upstream, methods=["GET", "POST"]),
            Route("/connect/{slug:str}/callback", connect_upstream_callback, methods=["GET"]),
        ]


def validate_upstream_url(url: str) -> str:
    """Validate a remote MCP server URL (`https://`, or loopback `http://` for local dev/tests)."""
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise ValueError(f"Invalid upstream URL {url!r}: must be a valid https:// URL")
    host = parsed.hostname.lower()
    if parsed.scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(f"Upstream URL must use https:// (got {url!r})")
    return cleaned
