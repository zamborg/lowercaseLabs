"""Shared admin service layer backing both the `zos` CLI and the `/admin` dashboard.

Every mutation goes through this service and writes the exact audit events to `events`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from mcp.shared.auth import OAuthClientInformationFull
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool
from pydantic import AnyUrl

from zos.audit import record_event
from zos.catalog import get_catalog_entry
from zos.db.queries import (
    attach_upstream_credential,
    clear_client_policy,
    clear_upstream_credential,
    create_actuator_provider_and_token,
    create_bearer_client,
    create_remote_upstream,
    delete_oauth_client,
    delete_remote_upstream,
    find_client_by_name_or_id,
    get_principal_and_space,
    get_remote_upstream,
    list_remote_upstreams,
    mint_bearer_token_for_client,
    revoke_token,
    set_client_policy,
    set_provider_backend,
    set_tool_tier,
    unmark_managed_tier_pattern,
)
from zos.db.queries import (
    disable_client as db_disable_client,
)
from zos.policy import Decision, Tier, build_tool_annotations, resolve_decision, resolve_tier
from zos.providers.upstream_manager import (
    ISSUER_MCP_BEARER,
    UpstreamManager,
    VaultOAuthTokenStorage,
    validate_upstream_url,
)
from zos.router import ZOS_STATUS_TOOL_NAME, ProviderRegistry, SessionTracker, format_tool_name
from zos.router.naming import BUILTIN_PROVIDER_SLUGS, validate_upstream_slug
from zos.vault import CredentialStore, CredentialVault

SWITCHABLE_PROVIDER_BACKENDS: dict[str, tuple[str, ...]] = {
    "gcal": ("api", "mcp"),
}
_DEFAULT_PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    "gcal": {"transport": "mcp", "url": "https://calendarmcp.googleapis.com/mcp/v1"},
}


class AdminService:
    """Single source of truth for ZOS owner administration queries and audited mutations."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> None:
        self.pool = pool

    # ── Mutations (shared by CLI and /admin) ──────────────────────────────

    async def register_bearer_client(
        self,
        *,
        name: str,
        trust_ceiling: str = "owner",
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        if trust_ceiling not in ("owner", "untrusted"):
            raise ValueError("trust_ceiling must be 'owner' or 'untrusted'")
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Client name must be non-empty")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        res = await create_bearer_client(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            name=clean_name,
            trust_ceiling=trust_ceiling,
        )
        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            client_id=res["client"]["id"],
            event_type="client.registered",
            payload={"name": clean_name, "kind": "bearer", "trust_ceiling": trust_ceiling},
        )
        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            client_id=res["client"]["id"],
            event_type="token.issued",
            payload={"token_id": str(res["token_id"]), "kind": "bearer"},
        )
        return res

    async def create_bearer_token(
        self,
        *,
        client_name: str,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        p, sp = await get_principal_and_space(self.pool, principal_slug)
        res = await mint_bearer_token_for_client(
            self.pool,
            principal_id=p["id"],
            client_name=client_name,
        )
        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            client_id=res["client"]["id"],
            event_type="token.issued",
            payload={"token_id": str(res["token_id"]), "kind": "bearer"},
        )
        return res

    async def revoke_tokens(
        self,
        *,
        target: str | None = None,
        client_identifier: str | None = None,
        principal_slug: str | None = None,
    ) -> int:
        if not target and not client_identifier:
            raise ValueError("Must specify target token or client_identifier")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        client_id: UUID | None = None
        token_id: UUID | None = None
        raw_token: str | None = None

        if client_identifier:
            c = await find_client_by_name_or_id(
                self.pool, principal_id=p["id"], identifier=client_identifier
            )
            if c is None:
                raise ValueError(f"Client not found: {client_identifier}")
            client_id = c["id"]
        elif target:
            try:
                token_id = UUID(target)
            except ValueError:
                raw_token = target

        count = await revoke_token(
            self.pool,
            token_id=token_id,
            raw_token=raw_token,
            client_id=client_id,
        )
        if count > 0:
            await record_event(
                self.pool,
                principal_id=p["id"],
                space_id=sp["id"],
                client_id=client_id,
                event_type="token.revoked",
                payload={"count": count, "token_id": str(token_id) if token_id else None},
            )
        return count

    async def delete_oauth_client(
        self,
        *,
        client_identifier: str,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        clean_id = client_identifier.strip()
        if not clean_id:
            raise ValueError("Must specify OAuth client ID or name to prune")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        c = await find_client_by_name_or_id(self.pool, principal_id=p["id"], identifier=clean_id)
        if c is None:
            raise ValueError(f"Client not found: {clean_id}")
        if c["kind"] != "oauth":
            raise ValueError("Only OAuth clients can be pruned; revoke bearer tokens instead.")

        deleted = await delete_oauth_client(self.pool, principal_id=p["id"], client_id=c["id"])
        if deleted is None:
            raise ValueError(f"OAuth client not found: {clean_id}")

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            client_id=None,
            event_type="client.deleted",
            payload={
                "deleted_client_id": str(deleted["id"]),
                "name": deleted["name"],
                "kind": "oauth",
                "oauth_client_id": deleted.get("oauth_client_id"),
            },
        )
        return deleted

    async def disable_client(
        self,
        *,
        client_identifier: str,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        """Soft-disable any client (bearer or oauth), revoking tokens while preserving history."""
        clean_id = client_identifier.strip()
        if not clean_id:
            raise ValueError("Must specify client ID or name to disable")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        c = await find_client_by_name_or_id(self.pool, principal_id=p["id"], identifier=clean_id)
        if c is None:
            raise ValueError(f"Active client not found: {clean_id}")

        disabled = await db_disable_client(self.pool, principal_id=p["id"], client_id=c["id"])
        if disabled is None:
            raise ValueError(f"Client already disabled or not found: {clean_id}")

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            client_id=disabled["id"],
            event_type="client.disabled",
            payload={
                "name": disabled["name"],
                "kind": disabled["kind"],
                "revoked_tokens": disabled["revoked_tokens"],
            },
        )
        return disabled

    async def set_policy(
        self,
        *,
        pattern: str,
        decision: str,
        client_identifier: str | None = None,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        clean_pattern = pattern.strip()
        if not clean_pattern:
            raise ValueError("Tool pattern must be non-empty")
        if decision not in ("allow", "deny", "require_approval"):
            raise ValueError("decision must be one of: allow, deny, require_approval")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        client_id: UUID | None = None
        scope_label = "*"
        if client_identifier:
            c = await find_client_by_name_or_id(
                self.pool, principal_id=p["id"], identifier=client_identifier
            )
            if c is None:
                raise ValueError(f"Client not found: {client_identifier}")
            client_id = c["id"]
            scope_label = c["name"]

        row = await set_client_policy(
            self.pool,
            principal_id=p["id"],
            client_id=client_id,
            tool_pattern=clean_pattern,
            decision=decision,
        )
        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            client_id=client_id,
            event_type="policy.changed",
            tool=clean_pattern,
            decision=decision,
            payload={"action": "set", "client": scope_label},
        )
        return row

    async def clear_policy(
        self,
        *,
        pattern: str,
        client_identifier: str | None = None,
        principal_slug: str | None = None,
    ) -> bool:
        clean_pattern = pattern.strip()
        if not clean_pattern:
            raise ValueError("Tool pattern must be non-empty")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        client_id: UUID | None = None
        scope_label = "*"
        if client_identifier:
            c = await find_client_by_name_or_id(
                self.pool, principal_id=p["id"], identifier=client_identifier
            )
            if c is None:
                raise ValueError(f"Client not found: {client_identifier}")
            client_id = c["id"]
            scope_label = c["name"]

        deleted = await clear_client_policy(
            self.pool,
            principal_id=p["id"],
            client_id=client_id,
            tool_pattern=clean_pattern,
        )
        if deleted:
            await record_event(
                self.pool,
                principal_id=p["id"],
                space_id=sp["id"],
                client_id=client_id,
                event_type="policy.changed",
                tool=clean_pattern,
                decision="cleared",
                payload={"action": "cleared", "client": scope_label},
            )
        return deleted

    async def add_actuator(
        self,
        *,
        slug: str,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        if not slug or "_" in slug or not slug.replace("-", "").isalnum():
            raise ValueError("actuator slug must match [a-zA-Z0-9-]+ with no underscores")
        p, sp = await get_principal_and_space(self.pool, principal_slug)
        res = await create_actuator_provider_and_token(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            slug=slug,
        )
        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=res["provider"]["id"],
            event_type="token.issued",
            payload={"token_id": str(res["token_id"]), "kind": "actuator", "slug": slug},
        )
        return res

    async def set_provider_backend(
        self,
        *,
        slug: str,
        backend: str,
        principal_slug: str | None = None,
        on_switch: Callable[[str, str], Awaitable[Any]] | None = None,
    ) -> dict[str, Any]:
        clean_slug = slug.strip().lower()
        clean_backend = backend.strip().lower()
        allowed = SWITCHABLE_PROVIDER_BACKENDS.get(clean_slug)
        if allowed is None:
            supported = ", ".join(sorted(SWITCHABLE_PROVIDER_BACKENDS))
            raise ValueError(f"Provider '{clean_slug}' does not support backend switching (supported: {supported})")
        if clean_backend not in allowed:
            raise ValueError(
                f"Invalid backend '{clean_backend}' for '{clean_slug}' (allowed: {', '.join(allowed)})"
            )

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        res = await set_provider_backend(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            slug=clean_slug,
            backend=clean_backend,
            default_config=_DEFAULT_PROVIDER_CONFIGS.get(clean_slug),
        )
        if on_switch is not None:
            await on_switch(clean_slug, clean_backend)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=res["id"],
            event_type="provider.backend_changed",
            outcome="ok",
            payload={
                "slug": clean_slug,
                "from": res["previous_backend"],
                "to": clean_backend,
                "hot_reloaded": on_switch is not None,
            },
        )
        return res

    async def record_backup(
        self,
        *,
        filename: str,
        size_bytes: int,
        completed_at: str | None = None,
        retained_count: int | None = None,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        clean_name = filename.strip()
        if not clean_name:
            raise ValueError("filename must be non-empty")
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        payload: dict[str, Any] = {
            "filename": clean_name,
            "size_bytes": int(size_bytes),
        }
        if completed_at:
            payload["completed_at"] = completed_at.strip()
        if retained_count is not None:
            payload["retained_count"] = int(retained_count)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            event_type="backup.completed",
            outcome="ok",
            payload=payload,
        )
        return payload

    async def set_tool_tier(
        self,
        *,
        pattern: str,
        tier: str,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        clean_pattern = pattern.strip()
        clean_tier = tier.strip().lower()
        if not clean_pattern:
            raise ValueError("Tool pattern must be non-empty")
        if clean_tier not in ("read", "write", "external"):
            raise ValueError("tier must be one of: read, write, external")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        row = await set_tool_tier(
            self.pool,
            principal_id=p["id"],
            tool_pattern=clean_pattern,
            tier=clean_tier,
        )
        await unmark_managed_tier_pattern(
            self.pool,
            principal_id=p["id"],
            tool_pattern=clean_pattern,
        )
        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            event_type="tier.changed",
            tool=clean_pattern,
            decision=clean_tier,
            outcome="ok",
            payload={"tool_pattern": clean_pattern, "tier": clean_tier},
        )
        return row

    async def add_remote_upstream(
        self,
        *,
        slug: str,
        url: str,
        auth_mode: str,
        default_tier: str = "external",
        extra_config: dict[str, Any] | None = None,
        principal_slug: str | None = None,
        upstream_manager: UpstreamManager | None = None,
    ) -> dict[str, Any]:
        clean_slug = validate_upstream_slug(slug)
        if clean_slug in BUILTIN_PROVIDER_SLUGS:
            raise ValueError(f"Slug '{clean_slug}' is reserved for a built-in provider")
        clean_url = validate_upstream_url(url)
        clean_auth = auth_mode.strip().lower()
        if clean_auth not in ("none", "bearer", "oauth"):
            raise ValueError("auth_mode must be one of: none, bearer, oauth")
        clean_tier = default_tier.strip().lower()
        if clean_tier not in ("read", "write", "external"):
            raise ValueError("default_tier must be one of: read, write, external")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        row = await create_remote_upstream(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            slug=clean_slug,
            url=clean_url,
            auth_mode=clean_auth,
            default_tier=clean_tier,
            extra_config=extra_config,
        )
        if upstream_manager is not None:
            await upstream_manager.hot_register_from_row(row, connect=True)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=row["id"],
            event_type="upstream.added",
            outcome="ok",
            payload={
                "slug": clean_slug,
                "url": clean_url,
                "auth_mode": clean_auth,
                "default_tier": clean_tier,
            },
        )
        return row

    async def install_catalog_connector(
        self,
        *,
        catalog_slug: str,
        as_slug: str | None = None,
        catalog_dir: Path | None = None,
        principal_slug: str | None = None,
        upstream_manager: UpstreamManager | None = None,
    ) -> dict[str, Any]:
        manifest = get_catalog_entry(catalog_slug, catalog_dir)
        if manifest is None:
            raise ValueError(f"Catalog connector not found: {catalog_slug}")

        target_slug = validate_upstream_slug(as_slug) if (as_slug and as_slug.strip()) else manifest.slug
        extra_config: dict[str, Any] = {
            "catalog_slug": manifest.slug,
            "manifest_version": manifest.manifest_version,
            "manifest_tiers": {
                "default": manifest.tiers.default,
                "patterns": dict(manifest.tiers.patterns),
            },
            "requires_client_registration": manifest.auth.requires_client_registration,
            "managed_tier_patterns": [],
        }
        if manifest.auth.token_help:
            extra_config["token_help"] = manifest.auth.token_help
        if manifest.auth.client_help:
            extra_config["client_help"] = manifest.auth.client_help

        row = await self.add_remote_upstream(
            slug=target_slug,
            url=manifest.runtime.url,
            auth_mode=manifest.auth.mode,
            default_tier=manifest.tiers.default,
            extra_config=extra_config,
            principal_slug=principal_slug,
            upstream_manager=upstream_manager,
        )
        p, sp = await get_principal_and_space(self.pool, principal_slug)
        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=row["id"],
            event_type="catalog.installed",
            outcome="ok",
            payload={
                "catalog_slug": manifest.slug,
                "slug": target_slug,
                "auth_mode": manifest.auth.mode,
                "default_tier": manifest.tiers.default,
            },
        )
        return {"manifest": manifest, "provider": row}

    async def set_upstream_oauth_client(
        self,
        *,
        slug: str,
        client_id: str,
        client_secret: str,
        secret_key: str,
        base_url: str,
        principal_slug: str | None = None,
        upstream_manager: UpstreamManager | None = None,
    ) -> dict[str, Any]:
        """Store pre-registered OAuth client_id + client_secret in vault for non-DCR upstreams."""
        clean_slug = slug.strip().lower()
        clean_cid = client_id.strip()
        clean_csec = client_secret.strip()
        if not clean_cid or not clean_csec:
            raise ValueError("Both client_id and client_secret are required")
        if not secret_key:
            raise ValueError("ZOS_SECRET_KEY is not configured; cannot encrypt OAuth client credentials")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        row = await get_remote_upstream(self.pool, slug=clean_slug, principal_id=p["id"])
        if row is None:
            raise ValueError(f"Remote MCP upstream not found: {clean_slug}")
        cfg = row["config"] if isinstance(row.get("config"), dict) else {}
        if str(cfg.get("auth_mode") or "").lower() != "oauth":
            raise ValueError(f"Upstream '{clean_slug}' is not an OAuth upstream")

        store = CredentialStore(self.pool, CredentialVault(secret_key))
        storage = VaultOAuthTokenStorage(
            store=store,
            principal_id=p["id"],
            slug=clean_slug,
            credential_id=row.get("credential_id"),
        )
        callback_url = f"{base_url.rstrip('/')}/connect/{clean_slug}/callback"
        client_info = OAuthClientInformationFull(
            client_id=clean_cid,
            client_secret=clean_csec,
            client_name=f"ZOS ({clean_slug})",
            redirect_uris=[AnyUrl(callback_url)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_post",
        )
        await storage.set_client_info(client_info)
        assert storage.credential_id is not None
        await attach_upstream_credential(
            self.pool,
            provider_id=row["id"],
            credential_id=storage.credential_id,
        )
        if upstream_manager is not None:
            refreshed = await get_remote_upstream(self.pool, slug=clean_slug, principal_id=p["id"])
            if refreshed is not None:
                await upstream_manager.hot_register_from_row(refreshed, connect=False)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=row["id"],
            event_type="upstream.oauth_client_set",
            outcome="ok",
            payload={"slug": clean_slug, "credential_id": str(storage.credential_id)},
        )
        return {"slug": clean_slug, "credential_id": storage.credential_id}

    async def set_upstream_bearer_token(
        self,
        *,
        slug: str,
        token: str,
        secret_key: str,
        principal_slug: str | None = None,
        upstream_manager: UpstreamManager | None = None,
    ) -> dict[str, Any]:
        clean_slug = slug.strip().lower()
        clean_token = token.strip()
        if not clean_token:
            raise ValueError("Token must be non-empty")
        if not secret_key:
            raise ValueError("ZOS_SECRET_KEY is not configured; cannot encrypt bearer token")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        row = await get_remote_upstream(self.pool, slug=clean_slug, principal_id=p["id"])
        if row is None:
            raise ValueError(f"Remote MCP upstream not found: {clean_slug}")
        cfg = row["config"] if isinstance(row.get("config"), dict) else {}
        if str(cfg.get("auth_mode") or "").lower() != "bearer":
            raise ValueError(
                f"Upstream '{clean_slug}' uses auth_mode='{cfg.get('auth_mode')}', not 'bearer'"
            )

        store = CredentialStore(self.pool, CredentialVault(secret_key))
        cred_id = await store.save_credential(
            principal_id=p["id"],
            issuer=ISSUER_MCP_BEARER,
            account=clean_slug,
            payload={"access_token": clean_token},
        )
        await attach_upstream_credential(
            self.pool,
            provider_id=row["id"],
            credential_id=cred_id,
        )
        if upstream_manager is not None:
            await upstream_manager.hot_update_bearer_token(clean_slug, cred_id)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=row["id"],
            event_type="upstream.token_set",
            outcome="ok",
            payload={"slug": clean_slug, "credential_id": str(cred_id)},
        )
        return {"slug": clean_slug, "provider_id": row["id"], "credential_id": cred_id}

    async def disconnect_remote_upstream(
        self,
        *,
        slug: str,
        principal_slug: str | None = None,
        upstream_manager: UpstreamManager | None = None,
    ) -> dict[str, Any]:
        clean_slug = slug.strip().lower()
        p, sp = await get_principal_and_space(self.pool, principal_slug)
        row = await get_remote_upstream(self.pool, slug=clean_slug, principal_id=p["id"])
        if row is None:
            raise ValueError(f"Remote MCP upstream not found: {clean_slug}")

        old_cred_id = await clear_upstream_credential(self.pool, provider_id=row["id"])
        if old_cred_id is not None:
            async with self.pool.connection() as conn:
                await conn.execute("DELETE FROM credentials WHERE id = %s", (old_cred_id,))
                await conn.commit()

        if upstream_manager is not None:
            await upstream_manager.hot_disconnect(clean_slug)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=row["id"],
            event_type="upstream.disconnected",
            outcome="ok",
            payload={"slug": clean_slug},
        )
        return {"slug": clean_slug, "provider_id": row["id"]}

    async def remove_remote_upstream(
        self,
        *,
        slug: str,
        principal_slug: str | None = None,
        upstream_manager: UpstreamManager | None = None,
    ) -> dict[str, Any]:
        clean_slug = slug.strip().lower()
        p, sp = await get_principal_and_space(self.pool, principal_slug)
        existing = await get_remote_upstream(self.pool, slug=clean_slug, principal_id=p["id"])
        if existing is not None:
            ecfg = existing.get("config") if isinstance(existing.get("config"), dict) else {}
            if ecfg.get("is_module"):
                raise ValueError(
                    f"'{clean_slug}' is a managed module; uninstall it from /admin/modules or `zos module uninstall {clean_slug}`"
                )
        deleted = await delete_remote_upstream(self.pool, principal_id=p["id"], slug=clean_slug)
        if deleted is None:
            raise ValueError(f"Remote MCP upstream not found: {clean_slug}")

        if upstream_manager is not None:
            await upstream_manager.hot_remove(clean_slug)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            event_type="upstream.removed",
            outcome="ok",
            payload={"slug": clean_slug, "deleted_provider_id": str(deleted["id"])},
        )
        return deleted

    async def list_remote_upstreams(
        self,
        *,
        upstream_manager: UpstreamManager | None = None,
        principal_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        p, _sp = await get_principal_and_space(self.pool, principal_slug)
        rows = await list_remote_upstreams(self.pool, principal_id=p["id"])
        items: list[dict[str, Any]] = []
        for row in rows:
            slug = str(row["slug"])
            cfg = row["config"] if isinstance(row.get("config"), dict) else {}
            if cfg.get("is_module"):
                continue
            auth_mode = str(cfg.get("auth_mode") or "none").lower()
            has_cred = row.get("credential_id") is not None
            rt = upstream_manager.get_runtime_info(slug) if upstream_manager is not None else None
            if rt is not None:
                status = rt["status"]
                online = rt["online"]
                tool_count = rt["tool_count"]
                last_error = rt["last_error"]
            else:
                online = False
                tool_count = 0
                last_error = None
                if auth_mode in ("bearer", "oauth") and not has_cred:
                    status = "not_configured"
                else:
                    status = "offline"
            items.append(
                {
                    "id": row["id"],
                    "slug": slug,
                    "url": str(cfg.get("url") or ""),
                    "auth_mode": auth_mode,
                    "default_tier": str(cfg.get("default_tier") or "external"),
                    "catalog_slug": cfg.get("catalog_slug"),
                    "requires_client_registration": bool(cfg.get("requires_client_registration", False)),
                    "token_help": str(cfg.get("token_help") or ""),
                    "client_help": str(cfg.get("client_help") or ""),
                    "has_credential": has_cred,
                    "status": status,
                    "online": online,
                    "tool_count": tool_count,
                    "last_error": last_error,
                    "last_seen_at": row["last_seen_at"],
                }
            )
        return items

    # ── Read Queries (for /admin pages) ───────────────────────────────────

    async def get_overview(
        self,
        registry: ProviderRegistry,
        sessions: SessionTracker,
    ) -> dict[str, Any]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT slug, kind, enabled, last_seen_at, config
                FROM providers
                ORDER BY slug ASC
                """
            )
            db_providers = {row["slug"]: dict(row) for row in await cur.fetchall()}

            cur = await conn.execute(
                """
                SELECT issuer, account, scopes, access_expires_at, updated_at
                FROM credentials
                WHERE issuer = 'google'
                ORDER BY updated_at DESC
                LIMIT 1
                """
            )
            cred_row = await cur.fetchone()

            cur = await conn.execute(
                """
                SELECT created_at, payload
                FROM events
                WHERE type = 'backup.completed' AND outcome = 'ok'
                ORDER BY id DESC
                LIMIT 1
                """
            )
            backup_row = await cur.fetchone()

            cur = await conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM pending_approvals
                WHERE status = 'pending' AND expires_at > now()
                """
            )
            pending_approvals_row = await cur.fetchone()

        providers: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in registry.all_providers():
            seen.add(p.namespace)
            db_row = db_providers.get(p.namespace)
            providers.append(
                {
                    "namespace": p.namespace,
                    "kind": p.kind,
                    "backend": getattr(p, "backend", None),
                    "switchable_backends": list(SWITCHABLE_PROVIDER_BACKENDS.get(p.namespace, ())),
                    "online": p.is_online(),
                    "last_seen_at": db_row["last_seen_at"] if db_row else None,
                }
            )
        for slug, db_row in sorted(db_providers.items()):
            if slug in seen:
                continue
            cfg = db_row.get("config") or {}
            providers.append(
                {
                    "namespace": slug,
                    "kind": db_row["kind"],
                    "backend": cfg.get("backend") if isinstance(cfg, dict) else None,
                    "switchable_backends": list(SWITCHABLE_PROVIDER_BACKENDS.get(slug, ())),
                    "online": False,
                    "last_seen_at": db_row["last_seen_at"],
                }
            )

        google_credential = None
        if cred_row is not None:
            google_credential = {
                "issuer": cred_row["issuer"],
                "account": cred_row["account"],
                "scopes": list(cred_row["scopes"] or []),
                "access_expires_at": cred_row["access_expires_at"],
                "updated_at": cred_row["updated_at"],
            }

        last_backup = None
        if backup_row is not None:
            bpay = backup_row["payload"] if isinstance(backup_row["payload"], dict) else {}
            last_backup = {
                "filename": bpay.get("filename"),
                "size_bytes": int(bpay["size_bytes"]) if bpay.get("size_bytes") is not None else None,
                "completed_at": bpay.get("completed_at") or backup_row["created_at"],
                "retained_count": bpay.get("retained_count"),
            }

        return {
            "active_sessions": sessions.count,
            "providers": providers,
            "google_credential": google_credential,
            "last_backup": last_backup,
            "pending_approvals_count": int(pending_approvals_row["cnt"]) if pending_approvals_row is not None else 0,
        }

    async def list_clients(self, *, include_disabled: bool = False) -> list[dict[str, Any]]:
        where_clause = "" if include_disabled else "WHERE c.disabled_at IS NULL"
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT
                    c.id,
                    c.name,
                    c.kind,
                    c.oauth_client_id,
                    c.trust_ceiling,
                    c.created_at,
                    c.last_seen_at,
                    c.disabled_at,
                    COUNT(t.id) FILTER (
                        WHERE t.revoked_at IS NULL
                          AND (t.expires_at IS NULL OR t.expires_at > now())
                          AND t.kind IN ('bearer', 'access', 'refresh')
                    ) AS active_token_count
                FROM clients c
                LEFT JOIN tokens t ON t.client_id = c.id
                {where_clause}
                GROUP BY c.id
                ORDER BY c.created_at DESC
                """
            )
            return [dict(row) for row in await cur.fetchall()]

    async def get_policy_matrix(self, registry: ProviderRegistry) -> dict[str, Any]:
        clients = await self.list_clients()
        try:
            p, _ = await get_principal_and_space(self.pool)
            principal_id: UUID | None = p["id"]
        except ValueError:
            principal_id = None

        tiers: dict[str, Tier] = {}
        global_rules: dict[str, Decision] = {}
        client_rules_by_id: dict[UUID, dict[str, Decision]] = {}

        if principal_id is not None:
            async with self.pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT tool_pattern, tier FROM tool_tiers WHERE principal_id = %s",
                    (principal_id,),
                )
                tiers = {row["tool_pattern"]: row["tier"] for row in await cur.fetchall()}

                cur = await conn.execute(
                    "SELECT client_id, tool_pattern, decision FROM client_policies WHERE principal_id = %s",
                    (principal_id,),
                )
                for row in await cur.fetchall():
                    cid = row["client_id"]
                    if cid is None:
                        global_rules[row["tool_pattern"]] = row["decision"]
                    else:
                        client_rules_by_id.setdefault(cid, {})[row["tool_pattern"]] = row["decision"]

        # Build merged tool catalog from online providers + zos_status + known tier seeds
        merged_tools: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        def _add_tool(
            exposed_name: str,
            provider_slug: str,
            online: bool,
            destructive_hint: bool | None = None,
            idempotent_hint: bool | None = None,
        ) -> None:
            if exposed_name in seen_names:
                return
            seen_names.add(exposed_name)
            tier = resolve_tier(exposed_name, tiers)
            ann = build_tool_annotations(
                tier,
                destructive_hint=destructive_hint,
                idempotent_hint=idempotent_hint,
            )
            global_override = global_rules.get(exposed_name)
            per_client: list[dict[str, Any]] = []
            for c in clients:
                c_rules = client_rules_by_id.get(c["id"], {})
                eff = resolve_decision(
                    exposed_name,
                    tier=tier,
                    trust_ceiling=c["trust_ceiling"],
                    client_rules=c_rules,
                    global_rules=global_rules,
                )
                per_client.append(
                    {
                        "client_id": c["id"],
                        "client_name": c["name"],
                        "effective": eff,
                        "explicit": c_rules.get(exposed_name),
                    }
                )
            merged_tools.append(
                {
                    "name": exposed_name,
                    "provider": provider_slug,
                    "online": online,
                    "tier": tier,
                    "read_only_hint": bool(ann.read_only_hint),
                    "destructive_hint": bool(ann.destructive_hint),
                    "open_world_hint": bool(ann.open_world_hint),
                    "global_override": global_override,
                    "per_client": per_client,
                }
            )

        _add_tool(ZOS_STATUS_TOOL_NAME, "zos", True, destructive_hint=False, idempotent_hint=True)

        for provider in registry.online_providers():
            try:
                ptools = await provider.list_tools()
            except Exception:
                ptools = []
            for pt in ptools:
                exposed = format_tool_name(provider.namespace, pt.name)
                _add_tool(
                    exposed,
                    provider.namespace,
                    True,
                    destructive_hint=pt.destructive_hint,
                    idempotent_hint=pt.idempotent_hint,
                )

        # Include seeded offline tools so owner can pre-configure policies even while offline
        for pattern in sorted(tiers):
            if "*" not in pattern and "?" not in pattern and pattern not in seen_names:
                prefix = pattern.split("_", 1)[0] if "_" in pattern else "zos"
                _add_tool(pattern, prefix, False)

        # Collect any wildcard / custom rules so they remain visible and clearable
        custom_rules: list[dict[str, Any]] = []
        client_names = {c["id"]: c["name"] for c in clients}
        for pat, dec in sorted(global_rules.items()):
            if pat not in seen_names:
                custom_rules.append({"client_id": None, "client_name": "*", "pattern": pat, "decision": dec})
        for cid, rules in client_rules_by_id.items():
            for pat, dec in sorted(rules.items()):
                if pat not in seen_names:
                    custom_rules.append(
                        {
                            "client_id": cid,
                            "client_name": client_names.get(cid, str(cid)),
                            "pattern": pat,
                            "decision": dec,
                        }
                    )

        return {
            "clients": clients,
            "tools": merged_tools,
            "custom_rules": custom_rules,
        }

    async def list_events(
        self,
        *,
        before_id: int | None = None,
        client_id: UUID | None = None,
        provider_id: UUID | None = None,
        event_type: str | None = None,
        outcome: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        clamped_limit = max(1, min(int(limit), 100))
        clauses: list[str] = []
        params: list[Any] = []

        if before_id is not None:
            clauses.append("e.id < %s")
            params.append(before_id)
        if client_id is not None:
            clauses.append("e.client_id = %s")
            params.append(client_id)
        if provider_id is not None:
            clauses.append("e.provider_id = %s")
            params.append(provider_id)
        if event_type:
            clauses.append("e.type = %s")
            params.append(event_type)
        if outcome:
            clauses.append("e.outcome = %s")
            params.append(outcome)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(clamped_limit + 1)

        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT
                    e.id,
                    e.created_at,
                    e.type,
                    e.tool,
                    e.decision,
                    e.outcome,
                    e.args_digest,
                    e.client_id,
                    c.name AS client_name,
                    e.provider_id,
                    pr.slug AS provider_slug
                FROM events e
                LEFT JOIN clients c ON c.id = e.client_id
                LEFT JOIN providers pr ON pr.id = e.provider_id
                {where_sql}
                ORDER BY e.id DESC
                LIMIT %s
                """,
                params,
            )
            rows = [dict(r) for r in await cur.fetchall()]

            cur = await conn.execute("SELECT id, name FROM clients ORDER BY name ASC")
            filter_clients = [dict(r) for r in await cur.fetchall()]

            cur = await conn.execute("SELECT id, slug FROM providers ORDER BY slug ASC")
            filter_providers = [dict(r) for r in await cur.fetchall()]

            cur = await conn.execute("SELECT DISTINCT type FROM events ORDER BY type ASC")
            filter_types = [str(r["type"]) for r in await cur.fetchall()]

        has_more = len(rows) > clamped_limit
        page = rows[:clamped_limit]
        for row in page:
            digest = row["args_digest"]
            row["args_digest_hex"] = bytes(digest).hex() if digest is not None else ""

        next_before_id = page[-1]["id"] if has_more and page else None
        return {
            "events": page,
            "next_before_id": next_before_id,
            "filter_clients": filter_clients,
            "filter_providers": filter_providers,
            "filter_types": filter_types,
        }
