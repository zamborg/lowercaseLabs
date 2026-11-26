"""Module lifecycle orchestration: catalog -> storage -> host -> UpstreamManager -> audit."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from zos.audit import record_event
from zos.catalog import ModuleManifest, get_module_catalog_entry
from zos.config import Settings
from zos.db.queries import (
    apply_manifest_tiers_for_tools,
    create_remote_upstream,
    delete_remote_upstream,
    get_principal_and_space,
    get_remote_upstream,
)
from zos.modules.host import InstalledModule, ModuleHost, create_module_host
from zos.modules.storage import ModuleStorageProvisioner
from zos.providers.upstream_manager import UpstreamManager
from zos.router.naming import BUILTIN_PROVIDER_SLUGS, validate_upstream_slug
from zos.vault import CredentialStore, CredentialVault

logger = logging.getLogger(__name__)


def validate_module_url(url: str) -> str:
    """Validate an internal or remote module MCP/UI URL (`http://` allowed for container/loopback)."""
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid module URL {url!r}: must be a valid http(s):// URL")
    return cleaned


class ModuleManager:
    """Manages module installation, attach, enable/disable, upgrade, and uninstall."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        settings: Settings,
        *,
        host: ModuleHost | None = None,
        upstream_manager: UpstreamManager | None = None,
        modules_dir: Path | None = None,
    ) -> None:
        self.pool = pool
        self.settings = settings
        self.host: ModuleHost = host or create_module_host(settings)
        self.upstream_manager = upstream_manager
        self.modules_dir = modules_dir or settings.modules_catalog_dir
        self._store: CredentialStore | None = (
            CredentialStore(pool, CredentialVault(settings.zos_secret_key))
            if settings.zos_secret_key
            else None
        )
        self.storage = ModuleStorageProvisioner(pool, self._store)

    async def get_module_secret(self, slug: str, *, principal_id: UUID) -> str | None:
        return await self.storage.get_module_secret(principal_id=principal_id, slug=slug)

    async def get_module(
        self,
        slug: str,
        *,
        principal_slug: str | None = None,
    ) -> dict[str, Any] | None:
        p, _sp = await get_principal_and_space(self.pool, principal_slug)
        clean = slug.strip().lower()
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, principal_id, space_id, slug, manifest_version, version,
                       image_pin, host, status, mcp_url, ui_url, ui_title,
                       upstream_id, config, installed_at, disabled_at
                FROM modules
                WHERE principal_id = %s AND slug = %s
                """,
                (p["id"], clean),
            )
            row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list_modules(
        self,
        *,
        principal_slug: str | None = None,
        include_runtime_status: bool = False,
    ) -> list[dict[str, Any]]:
        p, _sp = await get_principal_and_space(self.pool, principal_slug)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, principal_id, space_id, slug, manifest_version, version,
                       image_pin, host, status, mcp_url, ui_url, ui_title,
                       upstream_id, config, installed_at, disabled_at
                FROM modules
                WHERE principal_id = %s
                ORDER BY slug ASC
                """,
                (p["id"],),
            )
            rows = [dict(r) for r in await cur.fetchall()]

        items: list[dict[str, Any]] = []
        for row in rows:
            slug = str(row["slug"])
            rt = (
                self.upstream_manager.get_runtime_info(slug)
                if self.upstream_manager is not None
                else None
            )
            row["online"] = bool(rt["online"]) if rt is not None else False
            row["tool_count"] = int(rt["tool_count"]) if rt is not None else 0
            row["upstream_status"] = rt["status"] if rt is not None else (
                "disabled" if row["disabled_at"] is not None else row["status"]
            )
            row["last_error"] = rt["last_error"] if rt is not None else None
            if include_runtime_status:
                h_status = await self.host.status(slug)
                row["host_state"] = h_status.state
            items.append(row)
        return items

    async def _connect_with_retry(self, slug: str, *, attempts: int = 20, delay_s: float = 0.5) -> None:
        if self.upstream_manager is None:
            return
        provider = self.upstream_manager.get_provider(slug)
        if provider is None:
            return
        for _ in range(attempts):
            if provider.is_online():
                return
            with contextlib.suppress(Exception):
                ok = await provider.ensure_connected(broadcast_on_change=False)
                if ok:
                    return
            await asyncio.sleep(delay_s)

    def _provider_extra_config(self, manifest: ModuleManifest) -> dict[str, Any]:
        return {
            "is_module": True,
            "module_slug": manifest.slug,
            "catalog_slug": manifest.slug,
            "manifest_tiers": {
                "default": manifest.tiers.default,
                "patterns": dict(manifest.tiers.patterns),
            },
            "managed_tier_patterns": [],
        }

    def _module_config_meta(self, manifest: ModuleManifest) -> dict[str, Any]:
        return {
            "storage_postgres": manifest.storage.postgres,
            "egress": manifest.network.egress,
            "mcp_port": manifest.runtime.mcp_port,
            "mcp_path": manifest.runtime.mcp_path,
            "ui_port": manifest.runtime.ui_port,
            "ui_path": manifest.ui.path if manifest.ui is not None else "/",
            "default_tier": manifest.tiers.default,
            "manifest_tiers": {
                "default": manifest.tiers.default,
                "patterns": dict(manifest.tiers.patterns),
            },
        }

    def _validate_config_values(
        self,
        manifest: ModuleManifest,
        config_values: dict[str, str] | None,
    ) -> dict[str, str]:
        provided = {str(k).strip(): str(v) for k, v in (config_values or {}).items() if str(k).strip()}
        resolved: dict[str, str] = {}
        allowed_names = {f.name for f in manifest.config}
        unknown = set(provided) - allowed_names
        if unknown:
            raise ValueError(f"Unknown config field(s) for '{manifest.slug}': {', '.join(sorted(unknown))}")
        for field in manifest.config:
            val = provided.get(field.name)
            if val is None or val == "":
                if field.default is not None:
                    val = field.default
                elif field.required:
                    raise ValueError(f"Missing required config field '{field.name}' for module '{manifest.slug}'")
                else:
                    continue
            resolved[field.name] = val
        return resolved

    async def _seed_exact_manifest_tiers(
        self,
        *,
        principal_id: UUID,
        provider_id: UUID,
        manifest: ModuleManifest,
    ) -> None:
        """Seed exact (non-wildcard) manifest tier entries immediately without overwriting owner edits."""
        exact_tools: list[tuple[str, str]] = []
        prefix = f"{manifest.slug}_"
        for pattern in manifest.tiers.patterns:
            if any(ch in pattern for ch in "*?[]"):
                continue
            local = pattern[len(prefix) :] if pattern.startswith(prefix) else pattern
            exact_tools.append((pattern, local))
        if not exact_tools:
            return
        await apply_manifest_tiers_for_tools(
            self.pool,
            principal_id=principal_id,
            provider_id=provider_id,
            slug=manifest.slug,
            catalog_slug=manifest.slug,
            patterns=dict(manifest.tiers.patterns),
            exposed_tools=exact_tools,
        )

    async def install(
        self,
        slug: str,
        *,
        config_values: dict[str, str] | None = None,
        principal_slug: str | None = None,
        wait_ready: bool = True,
    ) -> dict[str, Any]:
        """Install a catalog v2 module: provision storage/secret, launch via host, register upstream."""
        clean_slug = validate_upstream_slug(slug)
        if clean_slug in BUILTIN_PROVIDER_SLUGS:
            raise ValueError(f"Module slug '{clean_slug}' collides with a built-in provider")

        manifest = get_module_catalog_entry(clean_slug, self.modules_dir)
        if manifest is None:
            raise ValueError(f"Unknown module in catalog: '{clean_slug}'")

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        existing = await self.get_module(clean_slug, principal_slug=principal_slug)
        if existing is not None:
            raise ValueError(f"Module '{clean_slug}' is already installed")

        resolved_cfg = self._validate_config_values(manifest, config_values)
        module_secret = await self.storage.ensure_module_secret(
            principal_id=p["id"],
            slug=clean_slug,
        )
        if resolved_cfg:
            await self.storage.save_module_config(
                principal_id=p["id"],
                slug=clean_slug,
                values=resolved_cfg,
            )

        env: dict[str, str] = {"ZOS_MODULE_SECRET": module_secret}
        if manifest.storage.postgres:
            mod_db_url = await self.storage.provision_postgres(
                principal_id=p["id"],
                slug=clean_slug,
                base_database_url=self.settings.database_url,
            )
            env["ZOS_MODULE_DATABASE_URL"] = mod_db_url

        installed: InstalledModule = await self.host.install(manifest, resolved_cfg, env=env)

        upstream_row: dict[str, Any] | None = None
        status = "installed" if installed.mcp_url else "pending_attach"
        if installed.mcp_url:
            mcp_url = validate_module_url(installed.mcp_url)
            upstream_row = await create_remote_upstream(
                self.pool,
                principal_id=p["id"],
                space_id=sp["id"],
                slug=clean_slug,
                url=mcp_url,
                auth_mode="none",
                default_tier=manifest.tiers.default,
                extra_config=self._provider_extra_config(manifest),
            )
        else:
            mcp_url = None

        ui_title = (
            (manifest.ui.title or manifest.name)
            if manifest.ui is not None or installed.ui_url
            else None
        )
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO modules (
                    principal_id, space_id, slug, manifest_version, version,
                    image_pin, host, status, mcp_url, ui_url, ui_title,
                    upstream_id, config
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, principal_id, space_id, slug, manifest_version, version,
                          image_pin, host, status, mcp_url, ui_url, ui_title,
                          upstream_id, config, installed_at, disabled_at
                """,
                (
                    p["id"],
                    sp["id"],
                    clean_slug,
                    manifest.manifest_version,
                    manifest.version,
                    installed.image_pin,
                    installed.host,
                    status,
                    mcp_url,
                    installed.ui_url,
                    ui_title,
                    upstream_row["id"] if upstream_row is not None else None,
                    Jsonb(self._module_config_meta(manifest)),
                ),
            )
            mod_row = await cur.fetchone()
            await conn.commit()
        assert mod_row is not None

        if upstream_row is not None:
            await self._seed_exact_manifest_tiers(
                principal_id=p["id"],
                provider_id=upstream_row["id"],
                manifest=manifest,
            )

        if upstream_row is not None and self.upstream_manager is not None:
            await self.upstream_manager.hot_register_from_row(upstream_row, connect=False)
            if wait_ready:
                await self._connect_with_retry(clean_slug)
                await self.upstream_manager.sessions.broadcast_tools_changed()

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=upstream_row["id"] if upstream_row is not None else None,
            event_type="module.installed",
            outcome="ok",
            payload={
                "slug": clean_slug,
                "version": manifest.version,
                "host": installed.host,
                "image_pin": installed.image_pin,
                "status": status,
            },
        )
        return {
            "module": dict(mod_row),
            "manifest": manifest,
            "installed": installed,
            "instructions": installed.instructions,
        }

    async def attach(
        self,
        slug: str,
        *,
        mcp_url: str,
        ui_url: str | None = None,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        """Attach an externally started (manual) module's MCP/UI URL and register it as an upstream."""
        clean_slug = validate_upstream_slug(slug)
        clean_mcp = validate_module_url(mcp_url)
        clean_ui = validate_module_url(ui_url) if ui_url and ui_url.strip() else None

        p, sp = await get_principal_and_space(self.pool, principal_slug)
        mod = await self.get_module(clean_slug, principal_slug=principal_slug)
        manifest = get_module_catalog_entry(clean_slug, self.modules_dir)

        if mod is None and manifest is None:
            raise ValueError(f"Module '{clean_slug}' is not installed and not found in catalog")

        default_tier = manifest.tiers.default if manifest is not None else "external"
        extra_cfg = (
            self._provider_extra_config(manifest)
            if manifest is not None
            else {"is_module": True, "module_slug": clean_slug}
        )

        # Ensure per-module secret exists
        await self.storage.ensure_module_secret(principal_id=p["id"], slug=clean_slug)

        existing_up = await get_remote_upstream(self.pool, slug=clean_slug, principal_id=p["id"])
        if existing_up is not None:
            new_cfg = dict(existing_up.get("config") or {})
            new_cfg.update({"url": clean_mcp, "auth_mode": "none", **extra_cfg})
            async with self.pool.connection() as conn:
                cur = await conn.execute(
                    """
                    UPDATE providers
                    SET config = %s, enabled = true
                    WHERE id = %s
                    RETURNING id, principal_id, space_id, slug, kind, credential_id, config, enabled, last_seen_at, created_at
                    """,
                    (Jsonb(new_cfg), existing_up["id"]),
                )
                upstream_row = dict(await cur.fetchone() or {})
                await conn.commit()
        else:
            upstream_row = await create_remote_upstream(
                self.pool,
                principal_id=p["id"],
                space_id=sp["id"],
                slug=clean_slug,
                url=clean_mcp,
                auth_mode="none",
                default_tier=default_tier,
                extra_config=extra_cfg,
            )

        ui_title = (
            (manifest.ui.title or manifest.name)
            if manifest is not None and (manifest.ui is not None or clean_ui)
            else (mod.get("ui_title") if mod is not None else clean_slug)
        )

        async with self.pool.connection() as conn:
            if mod is None:
                assert manifest is not None
                cur = await conn.execute(
                    """
                    INSERT INTO modules (
                        principal_id, space_id, slug, manifest_version, version,
                        image_pin, host, status, mcp_url, ui_url, ui_title,
                        upstream_id, config
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, 'manual', 'installed', %s, %s, %s, %s, %s)
                    RETURNING id, principal_id, space_id, slug, manifest_version, version,
                              image_pin, host, status, mcp_url, ui_url, ui_title,
                              upstream_id, config, installed_at, disabled_at
                    """,
                    (
                        p["id"],
                        sp["id"],
                        clean_slug,
                        manifest.manifest_version,
                        manifest.version,
                        manifest.runtime.image or manifest.runtime.build or "manual",
                        clean_mcp,
                        clean_ui,
                        ui_title,
                        upstream_row["id"],
                        Jsonb(self._module_config_meta(manifest)),
                    ),
                )
            else:
                cur = await conn.execute(
                    """
                    UPDATE modules
                    SET status = 'installed',
                        mcp_url = %s,
                        ui_url = COALESCE(%s, ui_url),
                        ui_title = COALESCE(%s, ui_title),
                        upstream_id = %s,
                        disabled_at = NULL
                    WHERE id = %s
                    RETURNING id, principal_id, space_id, slug, manifest_version, version,
                              image_pin, host, status, mcp_url, ui_url, ui_title,
                              upstream_id, config, installed_at, disabled_at
                    """,
                    (clean_mcp, clean_ui, ui_title, upstream_row["id"], mod["id"]),
                )
            updated = await cur.fetchone()
            await conn.commit()
        assert updated is not None

        if self.upstream_manager is not None:
            await self.upstream_manager.hot_register_from_row(upstream_row, connect=True)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=upstream_row["id"],
            event_type="module.installed" if mod is None else "module.enabled",
            outcome="ok",
            payload={"slug": clean_slug, "host": "manual", "attached": True},
        )
        return dict(updated)

    async def disable(
        self,
        slug: str,
        *,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        """Disable a module: stop container, hide tools, keep data, audit `module.disabled`."""
        clean_slug = slug.strip().lower()
        p, sp = await get_principal_and_space(self.pool, principal_slug)
        mod = await self.get_module(clean_slug, principal_slug=principal_slug)
        if mod is None:
            raise ValueError(f"Module not found: '{clean_slug}'")

        await self.host.stop(clean_slug)

        async with self.pool.connection() as conn:
            if mod["upstream_id"] is not None:
                await conn.execute(
                    "UPDATE providers SET enabled = false WHERE id = %s",
                    (mod["upstream_id"],),
                )
            cur = await conn.execute(
                """
                UPDATE modules
                SET status = 'disabled', disabled_at = now()
                WHERE id = %s
                RETURNING id, principal_id, space_id, slug, manifest_version, version,
                          image_pin, host, status, mcp_url, ui_url, ui_title,
                          upstream_id, config, installed_at, disabled_at
                """,
                (mod["id"],),
            )
            updated = await cur.fetchone()
            await conn.commit()
        assert updated is not None

        if self.upstream_manager is not None:
            await self.upstream_manager.hot_remove(clean_slug)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=mod["upstream_id"],
            event_type="module.disabled",
            outcome="ok",
            payload={"slug": clean_slug},
        )
        return dict(updated)

    async def enable(
        self,
        slug: str,
        *,
        principal_slug: str | None = None,
        wait_ready: bool = True,
    ) -> dict[str, Any]:
        """Re-enable a disabled module: start container, re-register upstream, audit `module.enabled`."""
        clean_slug = slug.strip().lower()
        p, sp = await get_principal_and_space(self.pool, principal_slug)
        mod = await self.get_module(clean_slug, principal_slug=principal_slug)
        if mod is None:
            raise ValueError(f"Module not found: '{clean_slug}'")

        await self.host.start(clean_slug)

        async with self.pool.connection() as conn:
            if mod["upstream_id"] is not None:
                await conn.execute(
                    "UPDATE providers SET enabled = true WHERE id = %s",
                    (mod["upstream_id"],),
                )
            new_status = "installed" if mod["mcp_url"] else "pending_attach"
            cur = await conn.execute(
                """
                UPDATE modules
                SET status = %s, disabled_at = NULL
                WHERE id = %s
                RETURNING id, principal_id, space_id, slug, manifest_version, version,
                          image_pin, host, status, mcp_url, ui_url, ui_title,
                          upstream_id, config, installed_at, disabled_at
                """,
                (new_status, mod["id"]),
            )
            updated = await cur.fetchone()
            await conn.commit()
        assert updated is not None

        if mod["upstream_id"] is not None and self.upstream_manager is not None:
            up_row = await get_remote_upstream(self.pool, slug=clean_slug, principal_id=p["id"])
            if up_row is not None:
                await self.upstream_manager.hot_register_from_row(up_row, connect=False)
                if wait_ready:
                    await self._connect_with_retry(clean_slug)
                    await self.upstream_manager.sessions.broadcast_tools_changed()

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=mod["upstream_id"],
            event_type="module.enabled",
            outcome="ok",
            payload={"slug": clean_slug},
        )
        return dict(updated)

    async def uninstall(
        self,
        slug: str,
        *,
        purge: bool = False,
        principal_slug: str | None = None,
    ) -> dict[str, Any]:
        """Uninstall a module, keeping its Postgres schema unless `purge=True`."""
        clean_slug = slug.strip().lower()
        p, sp = await get_principal_and_space(self.pool, principal_slug)
        mod = await self.get_module(clean_slug, principal_slug=principal_slug)
        if mod is None:
            raise ValueError(f"Module not found: '{clean_slug}'")

        await self.host.uninstall(clean_slug, purge=purge)

        if self.upstream_manager is not None:
            await self.upstream_manager.hot_remove(clean_slug)

        await delete_remote_upstream(self.pool, principal_id=p["id"], slug=clean_slug)

        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM modules WHERE id = %s", (mod["id"],))
            await conn.commit()

        if purge:
            await self.storage.purge_storage(principal_id=p["id"], slug=clean_slug)

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            event_type="module.uninstalled",
            outcome="ok",
            payload={"slug": clean_slug, "purge": purge},
        )
        return {"slug": clean_slug, "purge": purge}

    async def upgrade(
        self,
        slug: str,
        *,
        principal_slug: str | None = None,
        wait_ready: bool = True,
    ) -> dict[str, Any]:
        """Upgrade an installed module to the current catalog manifest."""
        clean_slug = slug.strip().lower()
        p, sp = await get_principal_and_space(self.pool, principal_slug)
        mod = await self.get_module(clean_slug, principal_slug=principal_slug)
        if mod is None:
            raise ValueError(f"Module not found: '{clean_slug}'")
        manifest = get_module_catalog_entry(clean_slug, self.modules_dir)
        if manifest is None:
            raise ValueError(f"Module '{clean_slug}' not found in catalog")

        saved_cfg = await self.storage.load_module_config(principal_id=p["id"], slug=clean_slug)
        module_secret = await self.storage.ensure_module_secret(principal_id=p["id"], slug=clean_slug)
        env: dict[str, str] = {"ZOS_MODULE_SECRET": module_secret}
        if manifest.storage.postgres:
            env["ZOS_MODULE_DATABASE_URL"] = await self.storage.provision_postgres(
                principal_id=p["id"],
                slug=clean_slug,
                base_database_url=self.settings.database_url,
            )

        installed = await self.host.upgrade(clean_slug, manifest, saved_cfg, env=env)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE modules
                SET manifest_version = %s,
                    version = %s,
                    image_pin = %s,
                    mcp_url = COALESCE(NULLIF(%s, ''), mcp_url),
                    ui_url = COALESCE(%s, ui_url),
                    config = %s
                WHERE id = %s
                RETURNING id, principal_id, space_id, slug, manifest_version, version,
                          image_pin, host, status, mcp_url, ui_url, ui_title,
                          upstream_id, config, installed_at, disabled_at
                """,
                (
                    manifest.manifest_version,
                    manifest.version,
                    installed.image_pin,
                    installed.mcp_url,
                    installed.ui_url,
                    Jsonb(self._module_config_meta(manifest)),
                    mod["id"],
                ),
            )
            updated = await cur.fetchone()
            if mod["upstream_id"] is not None:
                up_cur = await conn.execute(
                    "SELECT config FROM providers WHERE id = %s",
                    (mod["upstream_id"],),
                )
                up_existing = await up_cur.fetchone()
                if up_existing is not None:
                    up_cfg = dict(up_existing["config"]) if isinstance(up_existing["config"], dict) else {}
                    existing_managed = list(up_cfg.get("managed_tier_patterns") or [])
                    up_cfg.update(self._provider_extra_config(manifest))
                    up_cfg["managed_tier_patterns"] = existing_managed
                    if installed.mcp_url:
                        up_cfg["url"] = validate_module_url(installed.mcp_url)
                    await conn.execute(
                        "UPDATE providers SET config = %s WHERE id = %s",
                        (Jsonb(up_cfg), mod["upstream_id"]),
                    )
            await conn.commit()
        assert updated is not None

        if mod["upstream_id"] is not None:
            await self._seed_exact_manifest_tiers(
                principal_id=p["id"],
                provider_id=mod["upstream_id"],
                manifest=manifest,
            )

        if mod["upstream_id"] is not None and self.upstream_manager is not None:
            up_row = await get_remote_upstream(self.pool, slug=clean_slug, principal_id=p["id"])
            if up_row is not None:
                await self.upstream_manager.hot_register_from_row(up_row, connect=False)
                if wait_ready:
                    await self._connect_with_retry(clean_slug)
                    await self.upstream_manager.sessions.broadcast_tools_changed()

        await record_event(
            self.pool,
            principal_id=p["id"],
            space_id=sp["id"],
            provider_id=mod["upstream_id"],
            event_type="module.upgraded",
            outcome="ok",
            payload={"slug": clean_slug, "version": manifest.version, "image_pin": installed.image_pin},
        )
        return dict(updated)
