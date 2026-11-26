"""Database query helpers for principals, spaces, clients, tokens, policies, and providers."""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

V0_TIER_SEEDS: dict[str, str] = {
    "zos_status": "read",
}


async def _ensure_slug_available(
    conn: AsyncConnection[DictRow],
    *,
    principal_id: UUID,
    slug: str,
    skip_modules: bool = False,
) -> None:
    """Enforce strict slug uniqueness across all provider kinds (connectors, modules, devices)."""
    cur = await conn.execute(
        "SELECT id, kind, config FROM providers WHERE principal_id = %s AND slug = %s",
        (principal_id, slug),
    )
    prow = await cur.fetchone()
    if prow is not None:
        kind = str(prow["kind"])
        cfg = prow["config"] if isinstance(prow["config"], dict) else {}
        if kind == "actuator":
            label = "device"
        elif cfg.get("module_slug"):
            label = "module"
        else:
            label = "connector"
        raise ValueError(f"Slug '{slug}' is already in use by an existing {label}")

    if not skip_modules:
        cur = await conn.execute(
            "SELECT id FROM modules WHERE principal_id = %s AND slug = %s",
            (principal_id, slug),
        )
        if await cur.fetchone() is not None:
            raise ValueError(f"Slug '{slug}' is already in use by an installed module")


def hash_secret(raw: str) -> bytes:
    """Compute SHA-256 digest for opaque tokens and authorization codes."""
    return hashlib.sha256(raw.encode("utf-8")).digest()


def generate_token(prefix: str = "zos") -> tuple[str, bytes]:
    """Generate a high-entropy opaque token and its SHA-256 hash."""
    raw = f"{prefix}_{secrets.token_urlsafe(32)}"
    return raw, hash_secret(raw)


async def init_principal_and_space(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_slug: str = "owner",
    display_name: str = "Owner",
    space_slug: str = "main",
    space_name: str = "Main",
) -> dict[str, Any]:
    """Idempotently initialize principal, single space, and built-in tool tier seeds."""
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            """
                INSERT INTO principals (slug, display_name)
                VALUES (%s, %s)
                ON CONFLICT (slug) DO UPDATE SET display_name = EXCLUDED.display_name
                RETURNING id, slug, display_name
                """,
            (principal_slug, display_name),
        )
        principal = await cur.fetchone()
        assert principal is not None
        principal_id: UUID = principal["id"]

        cur = await conn.execute(
            """
                INSERT INTO spaces (principal_id, slug, name)
                VALUES (%s, %s, %s)
                ON CONFLICT (principal_id, slug) DO UPDATE SET name = EXCLUDED.name
                RETURNING id, slug, name
                """,
            (principal_id, space_slug, space_name),
        )
        space = await cur.fetchone()
        assert space is not None

        for pattern, tier in V0_TIER_SEEDS.items():
            await conn.execute(
                """
                    INSERT INTO tool_tiers (principal_id, tool_pattern, tier)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (principal_id, tool_pattern) DO NOTHING
                    """,
                (principal_id, pattern, tier),
            )

    return {
        "principal": dict(principal),
        "space": dict(space),
        "tier_seed_count": len(V0_TIER_SEEDS),
    }


async def get_principal_and_space(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    principal_slug: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch the principal (by slug, or the sole principal in V0) and its single space."""
    async with pool.connection() as conn:
        if principal_slug:
            cur = await conn.execute(
                "SELECT id, slug, display_name FROM principals WHERE slug = %s",
                (principal_slug,),
            )
            principal = await cur.fetchone()
        else:
            cur = await conn.execute(
                "SELECT id, slug, display_name FROM principals ORDER BY created_at ASC LIMIT 1"
            )
            principal = await cur.fetchone()
        if principal is None:
            raise ValueError("No principal found. Run `zos init` first.")

        cur = await conn.execute(
            """
            SELECT id, principal_id, slug, name
            FROM spaces
            WHERE principal_id = %s
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (principal["id"],),
        )
        space = await cur.fetchone()
        if space is None:
            raise ValueError("No space found for principal. Run `zos init` first.")

    return dict(principal), dict(space)


async def create_bearer_client(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    space_id: UUID,
    name: str,
    trust_ceiling: str = "owner",
) -> dict[str, Any]:
    """Create a bearer client and mint its initial opaque bearer token."""
    raw_token, token_hash = generate_token("zos_b")
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            """
                INSERT INTO clients (principal_id, default_space_id, name, kind, trust_ceiling)
                VALUES (%s, %s, %s, 'bearer', %s)
                RETURNING id, name, kind, trust_ceiling
                """,
            (principal_id, space_id, name, trust_ceiling),
        )
        client = await cur.fetchone()
        assert client is not None

        cur = await conn.execute(
            """
                INSERT INTO tokens (principal_id, client_id, kind, token_hash, scopes)
                VALUES (%s, %s, 'bearer', %s, %s)
                RETURNING id
                """,
            (principal_id, client["id"], token_hash, ["mcp"]),
        )
        token_row = await cur.fetchone()
        assert token_row is not None

    return {
        "client": dict(client),
        "token_id": token_row["id"],
        "token": raw_token,
    }


async def mint_bearer_token_for_client(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    client_name: str,
) -> dict[str, Any]:
    """Mint a new bearer token for an existing bearer client."""
    raw_token, token_hash = generate_token("zos_b")
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT id, name, default_space_id FROM clients
            WHERE principal_id = %s AND name = %s AND kind = 'bearer' AND disabled_at IS NULL
            """,
            (principal_id, client_name),
        )
        client = await cur.fetchone()
        if client is None:
            raise ValueError(f"Bearer client not found: {client_name}")

        cur = await conn.execute(
            """
            INSERT INTO tokens (principal_id, client_id, kind, token_hash, scopes)
            VALUES (%s, %s, 'bearer', %s, %s)
            RETURNING id
            """,
            (principal_id, client["id"], token_hash, ["mcp"]),
        )
        token_row = await cur.fetchone()
        await conn.commit()
        assert token_row is not None

    return {
        "client": dict(client),
        "token_id": token_row["id"],
        "token": raw_token,
    }


async def revoke_token(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    token_id: UUID | None = None,
    raw_token: str | None = None,
    client_id: UUID | None = None,
) -> int:
    """Revoke matching active token(s). Returns count revoked."""
    async with pool.connection() as conn:
        if token_id is not None:
            cur = await conn.execute(
                """
                UPDATE tokens SET revoked_at = now()
                WHERE id = %s AND revoked_at IS NULL
                """,
                (token_id,),
            )
        elif raw_token is not None:
            cur = await conn.execute(
                """
                UPDATE tokens SET revoked_at = now()
                WHERE token_hash = %s AND revoked_at IS NULL
                """,
                (hash_secret(raw_token),),
            )
        elif client_id is not None:
            cur = await conn.execute(
                """
                UPDATE tokens SET revoked_at = now()
                WHERE client_id = %s AND revoked_at IS NULL
                """,
                (client_id,),
            )
        else:
            raise ValueError("Must specify token_id, raw_token, or client_id")
        count = cur.rowcount or 0
        await conn.commit()
    return count


async def find_client_by_name_or_id(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    identifier: str,
    include_disabled: bool = False,
) -> dict[str, Any] | None:
    """Resolve a client by UUID, oauth_client_id, or bearer client name (active clients by default)."""
    disabled_clause = "" if include_disabled else " AND disabled_at IS NULL"
    async with pool.connection() as conn:
        try:
            parsed_uuid = UUID(identifier)
        except ValueError:
            parsed_uuid = None

        if parsed_uuid is not None:
            cur = await conn.execute(
                f"SELECT id, name, kind, default_space_id, trust_ceiling FROM clients "
                f"WHERE principal_id = %s AND id = %s{disabled_clause}",
                (principal_id, parsed_uuid),
            )
            row = await cur.fetchone()
            if row:
                return dict(row)

        cur = await conn.execute(
            f"""
            SELECT id, name, kind, default_space_id, trust_ceiling FROM clients
            WHERE principal_id = %s AND (oauth_client_id = %s OR name = %s){disabled_clause}
            ORDER BY CASE WHEN kind = 'bearer' THEN 0 ELSE 1 END, created_at ASC
            LIMIT 1
            """,
            (principal_id, identifier, identifier),
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def disable_client(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    client_id: UUID,
) -> dict[str, Any] | None:
    """Soft-disable any client and revoke its active tokens while preserving events and memories."""
    async with pool.connection() as conn, conn.transaction():
        tok_cur = await conn.execute(
            """
            UPDATE tokens
            SET revoked_at = now()
            WHERE client_id = %s AND revoked_at IS NULL
            """,
            (client_id,),
        )
        revoked_count = int(tok_cur.rowcount or 0)

        cur = await conn.execute(
            """
            UPDATE clients
            SET disabled_at = now()
            WHERE principal_id = %s AND id = %s AND disabled_at IS NULL
            RETURNING id, name, kind, oauth_client_id, disabled_at
            """,
            (principal_id, client_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        res = dict(row)
        res["revoked_tokens"] = revoked_count
        return res


async def delete_oauth_client(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    client_id: UUID,
) -> dict[str, Any] | None:
    """Delete an OAuth client (cascading codes/tokens/policies, preserving events). Returns deleted row."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            DELETE FROM clients
            WHERE principal_id = %s AND id = %s AND kind = 'oauth'
            RETURNING id, name, kind, oauth_client_id
            """,
            (principal_id, client_id),
        )
        row = await cur.fetchone()
        await conn.commit()
        return dict(row) if row else None


async def set_provider_backend(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    space_id: UUID,
    slug: str,
    backend: str,
    default_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist an upstream provider's backend in `providers.config` (preserving other keys)."""
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            """
            SELECT id, slug, kind, config, enabled, last_seen_at
            FROM providers
            WHERE principal_id = %s AND slug = %s
            FOR UPDATE
            """,
            (principal_id, slug),
        )
        existing = await cur.fetchone()
        if existing is not None:
            if existing["kind"] != "upstream":
                raise ValueError(f"Provider '{slug}' is not an upstream provider (kind={existing['kind']})")
            old_cfg = dict(existing["config"]) if isinstance(existing["config"], dict) else {}
            prev_backend = str(old_cfg.get("backend")) if old_cfg.get("backend") else None
            new_cfg = {**old_cfg, "backend": backend}
            cur = await conn.execute(
                """
                UPDATE providers
                SET config = %s
                WHERE id = %s
                RETURNING id, slug, kind, config, enabled, last_seen_at
                """,
                (Jsonb(new_cfg), existing["id"]),
            )
            row = await cur.fetchone()
            assert row is not None
        else:
            prev_backend = None
            new_cfg = {**(default_config or {}), "backend": backend}
            cur = await conn.execute(
                """
                INSERT INTO providers (principal_id, space_id, slug, kind, config, enabled)
                VALUES (%s, %s, %s, 'upstream', %s, true)
                RETURNING id, slug, kind, config, enabled, last_seen_at
                """,
                (principal_id, space_id, slug, Jsonb(new_cfg)),
            )
            row = await cur.fetchone()
            assert row is not None

    out = dict(row)
    out["previous_backend"] = prev_backend
    out["backend"] = backend
    return out


async def set_tool_tier(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    tool_pattern: str,
    tier: str,
) -> dict[str, Any]:
    """Upsert a tool tier pattern (`read` | `write` | `external`) in `tool_tiers`."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO tool_tiers (principal_id, tool_pattern, tier)
            VALUES (%s, %s, %s)
            ON CONFLICT (principal_id, tool_pattern) DO UPDATE
            SET tier = EXCLUDED.tier
            RETURNING principal_id, tool_pattern, tier
            """,
            (principal_id, tool_pattern, tier),
        )
        row = await cur.fetchone()
        await conn.commit()
    assert row is not None
    return dict(row)


async def delete_tool_tier(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    tool_pattern: str,
) -> bool:
    """Delete a tool tier pattern if present."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM tool_tiers WHERE principal_id = %s AND tool_pattern = %s",
            (principal_id, tool_pattern),
        )
        deleted = (cur.rowcount or 0) > 0
        await conn.commit()
    return deleted


async def create_remote_upstream(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    space_id: UUID,
    slug: str,
    url: str,
    auth_mode: str,
    default_tier: str = "external",
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a generic remote MCP upstream (`runtime: remote`) in `providers`."""
    config: dict[str, Any] = {
        "runtime": "remote",
        "transport": "mcp",
        "url": url,
        "auth_mode": auth_mode,
        "default_tier": default_tier,
    }
    if extra_config:
        config.update(extra_config)
    async with pool.connection() as conn, conn.transaction():
        await _ensure_slug_available(
            conn,
            principal_id=principal_id,
            slug=slug,
            skip_modules=bool(extra_config and extra_config.get("module_slug")),
        )

        cur = await conn.execute(
            """
            INSERT INTO providers (
                principal_id, space_id, slug, kind, credential_id, config, enabled
            )
            VALUES (%s, %s, %s, 'upstream', NULL, %s, true)
            RETURNING id, principal_id, space_id, slug, kind, credential_id, config, enabled, last_seen_at, created_at
            """,
            (principal_id, space_id, slug, Jsonb(config)),
        )
        row = await cur.fetchone()
        assert row is not None

        if default_tier != "external":
            await conn.execute(
                """
                INSERT INTO tool_tiers (principal_id, tool_pattern, tier)
                VALUES (%s, %s, %s)
                ON CONFLICT (principal_id, tool_pattern) DO UPDATE
                SET tier = EXCLUDED.tier
                """,
                (principal_id, f"{slug}_*", default_tier),
            )
    return dict(row)


async def apply_manifest_tiers_for_tools(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    provider_id: UUID,
    slug: str,
    catalog_slug: str,
    patterns: dict[str, str],
    exposed_tools: list[tuple[str, str]],
) -> list[str]:
    """Resolve manifest glob/exact tier patterns onto discovered tools without overriding owner tiers.

    `exposed_tools` is a list of `(exposed_name, local_sanitized_name)`.
    Returns list of newly written `tool_tiers` patterns.
    """
    from zos.policy.tiers import match_best_pattern
    from zos.router.naming import format_tool_name

    if not patterns or not exposed_tools:
        return []

    written: list[str] = []
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            "SELECT tool_pattern FROM tool_tiers WHERE principal_id = %s",
            (principal_id,),
        )
        existing_patterns = {str(r["tool_pattern"]) for r in await cur.fetchall()}

        cur = await conn.execute(
            "SELECT config FROM providers WHERE id = %s FOR UPDATE",
            (provider_id,),
        )
        prow = await cur.fetchone()
        if prow is None:
            return []
        cfg = dict(prow["config"]) if isinstance(prow["config"], dict) else {}
        managed = set(cfg.get("managed_tier_patterns") or [])

        for exposed_name, local_name in exposed_tools:
            if exposed_name in existing_patterns:
                continue
            matched = match_best_pattern(exposed_name, patterns)
            if matched is None and catalog_slug and catalog_slug != slug:
                canonical_name = format_tool_name(catalog_slug, local_name)
                matched = match_best_pattern(canonical_name, patterns)
            if matched is None:
                continue
            ins = await conn.execute(
                """
                INSERT INTO tool_tiers (principal_id, tool_pattern, tier)
                VALUES (%s, %s, %s)
                ON CONFLICT (principal_id, tool_pattern) DO NOTHING
                """,
                (principal_id, exposed_name, matched),
            )
            if (ins.rowcount or 0) > 0:
                existing_patterns.add(exposed_name)
                managed.add(exposed_name)
                written.append(exposed_name)

        if written:
            cfg["managed_tier_patterns"] = sorted(managed)
            await conn.execute(
                "UPDATE providers SET config = %s WHERE id = %s",
                (Jsonb(cfg), provider_id),
            )
    return written


async def unmark_managed_tier_pattern(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    tool_pattern: str,
) -> None:
    """Remove `tool_pattern` from any remote upstream's `managed_tier_patterns` after owner edit."""
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            """
            SELECT id, config FROM providers
            WHERE principal_id = %s AND kind = 'upstream' AND config->>'runtime' = 'remote'
            """,
            (principal_id,),
        )
        for row in await cur.fetchall():
            cfg = dict(row["config"]) if isinstance(row["config"], dict) else {}
            managed = list(cfg.get("managed_tier_patterns") or [])
            if tool_pattern in managed:
                cfg["managed_tier_patterns"] = [p for p in managed if p != tool_pattern]
                await conn.execute(
                    "UPDATE providers SET config = %s WHERE id = %s",
                    (Jsonb(cfg), row["id"]),
                )


async def list_remote_upstreams(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID | None = None,
) -> list[dict[str, Any]]:
    """List registered generic remote MCP upstream providers (`config->>'runtime' = 'remote'`)."""
    async with pool.connection() as conn:
        if principal_id is not None:
            cur = await conn.execute(
                """
                SELECT id, principal_id, space_id, slug, kind, credential_id, config, enabled, last_seen_at, created_at
                FROM providers
                WHERE principal_id = %s AND kind = 'upstream' AND config->>'runtime' = 'remote'
                ORDER BY slug ASC
                """,
                (principal_id,),
            )
        else:
            cur = await conn.execute(
                """
                SELECT id, principal_id, space_id, slug, kind, credential_id, config, enabled, last_seen_at, created_at
                FROM providers
                WHERE kind = 'upstream' AND config->>'runtime' = 'remote'
                ORDER BY slug ASC
                """
            )
        return [dict(r) for r in await cur.fetchall()]


async def get_remote_upstream(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    slug: str,
    principal_id: UUID | None = None,
) -> dict[str, Any] | None:
    """Fetch a single generic remote MCP upstream provider by slug."""
    async with pool.connection() as conn:
        if principal_id is not None:
            cur = await conn.execute(
                """
                SELECT id, principal_id, space_id, slug, kind, credential_id, config, enabled, last_seen_at, created_at
                FROM providers
                WHERE principal_id = %s AND slug = %s AND kind = 'upstream' AND config->>'runtime' = 'remote'
                """,
                (principal_id, slug),
            )
        else:
            cur = await conn.execute(
                """
                SELECT id, principal_id, space_id, slug, kind, credential_id, config, enabled, last_seen_at, created_at
                FROM providers
                WHERE slug = %s AND kind = 'upstream' AND config->>'runtime' = 'remote'
                LIMIT 1
                """,
                (slug,),
            )
        row = await cur.fetchone()
    return dict(row) if row else None


async def attach_upstream_credential(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    provider_id: UUID,
    credential_id: UUID,
) -> None:
    """Bind a vault credential ID to a remote upstream provider and mark last_seen_at."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE providers
            SET credential_id = %s, last_seen_at = now()
            WHERE id = %s
            """,
            (credential_id, provider_id),
        )
        await conn.commit()


async def clear_upstream_credential(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    provider_id: UUID,
) -> UUID | None:
    """Detach and return the previous credential_id on a remote upstream provider."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT credential_id FROM providers WHERE id = %s",
            (provider_id,),
        )
        row = await cur.fetchone()
        old_cred_id: UUID | None = row["credential_id"] if row else None
        await conn.execute(
            "UPDATE providers SET credential_id = NULL WHERE id = %s",
            (provider_id,),
        )
        await conn.commit()
    return old_cred_id


async def delete_remote_upstream(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    slug: str,
) -> dict[str, Any] | None:
    """Delete a remote upstream provider, its vault credential, and its `<slug>_*` default tier."""
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            """
            DELETE FROM providers
            WHERE principal_id = %s AND slug = %s AND kind = 'upstream' AND config->>'runtime' = 'remote'
            RETURNING id, slug, credential_id, config
            """,
            (principal_id, slug),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        cred_id = row["credential_id"]
        if cred_id is not None:
            await conn.execute("DELETE FROM credentials WHERE id = %s", (cred_id,))
        cfg = row["config"] if isinstance(row["config"], dict) else {}
        managed_patterns = [str(p) for p in (cfg.get("managed_tier_patterns") or []) if str(p)]
        patterns_to_drop = sorted({f"{slug}_*", *managed_patterns})
        await conn.execute(
            "DELETE FROM tool_tiers WHERE principal_id = %s AND tool_pattern = ANY(%s)",
            (principal_id, patterns_to_drop),
        )
    return dict(row)


async def set_client_policy(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    tool_pattern: str,
    decision: str,
    client_id: UUID | None = None,
) -> dict[str, Any]:
    """Upsert a client policy rule (client_id=None applies to all clients)."""
    if decision not in ("allow", "deny", "require_approval"):
        raise ValueError(f"Invalid decision {decision!r}; must be allow, deny, or require_approval")

    async with pool.connection() as conn:
        if client_id is None:
            cur = await conn.execute(
                """
                UPDATE client_policies
                SET decision = %s
                WHERE principal_id = %s AND client_id IS NULL AND tool_pattern = %s
                RETURNING id, principal_id, client_id, tool_pattern, decision
                """,
                (decision, principal_id, tool_pattern),
            )
            row = await cur.fetchone()
            if row is None:
                cur = await conn.execute(
                    """
                    INSERT INTO client_policies (principal_id, client_id, tool_pattern, decision)
                    VALUES (%s, NULL, %s, %s)
                    RETURNING id, principal_id, client_id, tool_pattern, decision
                    """,
                    (principal_id, tool_pattern, decision),
                )
                row = await cur.fetchone()
        else:
            cur = await conn.execute(
                """
                INSERT INTO client_policies (principal_id, client_id, tool_pattern, decision)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (principal_id, client_id, tool_pattern)
                DO UPDATE SET decision = EXCLUDED.decision
                RETURNING id, principal_id, client_id, tool_pattern, decision
                """,
                (principal_id, client_id, tool_pattern, decision),
            )
            row = await cur.fetchone()
        await conn.commit()
    assert row is not None
    return dict(row)


async def clear_client_policy(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    tool_pattern: str,
    client_id: UUID | None = None,
) -> bool:
    """Delete a client policy override row (client_id=None clears an all-clients rule)."""
    async with pool.connection() as conn:
        if client_id is None:
            cur = await conn.execute(
                """
                DELETE FROM client_policies
                WHERE principal_id = %s AND client_id IS NULL AND tool_pattern = %s
                """,
                (principal_id, tool_pattern),
            )
        else:
            cur = await conn.execute(
                """
                DELETE FROM client_policies
                WHERE principal_id = %s AND client_id = %s AND tool_pattern = %s
                """,
                (principal_id, client_id, tool_pattern),
            )
        deleted = (cur.rowcount or 0) > 0
        await conn.commit()
    return deleted


async def create_actuator_provider_and_token(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    space_id: UUID,
    slug: str,
) -> dict[str, Any]:
    """Create a new device (actuator) provider row and mint its initial token.

    Enforces strict slug uniqueness across devices, modules, and connectors.
    """
    raw_token, token_hash = generate_token("zos_a")
    async with pool.connection() as conn, conn.transaction():
        await _ensure_slug_available(conn, principal_id=principal_id, slug=slug)

        cur = await conn.execute(
            """
                INSERT INTO providers (principal_id, space_id, slug, kind, config, enabled)
                VALUES (%s, %s, %s, 'actuator', %s, true)
                RETURNING id, slug, kind, enabled, last_seen_at, created_at
                """,
            (principal_id, space_id, slug, Jsonb({})),
        )
        provider = await cur.fetchone()
        assert provider is not None

        cur = await conn.execute(
            """
                INSERT INTO tokens (principal_id, provider_id, kind, token_hash, scopes)
                VALUES (%s, %s, 'actuator', %s, %s)
                RETURNING id
                """,
            (principal_id, provider["id"], token_hash, ["actuator"]),
        )
        token_row = await cur.fetchone()
        assert token_row is not None

    return {
        "provider": dict(provider),
        "token_id": token_row["id"],
        "token": raw_token,
    }


async def list_device_providers(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
) -> list[dict[str, Any]]:
    """List all persisted device (`kind = 'actuator'`) providers for a principal."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT id, principal_id, space_id, slug, kind, enabled, last_seen_at, created_at
            FROM providers
            WHERE principal_id = %s AND kind = 'actuator'
            ORDER BY slug ASC
            """,
            (principal_id,),
        )
        return [dict(r) for r in await cur.fetchall()]


async def rotate_device_token(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    slug: str,
) -> dict[str, Any]:
    """Revoke all existing tokens for a device and mint a fresh actuator token."""
    raw_token, token_hash = generate_token("zos_a")
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            """
            SELECT id, slug, kind, enabled, last_seen_at, created_at
            FROM providers
            WHERE principal_id = %s AND slug = %s AND kind = 'actuator'
            FOR UPDATE
            """,
            (principal_id, slug),
        )
        provider = await cur.fetchone()
        if provider is None:
            raise ValueError(f"Device '{slug}' not found")

        rev = await conn.execute(
            """
            UPDATE tokens
            SET revoked_at = now()
            WHERE provider_id = %s AND kind = 'actuator' AND revoked_at IS NULL
            """,
            (provider["id"],),
        )
        revoked_count = rev.rowcount or 0

        cur = await conn.execute(
            """
            INSERT INTO tokens (principal_id, provider_id, kind, token_hash, scopes)
            VALUES (%s, %s, 'actuator', %s, %s)
            RETURNING id
            """,
            (principal_id, provider["id"], token_hash, ["actuator"]),
        )
        token_row = await cur.fetchone()
        assert token_row is not None

    return {
        "provider": dict(provider),
        "token_id": token_row["id"],
        "token": raw_token,
        "revoked_tokens": revoked_count,
    }


async def set_device_enabled(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    slug: str,
    enabled: bool,
) -> dict[str, Any]:
    """Enable or disable a device provider while keeping its row and tokens intact."""
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            """
            UPDATE providers
            SET enabled = %s
            WHERE principal_id = %s AND slug = %s AND kind = 'actuator'
            RETURNING id, slug, kind, enabled, last_seen_at, created_at
            """,
            (enabled, principal_id, slug),
        )
        row = await cur.fetchone()
        if row is None:
            raise ValueError(f"Device '{slug}' not found")
    return dict(row)


async def delete_device_provider(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    slug: str,
) -> dict[str, Any]:
    """Revoke all tokens and delete a device provider row (preserving audit events)."""
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            """
            SELECT id, slug, kind
            FROM providers
            WHERE principal_id = %s AND slug = %s AND kind = 'actuator'
            FOR UPDATE
            """,
            (principal_id, slug),
        )
        provider = await cur.fetchone()
        if provider is None:
            raise ValueError(f"Device '{slug}' not found")

        rev = await conn.execute(
            """
            UPDATE tokens
            SET revoked_at = now()
            WHERE provider_id = %s AND kind = 'actuator' AND revoked_at IS NULL
            """,
            (provider["id"],),
        )
        revoked_count = rev.rowcount or 0

        await conn.execute("DELETE FROM providers WHERE id = %s", (provider["id"],))
    return {
        "id": provider["id"],
        "slug": provider["slug"],
        "revoked_tokens": revoked_count,
    }


DEFAULT_ALLOWED_REDIRECT_HOSTS: tuple[str, ...] = ("chatgpt.com", "claude.ai", "agent.meta.ai")
_REDIRECT_HOST_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$"
)


def normalize_redirect_host(raw: str) -> str:
    """Validate and normalize an external OAuth redirect hostname (lowercase, no scheme/path/port)."""
    cleaned = raw.strip().lower().rstrip(".")
    if not cleaned or len(cleaned) > 253 or not _REDIRECT_HOST_RE.fullmatch(cleaned):
        raise ValueError(
            f"Invalid redirect host {raw!r}: expected a domain name such as 'chatgpt.com'"
        )
    return cleaned


async def seed_allowed_redirect_hosts(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    env_hosts: Sequence[str] = (),
) -> None:
    """Idempotently seed default + `ZOS_ALLOWED_REDIRECT_HOSTS` into Postgres."""
    async with pool.connection() as conn, conn.transaction():
        for host in DEFAULT_ALLOWED_REDIRECT_HOSTS:
            await conn.execute(
                """
                INSERT INTO allowed_redirect_hosts (host, source)
                VALUES (%s, 'default')
                ON CONFLICT (host) DO NOTHING
                """,
                (host,),
            )
        for raw in env_hosts:
            cleaned = raw.strip().lower().rstrip(".")
            if not cleaned or not _REDIRECT_HOST_RE.fullmatch(cleaned):
                continue
            await conn.execute(
                """
                INSERT INTO allowed_redirect_hosts (host, source)
                VALUES (%s, 'env')
                ON CONFLICT (host) DO NOTHING
                """,
                (cleaned,),
            )


async def list_allowed_redirect_hosts(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT host, source, created_at FROM allowed_redirect_hosts ORDER BY host ASC"
        )
        return [dict(r) for r in await cur.fetchall()]


async def add_allowed_redirect_host(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    host: str,
    source: str = "manual",
) -> str:
    clean = normalize_redirect_host(host)
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            """
            INSERT INTO allowed_redirect_hosts (host, source)
            VALUES (%s, %s)
            ON CONFLICT (host) DO NOTHING
            """,
            (clean, source),
        )
        await conn.execute("DELETE FROM dcr_rejected_attempts WHERE host = %s", (clean,))
    return clean


async def remove_allowed_redirect_host(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    host: str,
) -> bool:
    clean = host.strip().lower().rstrip(".")
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute("DELETE FROM allowed_redirect_hosts WHERE host = %s", (clean,))
        return (cur.rowcount or 0) > 0


async def record_dcr_rejected_attempt(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    host: str,
    client_name: str,
) -> None:
    clean_host = host.strip().lower().rstrip(".")
    clean_name = (client_name.strip() or "Unknown client")[:120]
    if not clean_host:
        return
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            """
            INSERT INTO dcr_rejected_attempts (host, client_name, created_at)
            VALUES (%s, %s, now())
            ON CONFLICT (host, client_name)
            DO UPDATE SET created_at = now()
            """,
            (clean_host, clean_name),
        )


async def list_dcr_rejected_attempts(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT id, host, client_name, created_at
            FROM dcr_rejected_attempts
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(r) for r in await cur.fetchall()]


async def delete_dcr_rejected_attempt(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    attempt_id: int,
) -> dict[str, Any] | None:
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            """
            DELETE FROM dcr_rejected_attempts
            WHERE id = %s
            RETURNING id, host, client_name, created_at
            """,
            (attempt_id,),
        )
        row = await cur.fetchone()
    return dict(row) if row is not None else None
