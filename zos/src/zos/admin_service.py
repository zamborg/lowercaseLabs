"""Shared admin service layer backing both the `zos` CLI and the `/admin` dashboard.

Every mutation goes through this service and writes the exact audit events to `events`.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.audit import record_event
from zos.db.queries import (
    clear_client_policy,
    create_actuator_provider_and_token,
    create_bearer_client,
    find_client_by_name_or_id,
    get_principal_and_space,
    mint_bearer_token_for_client,
    revoke_token,
    set_client_policy,
)
from zos.policy import Decision, Tier, build_tool_annotations, resolve_decision, resolve_tier
from zos.router import ZOS_STATUS_TOOL_NAME, ProviderRegistry, SessionTracker, format_tool_name


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

        return {
            "active_sessions": sessions.count,
            "providers": providers,
            "google_credential": google_credential,
        }

    async def list_clients(self) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT
                    c.id,
                    c.name,
                    c.kind,
                    c.oauth_client_id,
                    c.trust_ceiling,
                    c.created_at,
                    c.last_seen_at,
                    COUNT(t.id) FILTER (
                        WHERE t.revoked_at IS NULL
                          AND (t.expires_at IS NULL OR t.expires_at > now())
                          AND t.kind IN ('bearer', 'access', 'refresh')
                    ) AS active_token_count
                FROM clients c
                LEFT JOIN tokens t ON t.client_id = c.id
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
            destructive_hint: bool = False,
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
