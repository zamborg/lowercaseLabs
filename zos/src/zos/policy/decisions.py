"""Per-client policy decision resolution, pure PolicySnapshot, and DB-backed loader."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import mcp_types as types
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.auth.context import CallContext, TrustCeiling
from zos.policy.annotations import build_tool_annotations
from zos.policy.tiers import Tier, match_best_pattern, resolve_tier

Decision = Literal["allow", "deny", "require_approval"]


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    tier: Tier
    decision: Decision
    annotations: types.ToolAnnotations

TIER_DEFAULT_DECISION: dict[Tier, Decision] = {
    "read": "allow",
    "write": "allow",
    "external": "require_approval",
}


def resolve_decision(
    tool_name: str,
    *,
    tier: Tier,
    trust_ceiling: TrustCeiling,
    client_rules: Mapping[str, Decision] | None = None,
    global_rules: Mapping[str, Decision] | None = None,
) -> Decision:
    """Resolve final policy decision:
    1. Client-specific policy rule (longest literal match)
    2. All-clients policy rule (longest literal match)
    3. Tier default (read/write -> allow, external -> require_approval)
    4. trust_ceiling == 'untrusted' downgrades require_approval -> deny.
    """
    decision: Decision | None = None
    if client_rules:
        decision = match_best_pattern(tool_name, client_rules)
    if decision is None and global_rules:
        decision = match_best_pattern(tool_name, global_rules)
    if decision is None:
        decision = TIER_DEFAULT_DECISION[tier]

    if trust_ceiling == "untrusted" and decision == "require_approval":
        return "deny"
    return decision


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """In-memory snapshot of tiers and client policies loaded once per request."""

    tiers: Mapping[str, Tier]
    client_rules: Mapping[str, Decision]
    global_rules: Mapping[str, Decision]

    def evaluate(
        self,
        exposed_name: str,
        *,
        trust_ceiling: TrustCeiling,
        destructive_hint: bool | None = None,
        idempotent_hint: bool | None = None,
    ) -> PolicyEvaluation:
        tier = resolve_tier(exposed_name, self.tiers)
        decision = resolve_decision(
            exposed_name,
            tier=tier,
            trust_ceiling=trust_ceiling,
            client_rules=self.client_rules,
            global_rules=self.global_rules,
        )
        annotations = build_tool_annotations(
            tier,
            destructive_hint=destructive_hint,
            idempotent_hint=idempotent_hint,
        )
        return PolicyEvaluation(tier=tier, decision=decision, annotations=annotations)


class DatabasePolicyEvaluator:
    """Loads tool_tiers and client_policies from Postgres once per request (avoids N+1 queries)."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> None:
        self.pool = pool

    async def load_snapshot(self, ctx: CallContext) -> PolicySnapshot:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT tool_pattern, tier FROM tool_tiers WHERE principal_id = %s",
                (ctx.principal_id,),
            )
            tiers: dict[str, Tier] = {row["tool_pattern"]: row["tier"] for row in await cur.fetchall()}

            cur = await conn.execute(
                """
                SELECT client_id, tool_pattern, decision
                FROM client_policies
                WHERE principal_id = %s AND (client_id = %s OR client_id IS NULL)
                """,
                (ctx.principal_id, ctx.client_id),
            )
            client_rules: dict[str, Decision] = {}
            global_rules: dict[str, Decision] = {}
            for row in await cur.fetchall():
                if row["client_id"] is None:
                    global_rules[row["tool_pattern"]] = row["decision"]
                else:
                    client_rules[row["tool_pattern"]] = row["decision"]

        return PolicySnapshot(tiers=tiers, client_rules=client_rules, global_rules=global_rules)

    async def evaluate(
        self,
        ctx: CallContext,
        exposed_name: str,
        destructive_hint: bool | None = None,
        idempotent_hint: bool | None = None,
    ) -> PolicyEvaluation:
        snapshot = await self.load_snapshot(ctx)
        return snapshot.evaluate(
            exposed_name,
            trust_ceiling=ctx.trust_ceiling,
            destructive_hint=destructive_hint,
            idempotent_hint=idempotent_hint,
        )
