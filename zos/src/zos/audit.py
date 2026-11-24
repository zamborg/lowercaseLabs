"""Append-only audit event logging (events table).

Raw tool arguments are never stored by default; only SHA-256 of canonical JSON.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

Outcome = Literal["ok", "error", "denied"]


def canonical_json(data: Any) -> str:
    """Serialize any JSON-compatible value into canonical deterministic JSON."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def args_digest(args: MappingLike | None) -> bytes:
    """Return SHA-256 digest bytes of canonical JSON arguments."""
    payload = {} if args is None else args
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()


MappingLike = dict[str, Any]


async def record_event(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    principal_id: UUID,
    space_id: UUID,
    event_type: str,
    outcome: Outcome = "ok",
    client_id: UUID | None = None,
    provider_id: UUID | None = None,
    tool: str | None = None,
    decision: str | None = None,
    args: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> int:
    """Insert one immutable row into `events` and return its monotonic id."""
    digest = args_digest(args) if args is not None else None
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO events (
                principal_id, space_id, client_id, provider_id,
                type, tool, decision, outcome, args_digest, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                principal_id,
                space_id,
                client_id,
                provider_id,
                event_type,
                tool,
                decision,
                outcome,
                digest,
                Jsonb(payload or {}),
            ),
        )
        row = await cur.fetchone()
        await conn.commit()
    assert row is not None
    return int(row["id"])


class DatabaseAuditRecorder:
    """Router audit callback recording one `tool.called` event per tool invocation."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> None:
        self.pool = pool

    async def record_tool_call(
        self,
        ctx: Any,
        provider_id: UUID | None,
        tool: str,
        decision: str,
        outcome: str,
        args: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> None:
        await record_event(
            self.pool,
            principal_id=ctx.principal_id,
            space_id=ctx.space_id,
            client_id=ctx.client_id,
            provider_id=provider_id,
            event_type="tool.called",
            tool=tool,
            decision=decision,
            outcome=outcome,  # type: ignore[arg-type]
            args=args,
            payload=payload,
        )
