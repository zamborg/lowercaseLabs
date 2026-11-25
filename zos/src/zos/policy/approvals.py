"""Async tool-call approvals for MCP clients without elicitation support (M9c).

Arguments are encrypted at rest with the CredentialVault AEAD (AES-256-GCM) and
purged immediately once an approval is decided, consumed, or expired.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.audit import args_digest, record_event
from zos.auth.context import CallContext
from zos.vault import CredentialVault

ISSUER_PENDING_APPROVAL = "pending_approval"
DEFAULT_WAIT_SECONDS = 45.0
DEFAULT_TTL_SECONDS = 600


class ApprovalError(Exception):
    """Raised when an approval cannot be found or transitioned."""


class ApprovalManager:
    """Manages pending async tool approvals, in-process wakeups, and AEAD arg storage."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        *,
        secret_key: str,
        base_url: str,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.pool = pool
        self.base_url = base_url.rstrip("/")
        self.wait_seconds = max(0.0, float(wait_seconds))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._vault: CredentialVault | None = CredentialVault(secret_key) if secret_key.strip() else None
        self._waiters: dict[str, asyncio.Event] = {}

    @property
    def is_enabled(self) -> bool:
        return self._vault is not None

    def approval_url(self, approval_id: str) -> str:
        return f"{self.base_url}/admin/approvals/{approval_id}"

    async def sweep_expired(self) -> int:
        """Expire overdue approvals and ensure ciphertext is purged on all non-pending rows."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_approvals
                SET status = CASE
                        WHEN expires_at <= now() AND status IN ('pending', 'approved') THEN 'expired'
                        ELSE status
                    END,
                    args_ciphertext = NULL,
                    args_nonce = NULL
                WHERE (expires_at <= now() AND status IN ('pending', 'approved'))
                   OR (status <> 'pending' AND (args_ciphertext IS NOT NULL OR args_nonce IS NOT NULL))
                """
            )
            count = int(cur.rowcount or 0)
            await conn.commit()
        return count

    async def try_consume_approved(
        self,
        ctx: CallContext,
        tool: str,
        args: dict[str, Any],
    ) -> str | None:
        """Atomically consume one matching approved (unexpired) approval, if present."""
        digest = args_digest(args)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_approvals
                SET status = 'consumed',
                    consumed_at = now(),
                    args_ciphertext = NULL,
                    args_nonce = NULL
                WHERE id = (
                    SELECT id
                    FROM pending_approvals
                    WHERE principal_id = %s
                      AND client_id = %s
                      AND tool = %s
                      AND args_digest = %s
                      AND status = 'approved'
                      AND expires_at > now()
                    ORDER BY decided_at ASC NULLS LAST, created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id
                """,
                (ctx.principal_id, ctx.client_id, tool, digest),
            )
            row = await cur.fetchone()
            await conn.commit()
        return str(row["id"]) if row is not None else None

    async def _try_consume_by_id(self, approval_id: str) -> bool:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_approvals
                SET status = 'consumed',
                    consumed_at = now(),
                    args_ciphertext = NULL,
                    args_nonce = NULL
                WHERE id = %s
                  AND status = 'approved'
                  AND expires_at > now()
                RETURNING id
                """,
                (approval_id,),
            )
            row = await cur.fetchone()
            await conn.commit()
        return row is not None

    async def _get_status(self, approval_id: str) -> tuple[str, datetime] | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT status, expires_at FROM pending_approvals WHERE id = %s",
                (approval_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return str(row["status"]), row["expires_at"]

    async def request_and_wait(
        self,
        ctx: CallContext,
        tool: str,
        args: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Handle a `require_approval` tool call for a client without MCP elicitation.

        1. Sweep expired approvals.
        2. If a matching approved row already exists (retry path), atomically consume it and execute.
        3. Otherwise create an encrypted `pending_approvals` row, audit `approval.requested`,
           and hold up to `wait_seconds` for an approve/deny decision.
        4. On timeout, return a clear error containing `<ZOS_BASE_URL>/admin/approvals/<id>`
           so the assistant shows the link to the user to approve and retry.
        """
        if self._vault is None:
            raise RuntimeError("ZOS_SECRET_KEY is required for async approvals")

        await self.sweep_expired()

        # Fast path: retry after out-of-band approval
        consumed_id = await self.try_consume_approved(ctx, tool, args)
        if consumed_id is not None:
            return True, None

        approval_id = secrets.token_urlsafe(24)
        digest = args_digest(args)
        blob = self._vault.encrypt(
            args,
            principal_id=ctx.principal_id,
            issuer=ISSUER_PENDING_APPROVAL,
            account=approval_id,
        )
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=self.ttl_seconds)

        wait_ev = asyncio.Event()
        self._waiters[approval_id] = wait_ev

        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO pending_approvals (
                        id, principal_id, space_id, client_id, tool,
                        args_digest, args_ciphertext, args_nonce, key_version,
                        status, created_at, expires_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s)
                    """,
                    (
                        approval_id,
                        ctx.principal_id,
                        ctx.space_id,
                        ctx.client_id,
                        tool,
                        digest,
                        blob.ciphertext,
                        blob.nonce,
                        blob.key_version,
                        now,
                        expires_at,
                    ),
                )
                await conn.commit()

            await record_event(
                self.pool,
                principal_id=ctx.principal_id,
                space_id=ctx.space_id,
                client_id=ctx.client_id,
                event_type="approval.requested",
                tool=tool,
                decision="require_approval",
                outcome="ok",
                raw_args_digest=digest,
                payload={"approval_id": approval_id},
            )

            if self.wait_seconds > 0:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self.wait_seconds
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(wait_ev.wait(), timeout=min(remaining, 1.0))
                    wait_ev.clear()

                    state = await self._get_status(approval_id)
                    if state is None:
                        break
                    status, row_exp = state
                    if status == "approved":
                        if await self._try_consume_by_id(approval_id):
                            return True, None
                        # Expired or consumed concurrently
                        break
                    if status == "denied":
                        return False, f"Call to '{tool}' was denied by the owner."
                    if status in ("expired", "consumed") or row_exp <= datetime.now(UTC):
                        break
        finally:
            self._waiters.pop(approval_id, None)

        url = self.approval_url(approval_id)
        return (
            False,
            f"Approval pending for '{tool}'. "
            f"Please share this link with the user so they can review and approve it: {url} "
            f"— after they approve it, retry the exact same '{tool}' call.",
        )

    async def decide(self, approval_id: str, *, approve: bool) -> dict[str, Any]:
        """Approve or deny a pending approval, purging encrypted args and waking any waiter."""
        await self.sweep_expired()
        new_status = "approved" if approve else "denied"
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_approvals
                SET status = %s,
                    decided_at = now(),
                    args_ciphertext = NULL,
                    args_nonce = NULL
                WHERE id = %s
                  AND status = 'pending'
                  AND expires_at > now()
                RETURNING id, principal_id, space_id, client_id, tool, args_digest,
                          status, created_at, expires_at, decided_at
                """,
                (new_status, approval_id),
            )
            row = await cur.fetchone()
            await conn.commit()

        if row is None:
            existing = await self._get_status(approval_id)
            if existing is None:
                raise ApprovalError("Approval request not found.")
            status, _ = existing
            raise ApprovalError(f"Approval request is already '{status}' and cannot be changed.")

        row_dict = dict(row)
        latency_ms = max(
            0,
            int((row_dict["decided_at"] - row_dict["created_at"]).total_seconds() * 1000),
        )
        await record_event(
            self.pool,
            principal_id=row_dict["principal_id"],
            space_id=row_dict["space_id"],
            client_id=row_dict["client_id"],
            event_type="approval.decided",
            tool=row_dict["tool"],
            decision=new_status,
            outcome="ok" if approve else "denied",
            raw_args_digest=bytes(row_dict["args_digest"]),
            payload={
                "approval_id": approval_id,
                "outcome": new_status,
                "latency_ms": latency_ms,
            },
        )

        ev = self._waiters.get(approval_id)
        if ev is not None:
            ev.set()

        row_dict["latency_ms"] = latency_ms
        return row_dict

    async def count_pending(self) -> int:
        await self.sweep_expired()
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM pending_approvals
                WHERE status = 'pending' AND expires_at > now()
                """
            )
            row = await cur.fetchone()
        return int(row["cnt"]) if row is not None else 0

    async def list_approvals(self, *, recent_limit: int = 50) -> dict[str, list[dict[str, Any]]]:
        await self.sweep_expired()
        clamped = max(1, min(int(recent_limit), 200))
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT
                    pa.id,
                    pa.client_id,
                    c.name AS client_name,
                    pa.tool,
                    pa.args_digest,
                    pa.status,
                    pa.created_at,
                    pa.expires_at,
                    pa.decided_at,
                    pa.consumed_at
                FROM pending_approvals pa
                JOIN clients c ON c.id = pa.client_id
                WHERE pa.status = 'pending' AND pa.expires_at > now()
                ORDER BY pa.created_at DESC
                """
            )
            pending = [self._format_summary(dict(r)) for r in await cur.fetchall()]

            cur = await conn.execute(
                """
                SELECT
                    pa.id,
                    pa.client_id,
                    c.name AS client_name,
                    pa.tool,
                    pa.args_digest,
                    pa.status,
                    pa.created_at,
                    pa.expires_at,
                    pa.decided_at,
                    pa.consumed_at
                FROM pending_approvals pa
                JOIN clients c ON c.id = pa.client_id
                WHERE pa.status <> 'pending'
                ORDER BY COALESCE(pa.decided_at, pa.expires_at, pa.created_at) DESC
                LIMIT %s
                """,
                (clamped,),
            )
            recent = [self._format_summary(dict(r)) for r in await cur.fetchall()]

        return {"pending": pending, "recent": recent}

    async def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        await self.sweep_expired()
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT
                    pa.id,
                    pa.principal_id,
                    pa.client_id,
                    c.name AS client_name,
                    pa.tool,
                    pa.args_digest,
                    pa.args_ciphertext,
                    pa.args_nonce,
                    pa.key_version,
                    pa.status,
                    pa.created_at,
                    pa.expires_at,
                    pa.decided_at,
                    pa.consumed_at
                FROM pending_approvals pa
                JOIN clients c ON c.id = pa.client_id
                WHERE pa.id = %s
                """,
                (approval_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None

        item = self._format_summary(dict(row))
        args_obj: dict[str, Any] | None = None
        args_pretty: str | None = None
        if (
            item["status"] == "pending"
            and row["args_ciphertext"] is not None
            and row["args_nonce"] is not None
            and self._vault is not None
        ):
            args_obj = self._vault.decrypt(
                ciphertext=row["args_ciphertext"],
                nonce=row["args_nonce"],
                key_version=row["key_version"],
                principal_id=row["principal_id"],
                issuer=ISSUER_PENDING_APPROVAL,
                account=str(row["id"]),
            )
            args_pretty = json.dumps(args_obj, indent=2, sort_keys=True, ensure_ascii=False)
        item["args"] = args_obj
        item["args_pretty"] = args_pretty
        return item

    @staticmethod
    def _format_summary(row: dict[str, Any]) -> dict[str, Any]:
        digest = row.get("args_digest")
        return {
            "id": str(row["id"]),
            "client_id": row["client_id"],
            "client_name": str(row["client_name"]),
            "tool": str(row["tool"]),
            "args_digest_hex": bytes(digest).hex() if digest is not None else "",
            "status": str(row["status"]),
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "decided_at": row["decided_at"],
            "consumed_at": row["consumed_at"],
        }
