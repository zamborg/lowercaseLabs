"""Shared memory write/import service layer backing both `memory_put` and `zos memory import`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.audit import record_event
from zos.db.queries import get_principal_and_space

PutStatus = Literal["created", "updated", "unchanged"]


@dataclass(frozen=True, slots=True)
class MemoryImportItem:
    """Normalized memory record produced by a MemoryImporter."""

    key: str
    content: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryImportSummary:
    discovered: int
    created: int
    updated: int
    unchanged: int


def normalize_tags(raw_tags: Iterable[str] | None) -> list[str]:
    if not raw_tags:
        return []
    return sorted({str(t).strip() for t in raw_tags if str(t).strip()})


class MemoryService:
    """Single write path for memories so MCP `memory_put` and CLI imports share logic and auditing."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> None:
        self.pool = pool

    async def put_memory(
        self,
        *,
        principal_id: UUID,
        space_id: UUID,
        client_id: UUID,
        key: str,
        content: str,
        tags: Iterable[str] | None = None,
    ) -> tuple[dict[str, Any], PutStatus]:
        """Idempotently create or update a memory entry.

        Returns `(row_with_author, status)` where status is `'created'`, `'updated'`,
        or `'unchanged'` (when content and normalized tags are identical).
        """
        clean_key = key.strip()
        if not clean_key or not isinstance(content, str):
            raise ValueError("key must be non-empty and content must be a string")
        norm_tags = normalize_tags(tags)

        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                SELECT m.id, m.key, m.content, m.tags, m.created_at, m.updated_at,
                       c.name AS author
                FROM memories m
                LEFT JOIN clients c ON c.id = m.created_by_client_id
                WHERE m.principal_id = %s AND m.space_id = %s AND m.key = %s
                FOR UPDATE OF m
                """,
                (principal_id, space_id, clean_key),
            )
            existing = await cur.fetchone()

            if existing is not None:
                existing_tags = normalize_tags(existing["tags"])
                if existing["content"] == content and existing_tags == norm_tags:
                    return dict(existing), "unchanged"

                cur = await conn.execute(
                    """
                    UPDATE memories
                    SET content = %s,
                        tags = %s,
                        updated_at = now()
                    WHERE id = %s
                    RETURNING id, key, content, tags, created_at, updated_at, created_by_client_id
                    """,
                    (content, norm_tags, existing["id"]),
                )
                updated = await cur.fetchone()
                assert updated is not None
                out = dict(updated)
                out["author"] = existing["author"]
                return out, "updated"

            cur = await conn.execute(
                """
                INSERT INTO memories (principal_id, space_id, key, content, tags, created_by_client_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, key, content, tags, created_at, updated_at, created_by_client_id
                """,
                (principal_id, space_id, clean_key, content, norm_tags, client_id),
            )
            inserted = await cur.fetchone()
            assert inserted is not None

            c_cur = await conn.execute("SELECT name FROM clients WHERE id = %s", (client_id,))
            c_row = await c_cur.fetchone()
            out = dict(inserted)
            out["author"] = c_row["name"] if c_row else None
            return out, "created"

    async def get_or_create_import_client(
        self,
        *,
        principal_id: UUID,
        space_id: UUID,
        name: str = "memory-import",
    ) -> dict[str, Any]:
        """Ensure a dedicated bearer client exists for attributing imported memories."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, name, kind, default_space_id, trust_ceiling
                FROM clients
                WHERE principal_id = %s AND name = %s AND kind = 'bearer' AND disabled_at IS NULL
                """,
                (principal_id, name),
            )
            existing = await cur.fetchone()
            if existing is not None:
                return dict(existing)

            cur = await conn.execute(
                """
                INSERT INTO clients (principal_id, default_space_id, name, kind, trust_ceiling)
                VALUES (%s, %s, %s, 'bearer', 'owner')
                ON CONFLICT (principal_id, name) WHERE kind = 'bearer' AND disabled_at IS NULL DO UPDATE
                SET last_seen_at = now()
                RETURNING id, name, kind, default_space_id, trust_ceiling
                """,
                (principal_id, space_id, name),
            )
            row = await cur.fetchone()
            await conn.commit()
            assert row is not None
            return dict(row)

    async def _resolve_memory_provider_id(self, principal_id: UUID) -> UUID | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id FROM providers WHERE principal_id = %s AND slug = 'memory' AND kind = 'hosted' LIMIT 1",
                (principal_id,),
            )
            row = await cur.fetchone()
        return row["id"] if row else None

    async def import_items(
        self,
        items: list[MemoryImportItem],
        *,
        source: str,
        principal_slug: str | None = None,
        client_name: str = "memory-import",
    ) -> MemoryImportSummary:
        """Idempotently import normalized memory items with attribution and per-change auditing."""
        principal, space = await get_principal_and_space(self.pool, principal_slug)
        p_id: UUID = principal["id"]
        s_id: UUID = space["id"]
        client = await self.get_or_create_import_client(
            principal_id=p_id,
            space_id=s_id,
            name=client_name,
        )
        provider_id = await self._resolve_memory_provider_id(p_id)

        created = 0
        updated = 0
        unchanged = 0

        for item in items:
            _row, status = await self.put_memory(
                principal_id=p_id,
                space_id=s_id,
                client_id=client["id"],
                key=item.key,
                content=item.content,
                tags=item.tags,
            )
            if status == "created":
                created += 1
            elif status == "updated":
                updated += 1
            else:
                unchanged += 1

            if status in ("created", "updated"):
                await record_event(
                    self.pool,
                    principal_id=p_id,
                    space_id=s_id,
                    client_id=client["id"],
                    provider_id=provider_id,
                    event_type="memory.imported",
                    tool="memory_put",
                    decision="allow",
                    outcome="ok",
                    args={"key": item.key, "content": item.content, "tags": list(item.tags)},
                    payload={"source": source, "action": status, "key": item.key},
                )

        return MemoryImportSummary(
            discovered=len(items),
            created=created,
            updated=updated,
            unchanged=unchanged,
        )
