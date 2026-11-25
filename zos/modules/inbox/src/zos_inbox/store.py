"""Postgres storage layer for the ZOS Inbox module (uses only its own schema)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mailboxes (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    thread_id UUID NOT NULL,
    sender TEXT NOT NULL,
    recipients TEXT[] NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_thread_idx ON messages (thread_id, id);
CREATE INDEX IF NOT EXISTS messages_recipients_gin ON messages USING GIN (recipients);

CREATE TABLE IF NOT EXISTS message_states (
    message_id BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    reader TEXT NOT NULL,
    read_at TIMESTAMPTZ NULL,
    archived_at TIMESTAMPTZ NULL,
    PRIMARY KEY (message_id, reader)
);

CREATE INDEX IF NOT EXISTS message_states_reader_idx ON message_states (reader, message_id);
"""


def normalize_mailbox(name: str) -> str:
    cleaned = name.strip().lower()
    if not cleaned:
        raise ValueError("Mailbox name must be non-empty")
    if len(cleaned) > 80:
        raise ValueError("Mailbox name must be <= 80 characters")
    if any(ord(ch) < 32 or ch in (",", ";") for ch in cleaned):
        raise ValueError(f"Invalid mailbox name: {name!r}")
    return cleaned


def _iso(val: Any) -> str | None:
    if isinstance(val, datetime):
        return val.isoformat()
    return None if val is None else str(val)


def _format_message(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "thread_id": str(row["thread_id"]),
        "from": str(row["sender"]),
        "to": list(row["recipients"]),
        "subject": str(row["subject"]),
        "body": str(row["body"]),
        "created_at": _iso(row["created_at"]),
        "read": row.get("read_at") is not None,
        "read_at": _iso(row.get("read_at")),
        "archived": row.get("archived_at") is not None,
        "archived_at": _iso(row.get("archived_at")),
    }


def create_inbox_pool(database_url: str) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"row_factory": dict_row},
    )


class InboxStore:
    """Async Postgres store for mailboxes, email-like messages, and per-reader state."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> None:
        self.pool = pool

    async def init_schema(self) -> None:
        async with self.pool.connection() as conn:
            for stmt in SCHEMA_SQL.split(";"):
                s = stmt.strip()
                if s:
                    await conn.execute(s)
            for seed in ("owner", "general"):
                await conn.execute(
                    "INSERT INTO mailboxes (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                    (seed,),
                )
            await conn.commit()

    async def ensure_mailbox(self, name: str) -> str:
        clean = normalize_mailbox(name)
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO mailboxes (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                (clean,),
            )
            await conn.commit()
        return clean

    async def send_message(
        self,
        *,
        sender: str,
        to: list[str],
        subject: str,
        body: str,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        clean_sender = normalize_mailbox(sender)
        if not to:
            raise ValueError("At least one recipient mailbox is required")
        clean_to: list[str] = []
        seen: set[str] = set()
        for raw in to:
            mb = normalize_mailbox(raw)
            if mb not in seen:
                seen.add(mb)
                clean_to.append(mb)

        clean_subject = subject.strip()
        if not clean_subject:
            raise ValueError("Subject must be non-empty")

        tid = uuid.UUID(thread_id.strip()) if thread_id and thread_id.strip() else uuid.uuid4()

        async with self.pool.connection() as conn, conn.transaction():
            for mb in {clean_sender, *clean_to}:
                await conn.execute(
                    """
                    INSERT INTO mailboxes (name, last_activity_at)
                    VALUES (%s, now())
                    ON CONFLICT (name) DO UPDATE SET last_activity_at = now()
                    """,
                    (mb,),
                )
            cur = await conn.execute(
                """
                INSERT INTO messages (thread_id, sender, recipients, subject, body)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, thread_id, sender, recipients, subject, body, created_at
                """,
                (tid, clean_sender, clean_to, clean_subject, body),
            )
            row = await cur.fetchone()
            assert row is not None
            # Sender has implicitly read their own sent message
            await conn.execute(
                """
                INSERT INTO message_states (message_id, reader, read_at)
                VALUES (%s, %s, now())
                ON CONFLICT (message_id, reader) DO UPDATE SET read_at = COALESCE(message_states.read_at, now())
                """,
                (row["id"], clean_sender),
            )

        out = dict(row)
        out["read_at"] = out["created_at"]
        out["archived_at"] = None
        return _format_message(out)

    async def list_messages(
        self,
        *,
        caller: str,
        mailbox: str | None = None,
        unread_only: bool = True,
        include_archived: bool = False,
        limit: int = 20,
        cursor: str | None = None,
        allow_any_mailbox: bool = False,
    ) -> dict[str, Any]:
        clean_caller = normalize_mailbox(caller)
        target_mb = normalize_mailbox(mailbox) if mailbox and mailbox.strip() else clean_caller
        if not allow_any_mailbox and target_mb not in (clean_caller, "general"):
            raise PermissionError(
                f"Access denied: client '{clean_caller}' may list only its own mailbox or 'general'"
            )

        await self.ensure_mailbox(clean_caller)
        await self.ensure_mailbox(target_mb)

        capped_limit = max(1, min(int(limit), 100))
        reader = target_mb if (allow_any_mailbox and target_mb != "general") else clean_caller

        where = ["%s = ANY(m.recipients)"]
        params: list[Any] = [reader, target_mb]
        if not include_archived:
            where.append("s.archived_at IS NULL")
        if unread_only:
            where.append("s.read_at IS NULL")
        if cursor is not None and str(cursor).strip():
            try:
                before_id = int(str(cursor).strip())
            except ValueError as exc:
                raise ValueError(f"Invalid cursor {cursor!r}") from exc
            where.append("m.id < %s")
            params.append(before_id)

        params.append(capped_limit + 1)
        sql = f"""
            SELECT m.id, m.thread_id, m.sender, m.recipients, m.subject, m.body, m.created_at,
                   s.read_at, s.archived_at
            FROM messages m
            LEFT JOIN message_states s
              ON s.message_id = m.id AND s.reader = %s
            WHERE {' AND '.join(where)}
            ORDER BY m.id DESC
            LIMIT %s
        """
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, tuple(params))
            rows = [dict(r) for r in await cur.fetchall()]

        has_more = len(rows) > capped_limit
        page = rows[:capped_limit]
        next_cursor = str(page[-1]["id"]) if has_more and page else None
        return {
            "mailbox": target_mb,
            "messages": [_format_message(r) for r in page],
            "next_cursor": next_cursor,
        }

    async def list_all_messages(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Owner-only unified view across EVERY mailbox (including client-to-client messages).

        Never modifies any client's `message_states`.
        """
        capped_limit = max(1, min(int(limit), 100))
        where: list[str] = []
        params: list[Any] = []
        if cursor is not None and str(cursor).strip():
            before_id = int(str(cursor).strip())
            where.append("m.id < %s")
            params.append(before_id)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(capped_limit + 1)
        sql = f"""
            SELECT m.id, m.thread_id, m.sender, m.recipients, m.subject, m.body, m.created_at,
                   s.read_at, s.archived_at
            FROM messages m
            LEFT JOIN message_states s
              ON s.message_id = m.id AND s.reader = 'owner'
            {where_sql}
            ORDER BY m.id DESC
            LIMIT %s
        """
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, tuple(params))
            rows = [dict(r) for r in await cur.fetchall()]

        has_more = len(rows) > capped_limit
        page = rows[:capped_limit]
        next_cursor = str(page[-1]["id"]) if has_more and page else None
        return {
            "mailbox": "all",
            "messages": [_format_message(r) for r in page],
            "next_cursor": next_cursor,
        }

    def _can_access(self, row: dict[str, Any], caller: str, *, allow_any: bool) -> bool:
        if allow_any:
            return True
        recipients = [str(x).lower() for x in row["recipients"]]
        return caller in recipients or "general" in recipients or str(row["sender"]).lower() == caller

    async def read_message(
        self,
        *,
        caller: str,
        message_id: int,
        allow_any: bool = False,
    ) -> dict[str, Any]:
        clean_caller = normalize_mailbox(caller)
        await self.ensure_mailbox(clean_caller)

        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                SELECT id, thread_id, sender, recipients, subject, body, created_at
                FROM messages
                WHERE id = %s
                """,
                (int(message_id),),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(f"Message {message_id} not found")
            if not self._can_access(dict(row), clean_caller, allow_any=allow_any):
                raise PermissionError(f"Access denied for message {message_id}")

            st_cur = await conn.execute(
                """
                INSERT INTO message_states (message_id, reader, read_at)
                VALUES (%s, %s, now())
                ON CONFLICT (message_id, reader)
                DO UPDATE SET read_at = COALESCE(message_states.read_at, now())
                RETURNING read_at, archived_at
                """,
                (int(message_id), clean_caller),
            )
            st = await st_cur.fetchone()

        merged = dict(row)
        merged["read_at"] = st["read_at"] if st else None
        merged["archived_at"] = st["archived_at"] if st else None
        return _format_message(merged)

    async def archive_message(
        self,
        *,
        caller: str,
        message_id: int,
        allow_any: bool = False,
    ) -> dict[str, Any]:
        clean_caller = normalize_mailbox(caller)
        await self.ensure_mailbox(clean_caller)

        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                SELECT id, thread_id, sender, recipients, subject, body, created_at
                FROM messages
                WHERE id = %s
                """,
                (int(message_id),),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(f"Message {message_id} not found")
            if not self._can_access(dict(row), clean_caller, allow_any=allow_any):
                raise PermissionError(f"Access denied for message {message_id}")

            st_cur = await conn.execute(
                """
                INSERT INTO message_states (message_id, reader, read_at, archived_at)
                VALUES (%s, %s, now(), now())
                ON CONFLICT (message_id, reader)
                DO UPDATE SET
                    read_at = COALESCE(message_states.read_at, now()),
                    archived_at = COALESCE(message_states.archived_at, now())
                RETURNING read_at, archived_at
                """,
                (int(message_id), clean_caller),
            )
            st = await st_cur.fetchone()

        merged = dict(row)
        merged["read_at"] = st["read_at"] if st else None
        merged["archived_at"] = st["archived_at"] if st else None
        return _format_message(merged)

    async def get_thread(
        self,
        *,
        caller: str,
        thread_id: str,
        mark_read: bool = True,
        allow_any: bool = True,
    ) -> list[dict[str, Any]]:
        clean_caller = normalize_mailbox(caller)
        tid = uuid.UUID(thread_id.strip())
        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                SELECT m.id, m.thread_id, m.sender, m.recipients, m.subject, m.body, m.created_at,
                       s.read_at, s.archived_at
                FROM messages m
                LEFT JOIN message_states s
                  ON s.message_id = m.id AND s.reader = %s
                WHERE m.thread_id = %s
                ORDER BY m.id ASC
                """,
                (clean_caller, tid),
            )
            rows = [dict(r) for r in await cur.fetchall()]
            accessible = [r for r in rows if self._can_access(r, clean_caller, allow_any=allow_any)]
            if mark_read:
                for r in accessible:
                    if r.get("read_at") is None:
                        st_cur = await conn.execute(
                            """
                            INSERT INTO message_states (message_id, reader, read_at)
                            VALUES (%s, %s, now())
                            ON CONFLICT (message_id, reader)
                            DO UPDATE SET read_at = COALESCE(message_states.read_at, now())
                            RETURNING read_at
                            """,
                            (r["id"], clean_caller),
                        )
                        st = await st_cur.fetchone()
                        if st:
                            r["read_at"] = st["read_at"]
        return [_format_message(r) for r in accessible]

    async def directory(
        self,
        *,
        caller: str | None = None,
        owner_view: bool = False,
    ) -> list[dict[str, Any]]:
        clean_caller = normalize_mailbox(caller) if caller else None
        if clean_caller:
            await self.ensure_mailbox(clean_caller)

        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT name, created_at, last_activity_at FROM mailboxes ORDER BY name ASC"
            )
            mboxes = [dict(r) for r in await cur.fetchall()]

            out: list[dict[str, Any]] = []
            for mb in mboxes:
                name = str(mb["name"])
                unread: int | None = None
                if owner_view:
                    reader = "owner" if name == "general" else name
                    ucur = await conn.execute(
                        """
                        SELECT COUNT(*) AS cnt
                        FROM messages m
                        LEFT JOIN message_states s
                          ON s.message_id = m.id AND s.reader = %s
                        WHERE %s = ANY(m.recipients)
                          AND s.read_at IS NULL
                          AND s.archived_at IS NULL
                        """,
                        (reader, name),
                    )
                    urow = await ucur.fetchone()
                    unread = int(urow["cnt"]) if urow else 0
                elif clean_caller is not None and name in (clean_caller, "general"):
                    ucur = await conn.execute(
                        """
                        SELECT COUNT(*) AS cnt
                        FROM messages m
                        LEFT JOIN message_states s
                          ON s.message_id = m.id AND s.reader = %s
                        WHERE %s = ANY(m.recipients)
                          AND s.read_at IS NULL
                          AND s.archived_at IS NULL
                        """,
                        (clean_caller, name),
                    )
                    urow = await ucur.fetchone()
                    unread = int(urow["cnt"]) if urow else 0

                out.append(
                    {
                        "mailbox": name,
                        "last_activity_at": _iso(mb["last_activity_at"]),
                        "unread_count": unread,
                    }
                )
        # Keep owner and general first for readability
        priority = {"owner": 0, "general": 1}
        out.sort(key=lambda item: (priority.get(item["mailbox"], 2), item["mailbox"]))
        return out
