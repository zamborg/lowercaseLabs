"""Postgres storage layer for the ZOS Tasks module (uses only its own schema)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

VALID_STATUSES = frozenset({"open", "done", "cancelled"})
MAX_TITLE_LEN = 500
MAX_NOTES_LEN = 20_000
MAX_TAG_LEN = 64
MAX_LIST_LIMIT = 100

_UNSET: Any = object()

MIGRATIONS: list[tuple[str, str]] = [
    (
        "0001_init",
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id BIGSERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done', 'cancelled')),
            due DATE NULL,
            tags TEXT[] NOT NULL DEFAULT '{}',
            created_by TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ NULL
        );

        CREATE INDEX IF NOT EXISTS tasks_status_due_idx ON tasks (status, due NULLS LAST, id DESC);
        CREATE INDEX IF NOT EXISTS tasks_tags_gin ON tasks USING GIN (tags);
        """,
    ),
]


def normalize_title(raw: str) -> str:
    cleaned = str(raw).strip()
    if not cleaned:
        raise ValueError("Task title must be non-empty")
    if len(cleaned) > MAX_TITLE_LEN:
        raise ValueError(f"Task title must be <= {MAX_TITLE_LEN} characters")
    return cleaned


def normalize_notes(raw: str | None) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if len(text) > MAX_NOTES_LEN:
        raise ValueError(f"Task notes must be <= {MAX_NOTES_LEN} characters")
    return text


def normalize_status(raw: str) -> str:
    cleaned = str(raw).strip().lower()
    if cleaned not in VALID_STATUSES:
        raise ValueError(f"Invalid status {raw!r}; must be one of: open, done, cancelled")
    return cleaned


def normalize_due(raw: str | date | None) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    cleaned = str(raw).strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid due date {raw!r}; expected ISO date YYYY-MM-DD") from exc


def normalize_tags(raw: Sequence[str] | str | None) -> list[str]:
    if raw is None:
        return []
    parts = raw.split(",") if isinstance(raw, str) else list(raw)
    cleaned: set[str] = set()
    for item in parts:
        t = str(item).strip().lower()
        if not t:
            continue
        if len(t) > MAX_TAG_LEN:
            raise ValueError(f"Tag {t!r} exceeds {MAX_TAG_LEN} characters")
        cleaned.add(t)
    return sorted(cleaned)


def _iso(val: Any) -> str | None:
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, date):
        return val.isoformat()
    return None if val is None else str(val)


def _format_task(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "title": str(row["title"]),
        "notes": str(row["notes"] or ""),
        "status": str(row["status"]),
        "due": _iso(row.get("due")),
        "tags": list(row.get("tags") or []),
        "created_by": str(row["created_by"]),
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
        "completed_at": _iso(row.get("completed_at")),
    }


def create_tasks_pool(database_url: str) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"row_factory": dict_row},
    )


class TasksStore:
    """Async Postgres store for the human owner's personal todo list."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> None:
        self.pool = pool

    async def init_schema(self) -> list[str]:
        """Run idempotent, versioned schema migrations inside the module's own schema."""
        applied_now: list[str] = []
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks_schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.commit()

            cur = await conn.execute("SELECT version FROM tasks_schema_migrations")
            already = {str(r["version"]) for r in await cur.fetchall()}

            for version, sql_body in MIGRATIONS:
                if version in already:
                    continue
                async with conn.transaction():
                    await conn.execute(sql_body)
                    await conn.execute(
                        "INSERT INTO tasks_schema_migrations (version) VALUES (%s)",
                        (version,),
                    )
                applied_now.append(version)
        return applied_now

    async def add_task(
        self,
        *,
        title: str,
        created_by: str,
        notes: str | None = None,
        due: str | date | None = None,
        tags: Sequence[str] | str | None = None,
    ) -> dict[str, Any]:
        clean_title = normalize_title(title)
        clean_notes = normalize_notes(notes)
        clean_due = normalize_due(due)
        clean_tags = normalize_tags(tags)
        clean_creator = str(created_by).strip() or "owner"

        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                INSERT INTO tasks (title, notes, status, due, tags, created_by)
                VALUES (%s, %s, 'open', %s, %s, %s)
                RETURNING id, title, notes, status, due, tags, created_by, created_at, updated_at, completed_at
                """,
                (clean_title, clean_notes, clean_due, clean_tags, clean_creator),
            )
            row = await cur.fetchone()
            assert row is not None
        return _format_task(dict(row))

    async def get_task(self, task_id: int) -> dict[str, Any]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, title, notes, status, due, tags, created_by, created_at, updated_at, completed_at
                FROM tasks
                WHERE id = %s
                """,
                (int(task_id),),
            )
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Task {task_id} not found")
        return _format_task(dict(row))

    async def list_tasks(
        self,
        *,
        status: str | None = "open",
        due_before: str | date | None = None,
        tag: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List tasks with optional status/due_before/tag filters and keyset cursor pagination."""
        clamped_limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        where: list[str] = []
        params: list[Any] = []

        if status is not None:
            clean_status = str(status).strip().lower()
            if clean_status and clean_status != "all":
                where.append("status = %s")
                params.append(normalize_status(clean_status))

        if due_before is not None:
            parsed_before = normalize_due(due_before)
            if parsed_before is not None:
                where.append("due IS NOT NULL AND due <= %s")
                params.append(parsed_before)

        if tag is not None and str(tag).strip():
            where.append("%s = ANY(tags)")
            params.append(str(tag).strip().lower())

        if cursor is not None and str(cursor).strip():
            try:
                cursor_id = int(str(cursor).strip())
            except ValueError as exc:
                raise ValueError(f"Invalid cursor: {cursor!r}") from exc
            where.append("id < %s")
            params.append(cursor_id)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(clamped_limit + 1)

        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT id, title, notes, status, due, tags, created_by, created_at, updated_at, completed_at
                FROM tasks
                {where_sql}
                ORDER BY id DESC
                LIMIT %s
                """,
                params,
            )
            rows = [dict(r) for r in await cur.fetchall()]

        has_more = len(rows) > clamped_limit
        page = rows[:clamped_limit]
        next_cursor = str(page[-1]["id"]) if has_more and page else None
        return {
            "tasks": [_format_task(r) for r in page],
            "next_cursor": next_cursor,
        }

    async def update_task(
        self,
        task_id: int,
        *,
        title: str = _UNSET,
        notes: str | None = _UNSET,
        due: str | date | None = _UNSET,
        tags: Sequence[str] | str | None = _UNSET,
        status: str = _UNSET,
    ) -> dict[str, Any]:
        """Update selected fields on a task, maintaining updated_at and completed_at."""
        sets: list[str] = []
        params: list[Any] = []

        if title is not _UNSET:
            sets.append("title = %s")
            params.append(normalize_title(title))
        if notes is not _UNSET:
            sets.append("notes = %s")
            params.append(normalize_notes(notes))
        if due is not _UNSET:
            sets.append("due = %s")
            params.append(normalize_due(due))
        if tags is not _UNSET:
            sets.append("tags = %s")
            params.append(normalize_tags(tags))
        if status is not _UNSET:
            clean_status = normalize_status(status)
            sets.append("status = %s")
            params.append(clean_status)
            if clean_status == "done":
                sets.append("completed_at = COALESCE(completed_at, now())")
            else:
                sets.append("completed_at = NULL")

        if not sets:
            return await self.get_task(task_id)

        sets.append("updated_at = now()")
        params.append(int(task_id))

        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                f"""
                UPDATE tasks
                SET {', '.join(sets)}
                WHERE id = %s
                RETURNING id, title, notes, status, due, tags, created_by, created_at, updated_at, completed_at
                """,
                params,
            )
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Task {task_id} not found")
        return _format_task(dict(row))

    async def complete_task(self, task_id: int) -> dict[str, Any]:
        return await self.update_task(task_id, status="done")

    async def reopen_task(self, task_id: int) -> dict[str, Any]:
        return await self.update_task(task_id, status="open")

    async def delete_task(self, task_id: int) -> bool:
        """Delete a task permanently (owner UI only; never exposed over MCP)."""
        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute("DELETE FROM tasks WHERE id = %s", (int(task_id),))
            return (cur.rowcount or 0) > 0

    async def list_for_ui(
        self,
        *,
        tag: str | None = None,
        done_limit: int = 50,
    ) -> dict[str, Any]:
        """Fetch all open tasks (sorted by due date) and recent done/cancelled tasks for the owner UI."""
        tag_filter = str(tag).strip().lower() if tag else ""
        async with self.pool.connection() as conn:
            if tag_filter:
                cur_open = await conn.execute(
                    """
                    SELECT id, title, notes, status, due, tags, created_by, created_at, updated_at, completed_at
                    FROM tasks
                    WHERE status = 'open' AND %s = ANY(tags)
                    ORDER BY due ASC NULLS LAST, id DESC
                    """,
                    (tag_filter,),
                )
                cur_done = await conn.execute(
                    """
                    SELECT id, title, notes, status, due, tags, created_by, created_at, updated_at, completed_at
                    FROM tasks
                    WHERE status IN ('done', 'cancelled') AND %s = ANY(tags)
                    ORDER BY COALESCE(completed_at, updated_at) DESC, id DESC
                    LIMIT %s
                    """,
                    (tag_filter, int(done_limit)),
                )
            else:
                cur_open = await conn.execute(
                    """
                    SELECT id, title, notes, status, due, tags, created_by, created_at, updated_at, completed_at
                    FROM tasks
                    WHERE status = 'open'
                    ORDER BY due ASC NULLS LAST, id DESC
                    """
                )
                cur_done = await conn.execute(
                    """
                    SELECT id, title, notes, status, due, tags, created_by, created_at, updated_at, completed_at
                    FROM tasks
                    WHERE status IN ('done', 'cancelled')
                    ORDER BY COALESCE(completed_at, updated_at) DESC, id DESC
                    LIMIT %s
                    """,
                    (int(done_limit),),
                )
            cur_tags = await conn.execute(
                """
                SELECT DISTINCT unnest(tags) AS tag
                FROM tasks
                ORDER BY tag ASC
                """
            )
            open_rows = [_format_task(dict(r)) for r in await cur_open.fetchall()]
            done_rows = [_format_task(dict(r)) for r in await cur_done.fetchall()]
            all_tags = [str(r["tag"]) for r in await cur_tags.fetchall()]

        return {
            "open": open_rows,
            "done": done_rows,
            "tags": all_tags,
        }
