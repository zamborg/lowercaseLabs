"""Single-instance Postgres session advisory lock for ZOS.

ZOS keeps MCP sessions, async approval waiters, actuator WebSockets, and upstream
MCP sessions in memory, so exactly one ZOS process may run per deployment/database.
"""

from __future__ import annotations

import contextlib

import psycopg
from psycopg.rows import dict_row

# ASCII "ZOS_INST" as signed 64-bit BIGINT
ZOS_INSTANCE_LOCK_KEY = 0x5A4F535F494E5354


class SingleInstanceError(RuntimeError):
    """Raised on startup if another ZOS instance already holds the database advisory lock."""


class InstanceAdvisoryLock:
    """Holds a dedicated Postgres connection with a session-level advisory lock."""

    def __init__(self, database_url: str, *, lock_key: int = ZOS_INSTANCE_LOCK_KEY) -> None:
        self.database_url = database_url
        self.lock_key = lock_key
        self._conn: psycopg.AsyncConnection[dict[str, object]] | None = None

    async def acquire(self) -> None:
        if self._conn is not None:
            return
        conn = await psycopg.AsyncConnection.connect(
            self.database_url,
            autocommit=True,
            row_factory=dict_row,
        )
        try:
            cur = await conn.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired",
                (self.lock_key,),
            )
            row = await cur.fetchone()
            acquired = bool(row and row.get("acquired"))
            if not acquired:
                await conn.close()
                raise SingleInstanceError(
                    "Another ZOS instance is already running against this database "
                    f"(Postgres advisory lock {self.lock_key} is held). "
                    "ZOS is strictly single-instance per deployment."
                )
            self._conn = conn
        except Exception:
            if self._conn is None:
                with contextlib.suppress(Exception):
                    await conn.close()
            raise

    async def release(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is None:
            return
        with contextlib.suppress(Exception):
            await conn.execute("SELECT pg_advisory_unlock(%s)", (self.lock_key,))
        with contextlib.suppress(Exception):
            await conn.close()
