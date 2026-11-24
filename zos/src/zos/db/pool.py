"""Async PostgreSQL connection pool using psycopg 3."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool


def create_pool(
    database_url: str,
    *,
    min_size: int = 1,
    max_size: int = 10,
) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    """Create an unopened AsyncConnectionPool returning dict rows."""
    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=min_size,
        max_size=max_size,
        open=False,
        kwargs={"row_factory": dict_row},
    )


@asynccontextmanager
async def open_pool(
    database_url: str,
    *,
    min_size: int = 1,
    max_size: int = 10,
) -> AsyncIterator[AsyncConnectionPool[AsyncConnection[DictRow]]]:
    """Async context manager that opens and cleanly closes a pool."""
    pool = create_pool(database_url, min_size=min_size, max_size=max_size)
    await pool.open()
    try:
        yield pool
    finally:
        await pool.close()
