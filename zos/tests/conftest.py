"""Shared pytest fixtures for ZOS unit, integration, and e2e tests."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.db import open_pool, run_migrations

DEFAULT_TEST_DB_URL = "postgresql://zos:zos@127.0.0.1:55432/zos_test"
MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def get_test_database_url() -> str:
    return os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DB_URL)


@pytest.fixture
async def db_pool() -> AsyncIterator[AsyncConnectionPool[AsyncConnection[DictRow]]]:
    """Open a pool against local Postgres 17, ensure schema is migrated, and truncate tables."""
    url = get_test_database_url()
    async with open_pool(url, min_size=1, max_size=5) as pool:
        await run_migrations(pool, MIGRATIONS_DIR)
        async with pool.connection() as conn:
            await conn.execute("TRUNCATE principals RESTART IDENTITY CASCADE")
            await conn.execute("TRUNCATE dcr_rejected_attempts RESTART IDENTITY")
            await conn.execute("DELETE FROM allowed_redirect_hosts")
            await conn.execute(
                """
                INSERT INTO allowed_redirect_hosts (host, source)
                VALUES ('chatgpt.com', 'default'), ('claude.ai', 'default')
                ON CONFLICT (host) DO NOTHING
                """
            )
            await conn.commit()
        yield pool
