"""Plain numbered SQL migration runner backed by schema_migrations."""

from __future__ import annotations

from pathlib import Path

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool


async def run_migrations(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    migrations_dir: Path,
) -> list[str]:
    """Apply pending *.sql migrations in lexicographical order.

    Returns the list of migration filenames applied in this run.
    """
    sql_files = sorted(p for p in migrations_dir.glob("*.sql") if p.is_file())
    applied_now: list[str] = []

    async with pool.connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.commit()

        cur = await conn.execute("SELECT filename FROM schema_migrations")
        rows = await cur.fetchall()
        already_applied = {row["filename"] for row in rows}

        for sql_file in sql_files:
            if sql_file.name in already_applied:
                continue
            sql = sql_file.read_text(encoding="utf-8")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (sql_file.name,),
                )
            applied_now.append(sql_file.name)

    return applied_now
