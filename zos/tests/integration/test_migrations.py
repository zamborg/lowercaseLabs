"""Integration tests for M1.1 DB migrations and schema constraints."""

from __future__ import annotations

import uuid
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest
from psycopg.errors import CheckViolation, UniqueViolation

from tests.conftest import MIGRATIONS_DIR, get_test_database_url
from zos.db import open_pool, run_migrations


def _replace_db_name(url: str, new_db: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path=f"/{new_db}"))


async def test_fresh_db_migrates_and_second_run_is_noop() -> None:
    base_url = get_test_database_url()
    temp_db = f"zos_mig_{uuid.uuid4().hex[:8]}"
    admin_url = _replace_db_name(base_url, "postgres")
    temp_url = _replace_db_name(base_url, temp_db)

    async with await psycopg.AsyncConnection.connect(admin_url, autocommit=True) as admin:
        await admin.execute(f"CREATE DATABASE {temp_db}")

    try:
        async with open_pool(temp_url) as pool:
            first = await run_migrations(pool, MIGRATIONS_DIR)
            expected_files = sorted(p.name for p in MIGRATIONS_DIR.glob("*.sql"))
            assert first == expected_files
            assert "0001_init.sql" in first

            second = await run_migrations(pool, MIGRATIONS_DIR)
            assert second == []

            async with pool.connection() as conn:
                cur = await conn.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                    """
                )
                tables = {row["table_name"] for row in await cur.fetchall()}

            expected = {
                "principals",
                "spaces",
                "clients",
                "oauth_codes",
                "tokens",
                "credentials",
                "providers",
                "tool_tiers",
                "client_policies",
                "events",
                "modules",
                "allowed_redirect_hosts",
                "dcr_rejected_attempts",
                "schema_migrations",
            }
            assert expected.issubset(tables)
            assert "memories" not in tables
    finally:
        async with await psycopg.AsyncConnection.connect(admin_url, autocommit=True) as admin:
            await admin.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
                """,
                (temp_db,),
            )
            await admin.execute(f"DROP DATABASE IF EXISTS {temp_db}")


async def test_key_schema_constraints(db_pool) -> None:
    async with db_pool.connection() as conn:
        cur = await conn.execute(
            "INSERT INTO principals (slug, display_name) VALUES ('zubin', 'Zubin') RETURNING id"
        )
        principal_id = (await cur.fetchone())["id"]

        cur = await conn.execute(
            "INSERT INTO spaces (principal_id, slug, name) VALUES (%s, 'main', 'Main') RETURNING id",
            (principal_id,),
        )
        space_id = (await cur.fetchone())["id"]

        # Duplicate bearer client name for same principal fails
        cur = await conn.execute(
            """
            INSERT INTO clients (principal_id, default_space_id, name, kind)
            VALUES (%s, %s, 'cli', 'bearer') RETURNING id
            """,
            (principal_id, space_id),
        )
        client_id = (await cur.fetchone())["id"]
        await conn.commit()

        with pytest.raises(UniqueViolation):
            await conn.execute(
                """
                INSERT INTO clients (principal_id, default_space_id, name, kind)
                VALUES (%s, %s, 'cli', 'bearer')
                """,
                (principal_id, space_id),
            )
        await conn.rollback()

        # Two OAuth clients MAY share a display name (distinct oauth_client_id)
        await conn.execute(
            """
            INSERT INTO clients (principal_id, default_space_id, name, kind, oauth_client_id)
            VALUES (%s, %s, 'Claude Code', 'oauth', 'oauth-1'),
                   (%s, %s, 'Claude Code', 'oauth', 'oauth-2')
            """,
            (principal_id, space_id, principal_id, space_id),
        )

        # Provider row for token XOR test
        cur = await conn.execute(
            """
            INSERT INTO providers (principal_id, space_id, slug, kind)
            VALUES (%s, %s, 'mac', 'actuator') RETURNING id
            """,
            (principal_id, space_id),
        )
        provider_id = (await cur.fetchone())["id"]
        await conn.commit()

        # Token with neither client_id nor provider_id fails CHECK
        with pytest.raises(CheckViolation):
            await conn.execute(
                """
                INSERT INTO tokens (principal_id, kind, token_hash)
                VALUES (%s, 'bearer', %s)
                """,
                (principal_id, b"\x01" * 32),
            )
        await conn.rollback()

        # Token with both client_id and provider_id fails CHECK
        with pytest.raises(CheckViolation):
            await conn.execute(
                """
                INSERT INTO tokens (principal_id, client_id, provider_id, kind, token_hash)
                VALUES (%s, %s, %s, 'bearer', %s)
                """,
                (principal_id, client_id, provider_id, b"\x02" * 32),
            )
        await conn.rollback()

        # client_policies UNIQUE NULLS NOT DISTINCT prevents duplicate all-clients rules
        await conn.execute(
            """
            INSERT INTO client_policies (principal_id, client_id, tool_pattern, decision)
            VALUES (%s, NULL, 'gmail_send', 'deny')
            """,
            (principal_id,),
        )
        await conn.commit()

        with pytest.raises(UniqueViolation):
            await conn.execute(
                """
                INSERT INTO client_policies (principal_id, client_id, tool_pattern, decision)
                VALUES (%s, NULL, 'gmail_send', 'allow')
                """,
                (principal_id,),
            )
        await conn.rollback()


async def test_migration_0011_drops_legacy_memories_and_google_creds_keeping_tiers(tmp_path) -> None:
    """Upgrading from pre-0011 drops public.memories and issuer='google' creds while preserving tiers & other creds."""
    base_url = get_test_database_url()
    temp_db = f"zos_m12_clean_{uuid.uuid4().hex[:8]}"
    admin_url = _replace_db_name(base_url, "postgres")
    temp_url = _replace_db_name(base_url, temp_db)

    async with await psycopg.AsyncConnection.connect(admin_url, autocommit=True) as admin:
        await admin.execute(f"CREATE DATABASE {temp_db}")

    try:
        all_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        pre_files = [p for p in all_files if not p.name.startswith("0011_")]
        mig_0011 = next(p for p in all_files if p.name.startswith("0011_"))
        for mf in pre_files:
            (tmp_path / mf.name).write_text(mf.read_text(encoding="utf-8"), encoding="utf-8")

        async with open_pool(temp_url) as pool:
            # Apply 0001..0010 via run_migrations to simulate an existing live DB right before 0011
            await run_migrations(pool, tmp_path)

            async with pool.connection() as conn:
                # Seed principal, legacy public.memories row, google + non-google credentials, and memory_put tier
                p_id = (
                    await (
                        await conn.execute(
                            "INSERT INTO principals (slug, display_name) VALUES ('owner', 'Owner') RETURNING id"
                        )
                    ).fetchone()
                )["id"]
                s_id = (
                    await (
                        await conn.execute(
                            "INSERT INTO spaces (principal_id, slug, name) VALUES (%s, 'main', 'Main') RETURNING id",
                            (p_id,),
                        )
                    ).fetchone()
                )["id"]
                await conn.execute(
                    "INSERT INTO public.memories (principal_id, space_id, key, content) VALUES (%s, %s, 'k1', 'v1')",
                    (p_id, s_id),
                )
                await conn.execute(
                    """
                    INSERT INTO credentials (principal_id, issuer, account, ciphertext, nonce)
                    VALUES (%s, 'google', 'primary', %s, %s),
                           (%s, 'module:memory', 'db_dsn', %s, %s)
                    """,
                    (p_id, b"\x01" * 16, b"\x02" * 12, p_id, b"\x03" * 16, b"\x04" * 12),
                )
                await conn.execute(
                    """
                    INSERT INTO tool_tiers (principal_id, tool_pattern, tier)
                    VALUES (%s, 'memory_put', 'write')
                    ON CONFLICT DO NOTHING
                    """,
                    (p_id,),
                )
                await conn.commit()

            # Apply 0011 via run_migrations
            applied = await run_migrations(pool, MIGRATIONS_DIR)
            assert applied == [mig_0011.name]

            async with pool.connection() as conn:
                reg = (await (await conn.execute("SELECT to_regclass('public.memories') AS r")).fetchone())["r"]
                assert reg is None
                g_cnt = (
                    await (
                        await conn.execute("SELECT COUNT(*) AS cnt FROM credentials WHERE issuer = 'google'")
                    ).fetchone()
                )["cnt"]
                assert g_cnt == 0
                m_cnt = (
                    await (
                        await conn.execute("SELECT COUNT(*) AS cnt FROM credentials WHERE issuer = 'module:memory'")
                    ).fetchone()
                )["cnt"]
                assert m_cnt == 1
                tier = (
                    await (
                        await conn.execute("SELECT tier FROM tool_tiers WHERE tool_pattern = 'memory_put'")
                    ).fetchone()
                )
                assert tier is not None and tier["tier"] == "write"
    finally:
        async with await psycopg.AsyncConnection.connect(admin_url, autocommit=True) as admin:
            await admin.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
                """,
                (temp_db,),
            )
            await admin.execute(f"DROP DATABASE IF EXISTS {temp_db}")
