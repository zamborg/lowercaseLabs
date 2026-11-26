"""Integration tests proving ZOS core works under a restricted non-superuser Postgres role."""

from __future__ import annotations

import uuid
from urllib.parse import quote, urlparse, urlunparse

import psycopg
import pytest

from tests.conftest import get_test_database_url
from zos.config import Settings
from zos.db import create_pool, run_migrations
from zos.db.queries import init_principal_and_space
from zos.modules import ModuleStorageProvisioner
from zos.vault import CredentialStore, CredentialVault, generate_secret_key


def _replace_db_and_user(base_url: str, *, user: str, password: str, dbname: str) -> str:
    p = urlparse(base_url)
    host = p.hostname or "127.0.0.1"
    port = f":{p.port}" if p.port else ""
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{host}{port}"
    return urlunparse((p.scheme, netloc, f"/{dbname}", "", "", ""))


async def test_non_superuser_core_role_migrations_and_module_isolation() -> None:
    """Run all ZOS migrations + module role/schema provisioning + isolation + purge as non-superuser."""
    admin_url = get_test_database_url()
    suffix = uuid.uuid4().hex[:8]
    app_role = f"zos_app_{suffix}"
    app_pw = f"pw_{uuid.uuid4().hex}"
    test_db = f"zos_ns_{suffix}"
    mod_slug = f"nsmod{suffix[:4]}"

    # Bootstrap isolated DB owned by non-superuser role with ONLY LOGIN + CREATEROLE
    async with await psycopg.AsyncConnection.connect(admin_url, autocommit=True) as admin:
        await admin.execute(
            f"CREATE ROLE {app_role} WITH LOGIN PASSWORD '{app_pw}' "
            "CREATEROLE NOSUPERUSER NOCREATEDB NOREPLICATION"
        )
        await admin.execute(f"CREATE DATABASE {test_db} OWNER {app_role}")

    # Transfer public schema ownership to app_role (matches deploy/postgres/01-init-app-role.sh)
    admin_new_db_url = _replace_db_and_user(
        admin_url,
        user=urlparse(admin_url).username or "zos",
        password=urlparse(admin_url).password or "zos",
        dbname=test_db,
    )
    async with await psycopg.AsyncConnection.connect(admin_new_db_url, autocommit=True) as admin2:
        await admin2.execute(f"ALTER SCHEMA public OWNER TO {app_role}")
        await admin2.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")

    app_url = _replace_db_and_user(admin_url, user=app_role, password=app_pw, dbname=test_db)
    pool = create_pool(app_url, min_size=1, max_size=4)
    await pool.open()
    try:
        # Verify the role is genuinely NOT a superuser
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT rolsuper, rolcreaterole, rolcreatedb FROM pg_roles WHERE rolname = current_user"
            )
            r = await cur.fetchone()
            assert r is not None
            assert r["rolsuper"] is False
            assert r["rolcreaterole"] is True
            assert r["rolcreatedb"] is False

        # 1. Run ALL migrations (0001..0007, including CREATE EXTENSION IF NOT EXISTS pg_trgm)
        cfg = Settings(database_url=app_url, zos_secret_key=generate_secret_key())
        applied = await run_migrations(pool, cfg.migrations_dir)
        assert len(applied) >= 7

        # 2. Bootstrap principal, space, memory provider, and default tiers
        init = await init_principal_and_space(pool, principal_slug="owner")
        principal_id = init["principal"]["id"]

        # 3. Provision a module Postgres role/schema as non-superuser
        store = CredentialStore(pool, CredentialVault(cfg.zos_secret_key))
        prov = ModuleStorageProvisioner(pool, store)
        mod_dsn = await prov.provision_postgres(
            principal_id=principal_id,
            slug=mod_slug,
            base_database_url=app_url,
        )

        # 4. Connect AS the module role: can create/read its own schema, CANNOT read core tables
        async with await psycopg.AsyncConnection.connect(mod_dsn, autocommit=True) as mod_conn:
            await mod_conn.execute("CREATE TABLE notes (id INT PRIMARY KEY, body TEXT)")
            await mod_conn.execute("INSERT INTO notes (id, body) VALUES (1, 'isolated')")
            cur = await mod_conn.execute("SELECT body FROM notes WHERE id = 1")
            assert await cur.fetchone() == ("isolated",)

            for core_table in ("credentials", "events", "memories", "tokens"):
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    await mod_conn.execute(f"SELECT COUNT(*) FROM public.{core_table}")

        # 5. Core role (non-superuser) CAN read module schema for pg_dump backups (via membership)
        async with pool.connection() as conn:
            cur = await conn.execute(f"SELECT body FROM mod_{mod_slug}.notes WHERE id = 1")
            row = await cur.fetchone()
            assert row is not None and row["body"] == "isolated"

        # 6. Purge module storage cleanly as non-superuser (no pg_terminate_backend crash)
        await prov.purge_storage(principal_id=principal_id, slug=mod_slug)
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (f"mod_{mod_slug}",),
            )
            assert await cur.fetchone() is None
            cur = await conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (f"mod_{mod_slug}",))
            assert await cur.fetchone() is None
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(admin_url, autocommit=True) as admin:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                (test_db,),
            )
            await admin.execute(f"DROP DATABASE IF EXISTS {test_db}")
            await admin.execute(f"DROP ROLE IF EXISTS mod_{mod_slug}")
            await admin.execute(f"DROP ROLE IF EXISTS {app_role}")
