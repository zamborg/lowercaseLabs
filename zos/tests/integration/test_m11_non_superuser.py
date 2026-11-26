"""Integration tests proving ZOS core works under a restricted non-superuser Postgres role."""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

import psycopg
import pytest

from tests.conftest import get_test_database_url
from zos.app import create_app
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

        # 3. Provision a module Postgres role/schema with pg_trgm extension as non-superuser (and re-provision on upgrade!)
        store = CredentialStore(pool, CredentialVault(cfg.zos_secret_key))
        prov = ModuleStorageProvisioner(pool, store)
        mod_dsn = await prov.provision_postgres(
            principal_id=principal_id,
            slug=mod_slug,
            base_database_url=app_url,
            extensions=["pg_trgm"],
        )
        # Second call exercises the `exists=True` ALTER ROLE path as non-superuser!
        mod_dsn_re = await prov.provision_postgres(
            principal_id=principal_id,
            slug=mod_slug,
            base_database_url=app_url,
            extensions=["pg_trgm"],
        )
        assert mod_dsn_re == mod_dsn

        # Verify pg_trgm was relocated into the shared `extensions` schema
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT n.nspname AS ext_schema
                FROM pg_extension e
                JOIN pg_namespace n ON n.oid = e.extnamespace
                WHERE e.extname = 'pg_trgm'
                """
            )
            erow = await cur.fetchone()
            assert erow is not None and erow["ext_schema"] == "extensions"

        # 4. Connect AS the module role: can create/read its own schema and use pg_trgm, CANNOT read core tables
        async with await psycopg.AsyncConnection.connect(mod_dsn, autocommit=True) as mod_conn:
            await mod_conn.execute("CREATE TABLE notes (id INT PRIMARY KEY, body TEXT)")
            await mod_conn.execute(
                "CREATE INDEX notes_trgm_idx ON notes USING GIN (body gin_trgm_ops)"
            )
            await mod_conn.execute("INSERT INTO notes (id, body) VALUES (1, 'isolated memory note')")
            cur = await mod_conn.execute(
                "SELECT body, similarity(body, 'isolated') FROM notes WHERE id = 1"
            )
            row = await cur.fetchone()
            assert row is not None and row[0] == "isolated memory note" and float(row[1]) > 0.2

            for core_table in ("credentials", "events", "clients", "tokens"):
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    await mod_conn.execute(f"SELECT COUNT(*) FROM public.{core_table}")

        # 5. Core role (non-superuser) CAN read module schema for pg_dump backups (via membership)
        async with pool.connection() as conn:
            cur = await conn.execute(f"SELECT body FROM mod_{mod_slug}.notes WHERE id = 1")
            row = await cur.fetchone()
            assert row is not None and row["body"] == "isolated memory note"

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


async def test_startup_warns_when_connected_as_superuser_and_migrate_script_is_idempotent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Core logs a startup WARNING for Postgres superuser, none for non-superuser; migrate script is idempotent."""
    admin_url = get_test_database_url()
    suffix = uuid.uuid4().hex[:8]
    app_role = f"zos_warn_{suffix}"
    app_pw = f"pw_{uuid.uuid4().hex}"
    test_db = f"zos_warn_{suffix}"

    # 1. Superuser connection emits WARNING
    cfg_super = Settings(database_url=admin_url, zos_secret_key=generate_secret_key())
    app_super = create_app(cfg_super, run_migrations_on_startup=False)
    with caplog.at_level(logging.WARNING, logger="zos.app"):
        async with app_super.router.lifespan_context(app_super):
            pass
    assert any("connected to Postgres as superuser" in r.message for r in caplog.records)

    # 2. Create a superuser-owned DB with migrations applied, then migrate to non-superuser role
    async with await psycopg.AsyncConnection.connect(admin_url, autocommit=True) as admin:
        await admin.execute(f"CREATE DATABASE {test_db}")
    admin_db_url = _replace_db_and_user(
        admin_url,
        user=urlparse(admin_url).username or "zos",
        password=urlparse(admin_url).password or "zos",
        dbname=test_db,
    )
    super_pool = create_pool(admin_db_url, min_size=1, max_size=2)
    await super_pool.open()
    try:
        await run_migrations(super_pool, Settings(database_url=admin_db_url).migrations_dir)
    finally:
        await super_pool.close()

    script = Path(__file__).resolve().parents[2] / "deploy" / "postgres" / "migrate-existing-to-app-role.sh"
    has_pg_container = (
        subprocess.run(["docker", "inspect", "zos-pg-test"], capture_output=True).returncode == 0
    )
    try:
        if has_pg_container:
            # Run script TWICE to verify idempotency on existing volumes
            for _ in range(2):
                res = subprocess.run(
                    [
                        str(script),
                        "--container", "zos-pg-test",
                        "--db", test_db,
                        "--super-user", urlparse(admin_url).username or "zos",
                        "--super-password", urlparse(admin_url).password or "zos",
                        "--app-user", app_role,
                        "--app-password", app_pw,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                assert f"role={app_role}" in res.stdout
        else:
            # CI service-container fallback: create equivalent restricted role directly
            async with await psycopg.AsyncConnection.connect(admin_db_url, autocommit=True) as admin2:
                await admin2.execute(
                    f"CREATE ROLE {app_role} WITH LOGIN PASSWORD '{app_pw}' "
                    "CREATEROLE NOSUPERUSER NOCREATEDB NOREPLICATION"
                )
                await admin2.execute(f"ALTER DATABASE {test_db} OWNER TO {app_role}")
                await admin2.execute(f"ALTER SCHEMA public OWNER TO {app_role}")
                # Mirror migrate-existing-to-app-role.sh: hand over existing objects too.
                await admin2.execute(
                    f"""
                    DO $$
                    DECLARE r record;
                    BEGIN
                      FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP
                        EXECUTE format('ALTER TABLE public.%I OWNER TO {app_role}', r.tablename);
                      END LOOP;
                      FOR r IN SELECT sequencename FROM pg_sequences WHERE schemaname = 'public' LOOP
                        EXECUTE format('ALTER SEQUENCE public.%I OWNER TO {app_role}', r.sequencename);
                      END LOOP;
                      FOR r IN SELECT p.proname, pg_get_function_identity_arguments(p.oid) AS args
                               FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                               LEFT JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
                               WHERE n.nspname = 'public' AND d.objid IS NULL LOOP
                        EXECUTE format('ALTER FUNCTION public.%I(%s) OWNER TO {app_role}', r.proname, r.args);
                      END LOOP;
                    END $$
                    """
                )

        # 3. Non-superuser connection emits NO superuser warning
        app_url = _replace_db_and_user(admin_url, user=app_role, password=app_pw, dbname=test_db)
        caplog.clear()
        app_ns = create_app(Settings(database_url=app_url, zos_secret_key=generate_secret_key()))
        with caplog.at_level(logging.WARNING, logger="zos.app"):
            async with app_ns.router.lifespan_context(app_ns):
                pass
        assert not any("connected to Postgres as superuser" in r.message for r in caplog.records)
    finally:
        async with await psycopg.AsyncConnection.connect(admin_url, autocommit=True) as admin:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                (test_db,),
            )
            await admin.execute(f"DROP DATABASE IF EXISTS {test_db}")
            await admin.execute(f"DROP ROLE IF EXISTS {app_role}")
