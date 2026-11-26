"""Isolated Postgres role/schema provisioning and vault secret storage for ZOS modules."""

from __future__ import annotations

import secrets
from urllib.parse import quote, urlparse, urlunparse
from uuid import UUID

import psycopg.errors
from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.router.naming import validate_upstream_slug
from zos.vault import CredentialStore

ISSUER_MODULE_DB = "module_db"
ISSUER_MODULE_SECRET = "module_secret"
ISSUER_MODULE_CONFIG = "module_config"


def module_role_and_schema(slug: str) -> tuple[str, str]:
    """Return validated `(role_name, schema_name)` for a module slug."""
    clean = validate_upstream_slug(slug)
    ident = f"mod_{clean}"
    return ident, ident


def build_module_database_url(base_database_url: str, *, role: str, password: str) -> str:
    """Replace credentials in `base_database_url` with the module's dedicated Postgres role."""
    parsed = urlparse(base_database_url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("Invalid base DATABASE_URL")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port_part = f":{parsed.port}" if parsed.port else ""
    userinfo = f"{quote(role, safe='')}:{quote(password, safe='')}"
    netloc = f"{userinfo}@{host}{port_part}"
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


class ModuleStorageProvisioner:
    """Provisions isolated Postgres roles/schemas and vault-stored module secrets."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        store: CredentialStore | None,
    ) -> None:
        self.pool = pool
        self.store = store

    def _require_store(self) -> CredentialStore:
        if self.store is None:
            raise ValueError("ZOS_SECRET_KEY is required to provision module credentials")
        return self.store

    async def ensure_module_secret(self, *, principal_id: UUID, slug: str) -> str:
        """Return existing per-module HMAC secret or mint and store a new one in the vault."""
        store = self._require_store()
        clean = validate_upstream_slug(slug)
        found = await store.find_credential(
            principal_id=principal_id,
            issuer=ISSUER_MODULE_SECRET,
            account=clean,
        )
        if found is not None:
            _row, payload = found
            existing = str(payload.get("secret") or "")
            if existing:
                return existing
        secret = secrets.token_urlsafe(32)
        await store.save_credential(
            principal_id=principal_id,
            issuer=ISSUER_MODULE_SECRET,
            account=clean,
            payload={"secret": secret},
        )
        return secret

    async def get_module_secret(self, *, principal_id: UUID, slug: str) -> str | None:
        """Load per-module HMAC secret from the vault, or None if not found."""
        if self.store is None:
            return None
        clean = slug.strip().lower()
        found = await self.store.find_credential(
            principal_id=principal_id,
            issuer=ISSUER_MODULE_SECRET,
            account=clean,
        )
        if found is None:
            return None
        _row, payload = found
        val = str(payload.get("secret") or "")
        return val or None

    async def save_module_config(
        self,
        *,
        principal_id: UUID,
        slug: str,
        values: dict[str, str],
    ) -> None:
        """Persist module configuration values encrypted in the vault."""
        store = self._require_store()
        clean = validate_upstream_slug(slug)
        await store.save_credential(
            principal_id=principal_id,
            issuer=ISSUER_MODULE_CONFIG,
            account=clean,
            payload={"values": dict(values)},
        )

    async def load_module_config(self, *, principal_id: UUID, slug: str) -> dict[str, str]:
        """Load encrypted module configuration values from the vault."""
        if self.store is None:
            return {}
        clean = slug.strip().lower()
        found = await self.store.find_credential(
            principal_id=principal_id,
            issuer=ISSUER_MODULE_CONFIG,
            account=clean,
        )
        if found is None:
            return {}
        _row, payload = found
        raw = payload.get("values")
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}

    async def provision_postgres(
        self,
        *,
        principal_id: UUID,
        slug: str,
        base_database_url: str,
    ) -> str:
        """Create isolated Postgres role `mod_<slug>` and schema `mod_<slug>`, returning its DSN.

        Revokes `public` schema and core table access from `PUBLIC` and `mod_<slug>` so the
        module role can read/write only inside its own schema.
        """
        store = self._require_store()
        role_name, schema_name = module_role_and_schema(slug)

        # Reuse stored password if re-installing/upgrading with preserved schema, else generate
        password = ""
        found = await store.find_credential(
            principal_id=principal_id,
            issuer=ISSUER_MODULE_DB,
            account=validate_upstream_slug(slug),
        )
        if found is not None:
            _row, payload = found
            password = str(payload.get("password") or "")
        if not password:
            password = secrets.token_urlsafe(32)
            await store.save_credential(
                principal_id=principal_id,
                issuer=ISSUER_MODULE_DB,
                account=validate_upstream_slug(slug),
                payload={"role": role_name, "schema": schema_name, "password": password},
            )

        role_id = sql.Identifier(role_name)
        schema_id = sql.Identifier(schema_name)

        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT current_database() AS db")
            db_row = await cur.fetchone()
            assert db_row is not None
            db_id = sql.Identifier(str(db_row["db"]))

            cur = await conn.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s",
                (role_name,),
            )
            exists = await cur.fetchone() is not None
            if exists:
                # PG 16+: ALTER ROLE rejects changing any attribute (even to NO*) that the
                # calling role does not itself hold; only update LOGIN, PASSWORD, NOCREATEROLE, NOINHERIT.
                await conn.execute(
                    sql.SQL(
                        "ALTER ROLE {role} WITH LOGIN PASSWORD {pw} NOCREATEROLE NOINHERIT"
                    ).format(role=role_id, pw=sql.Literal(password))
                )
            else:
                # NOSUPERUSER, NOCREATEDB, NOCREATEROLE, NOREPLICATION are Postgres defaults.
                await conn.execute(
                    sql.SQL(
                        "CREATE ROLE {role} WITH LOGIN PASSWORD {pw} NOINHERIT"
                    ).format(role=role_id, pw=sql.Literal(password))
                )

            # In Postgres 16+, a non-superuser CREATEROLE core role must be a member of
            # `mod_<slug>` to run `CREATE/ALTER SCHEMA ... AUTHORIZATION mod_<slug>` and to
            # include module tables in `pg_dump` backups. `mod_<slug>` itself has NOINHERIT
            # and receives zero membership in the core role.
            try:
                async with conn.transaction():
                    await conn.execute(
                        sql.SQL("GRANT {role} TO CURRENT_USER").format(role=role_id)
                    )
            except psycopg.errors.InsufficientPrivilege:
                pass

            await conn.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema} AUTHORIZATION {role}").format(
                    schema=schema_id,
                    role=role_id,
                )
            )
            await conn.execute(
                sql.SQL("ALTER SCHEMA {schema} OWNER TO {role}").format(
                    schema=schema_id,
                    role=role_id,
                )
            )
            # Reassign ownership of any pre-existing tables/sequences in `mod_<slug>`
            # (e.g. after restoring a `--no-owner` pg_dump backup onto a new host).
            t_cur = await conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = %s",
                (schema_name,),
            )
            for trow in await t_cur.fetchall():
                await conn.execute(
                    sql.SQL("ALTER TABLE {schema}.{tbl} OWNER TO {role}").format(
                        schema=schema_id,
                        tbl=sql.Identifier(str(trow["tablename"])),
                        role=role_id,
                    )
                )
            s_cur = await conn.execute(
                "SELECT sequencename FROM pg_sequences WHERE schemaname = %s",
                (schema_name,),
            )
            for srow in await s_cur.fetchall():
                await conn.execute(
                    sql.SQL("ALTER SEQUENCE {schema}.{seq} OWNER TO {role}").format(
                        schema=schema_id,
                        seq=sql.Identifier(str(srow["sequencename"])),
                        role=role_id,
                    )
                )
            await conn.execute(
                sql.SQL("ALTER ROLE {role} SET search_path TO {schema}").format(
                    role=role_id,
                    schema=schema_id,
                )
            )
            await conn.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {db} TO {role}").format(
                    db=db_id,
                    role=role_id,
                )
            )
            await conn.execute(
                sql.SQL("GRANT USAGE, CREATE ON SCHEMA {schema} TO {role}").format(
                    schema=schema_id,
                    role=role_id,
                )
            )

            # Explicitly lock down public schema and core tables from PUBLIC and the module role
            await conn.execute(
                sql.SQL("REVOKE ALL ON SCHEMA public FROM PUBLIC, {role}").format(role=role_id)
            )
            await conn.execute(
                sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC, {role}").format(
                    role=role_id
                )
            )
            await conn.execute(
                sql.SQL("REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC, {role}").format(
                    role=role_id
                )
            )
            await conn.commit()

        return build_module_database_url(base_database_url, role=role_name, password=password)

    async def purge_storage(self, *, principal_id: UUID, slug: str) -> None:
        """Drop `mod_<slug>` schema and role, and delete module vault credentials."""
        clean = validate_upstream_slug(slug)
        role_name, schema_name = module_role_and_schema(clean)
        role_id = sql.Identifier(role_name)
        schema_id = sql.Identifier(schema_name)

        async with self.pool.connection() as conn:
            # Best-effort termination of lingering module DB sessions (requires superuser or
            # pg_signal_backend; skip cleanly when running under a restricted non-superuser role).
            try:
                async with conn.transaction():
                    await conn.execute(
                        """
                        SELECT pg_terminate_backend(pid)
                        FROM pg_stat_activity
                        WHERE usename = %s AND pid <> pg_backend_pid()
                        """,
                        (role_name,),
                    )
            except psycopg.errors.InsufficientPrivilege:
                pass

            await conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {schema} CASCADE").format(schema=schema_id)
            )
            cur = await conn.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s",
                (role_name,),
            )
            if await cur.fetchone() is not None:
                await conn.execute(
                    sql.SQL("DROP OWNED BY {role} CASCADE").format(role=role_id)
                )
                await conn.execute(
                    sql.SQL("DROP ROLE IF EXISTS {role}").format(role=role_id)
                )
            await conn.execute(
                """
                DELETE FROM credentials
                WHERE principal_id = %s
                  AND account = %s
                  AND issuer = ANY(%s)
                """,
                (principal_id, clean, [ISSUER_MODULE_DB, ISSUER_MODULE_SECRET, ISSUER_MODULE_CONFIG]),
            )
            await conn.commit()
