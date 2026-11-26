"""Encrypted Postgres storage for Google OAuth tokens, states, and per-module settings."""

from __future__ import annotations

import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

CREDENTIAL_ROW_ID = "primary"
STATE_TTL_SECONDS = 600

MIGRATIONS: list[tuple[str, str]] = [
    (
        "0001_init",
        """
        CREATE TABLE IF NOT EXISTS google_credentials (
            id TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            scopes TEXT[] NOT NULL DEFAULT '{}',
            ciphertext BYTEA NOT NULL,
            nonce BYTEA NOT NULL,
            access_expires_at TIMESTAMPTZ NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS google_oauth_states (
            state TEXT PRIMARY KEY,
            expires_at TIMESTAMPTZ NOT NULL
        );

        CREATE TABLE IF NOT EXISTS google_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ),
]


def derive_vault_key(module_secret: str, *, service: str) -> bytes:
    """Derive a 32-byte AES-256-GCM key from `ZOS_MODULE_SECRET` via HKDF-SHA256."""
    if not module_secret:
        raise ValueError("ZOS_MODULE_SECRET must be non-empty")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"zos-google-module-vault-v1",
        info=f"zos-google:{service}:tokens".encode(),
    )
    return hkdf.derive(module_secret.encode("utf-8"))


def create_google_pool(database_url: str) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"row_factory": dict_row},
    )


class GoogleTokenStore:
    """Stores encrypted Google OAuth tokens, single-use OAuth states, and settings in module schema."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        *,
        module_secret: str,
        service: str,
    ) -> None:
        self.pool = pool
        self.service = service
        self._aesgcm = AESGCM(derive_vault_key(module_secret, service=service))

    def _aad(self, account: str) -> bytes:
        return f"{self.service}:{account}".encode()

    def _encrypt(self, payload: dict[str, Any], *, account: str) -> tuple[bytes, bytes]:
        nonce = os.urandom(12)
        plaintext = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ct = self._aesgcm.encrypt(nonce, plaintext, self._aad(account))
        return ct, nonce

    def _decrypt(self, ciphertext: bytes, nonce: bytes, *, account: str) -> dict[str, Any]:
        plaintext = self._aesgcm.decrypt(nonce, ciphertext, self._aad(account))
        data = json.loads(plaintext.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Decrypted token payload is not a JSON object")
        return data

    async def init_schema(self) -> list[str]:
        """Run idempotent, versioned schema migrations inside the module's own schema."""
        applied_now: list[str] = []
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS google_schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.commit()

            cur = await conn.execute("SELECT version FROM google_schema_migrations")
            already = {str(r["version"]) for r in await cur.fetchall()}

            for version, sql_body in MIGRATIONS:
                if version in already:
                    continue
                async with conn.transaction():
                    for stmt in sql_body.split(";"):
                        s = stmt.strip()
                        if s:
                            await conn.execute(s)
                    await conn.execute(
                        "INSERT INTO google_schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
                        (version,),
                    )
                applied_now.append(version)
        return applied_now

    async def create_oauth_state(self, *, ttl_seconds: int = STATE_TTL_SECONDS) -> str:
        state = secrets.token_urlsafe(24)
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute("DELETE FROM google_oauth_states WHERE expires_at < now()")
            await conn.execute(
                "INSERT INTO google_oauth_states (state, expires_at) VALUES (%s, %s)",
                (state, expires_at),
            )
        return state

    async def consume_oauth_state(self, state: str) -> bool:
        clean = state.strip()
        if not clean:
            return False
        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                DELETE FROM google_oauth_states
                WHERE state = %s AND expires_at >= now()
                RETURNING state
                """,
                (clean,),
            )
            row = await cur.fetchone()
        return row is not None

    async def save_tokens(
        self,
        *,
        account: str,
        scopes: list[str],
        access_token: str,
        refresh_token: str,
        expires_in: int,
    ) -> None:
        expires_at = datetime.now(UTC) + timedelta(seconds=max(0, int(expires_in)))
        payload = {
            "access_token": access_token,
            "refresh_token": refresh_token,
        }
        ciphertext, nonce = self._encrypt(payload, account=account)
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO google_credentials (id, account, scopes, ciphertext, nonce, access_expires_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    account = EXCLUDED.account,
                    scopes = EXCLUDED.scopes,
                    ciphertext = EXCLUDED.ciphertext,
                    nonce = EXCLUDED.nonce,
                    access_expires_at = EXCLUDED.access_expires_at,
                    updated_at = now()
                """,
                (CREDENTIAL_ROW_ID, account, list(scopes), ciphertext, nonce, expires_at),
            )

    async def load_tokens(self) -> dict[str, Any] | None:
        """Load decrypted token record (internal use only; never expose to UI or logs)."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT account, scopes, ciphertext, nonce, access_expires_at, updated_at
                FROM google_credentials
                WHERE id = %s
                """,
                (CREDENTIAL_ROW_ID,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        payload = self._decrypt(bytes(row["ciphertext"]), bytes(row["nonce"]), account=str(row["account"]))
        return {
            "account": str(row["account"]),
            "scopes": list(row["scopes"] or []),
            "access_token": str(payload.get("access_token") or ""),
            "refresh_token": str(payload.get("refresh_token") or ""),
            "access_expires_at": row["access_expires_at"],
            "updated_at": row["updated_at"],
        }

    async def get_connection_status(self) -> dict[str, Any]:
        """Return non-secret connection status metadata safe for the owner UI."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT account, scopes, access_expires_at, updated_at
                FROM google_credentials
                WHERE id = %s
                """,
                (CREDENTIAL_ROW_ID,),
            )
            row = await cur.fetchone()
        if row is None:
            return {
                "connected": False,
                "account": None,
                "scopes": [],
                "updated_at": None,
            }
        return {
            "connected": True,
            "account": str(row["account"]),
            "scopes": list(row["scopes"] or []),
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def delete_tokens(self) -> None:
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute("DELETE FROM google_credentials WHERE id = %s", (CREDENTIAL_ROW_ID,))

    async def get_setting(self, key: str) -> str | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT value FROM google_settings WHERE key = %s", (key,))
            row = await cur.fetchone()
        return str(row["value"]) if row is not None else None

    async def set_setting(self, key: str, value: str) -> None:
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO google_settings (key, value, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                (key, value),
            )
