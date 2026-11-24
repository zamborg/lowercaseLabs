"""AEAD (AES-GCM) credential vault with key_version and per-credential async refresh locking."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.audit import record_event

NONCE_BYTES = 12
KEY_BYTES = 32
REFRESH_SKEW_SECONDS = 60


def decode_secret_key(raw_key: str) -> bytes:
    """Decode a base64 / urlsafe-base64 32-byte encryption key from ZOS_SECRET_KEY."""
    if not raw_key or not raw_key.strip():
        raise ValueError("ZOS_SECRET_KEY is not configured")
    padded = raw_key.strip()
    padding = (-len(padded)) % 4
    padded += "=" * padding
    try:
        key = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception as exc:
        raise ValueError("ZOS_SECRET_KEY must be valid base64") from exc
    if len(key) != KEY_BYTES:
        raise ValueError(f"ZOS_SECRET_KEY must decode to exactly {KEY_BYTES} bytes, got {len(key)}")
    return key


def generate_secret_key() -> str:
    """Generate a new urlsafe base64 32-byte key suitable for ZOS_SECRET_KEY."""
    return base64.urlsafe_b64encode(os.urandom(KEY_BYTES)).decode("ascii")


def _aad(principal_id: UUID, issuer: str, account: str, key_version: int) -> bytes:
    return f"{principal_id}|{issuer}|{account}|v{key_version}".encode()


@dataclass(frozen=True, slots=True)
class EncryptedBlob:
    ciphertext: bytes
    nonce: bytes
    key_version: int


class CredentialVault:
    """Encrypts and decrypts upstream OAuth credentials in memory using AES-256-GCM."""

    def __init__(self, key_b64: str, *, key_version: int = 1) -> None:
        self._key = decode_secret_key(key_b64)
        self._aesgcm = AESGCM(self._key)
        self.key_version = key_version

    def encrypt(
        self,
        payload: dict[str, Any],
        *,
        principal_id: UUID,
        issuer: str,
        account: str,
    ) -> EncryptedBlob:
        nonce = os.urandom(NONCE_BYTES)
        plaintext = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        aad = _aad(principal_id, issuer, account, self.key_version)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, aad)
        return EncryptedBlob(ciphertext=ciphertext, nonce=nonce, key_version=self.key_version)

    def decrypt(
        self,
        *,
        ciphertext: bytes,
        nonce: bytes,
        key_version: int,
        principal_id: UUID,
        issuer: str,
        account: str,
    ) -> dict[str, Any]:
        if key_version != self.key_version:
            raise ValueError(f"Unsupported credential key_version={key_version} (expected {self.key_version})")
        aad = _aad(principal_id, issuer, account, key_version)
        plaintext = self._aesgcm.decrypt(bytes(nonce), bytes(ciphertext), aad)
        return json.loads(plaintext.decode("utf-8"))


class CredentialRevokedError(Exception):
    """Raised when an upstream refresh token is rejected/revoked by the issuer."""


RefreshTokenFn = Callable[[str], Awaitable[tuple[str, int, str | None]]]
"""Given a refresh_token, returns (new_access_token, expires_in_seconds, optional_new_refresh_token)."""


class CredentialStore:
    """Postgres-backed encrypted credential store with per-credential asyncio.Lock around refresh."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        vault: CredentialVault,
    ) -> None:
        self.pool = pool
        self.vault = vault
        self._locks: dict[UUID, asyncio.Lock] = {}

    def _lock_for(self, credential_id: UUID) -> asyncio.Lock:
        lock = self._locks.get(credential_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[credential_id] = lock
        return lock

    async def save_google_credential(
        self,
        *,
        principal_id: UUID,
        account: str,
        scopes: list[str],
        access_token: str,
        refresh_token: str,
        expires_in: int,
    ) -> UUID:
        blob = self.vault.encrypt(
            {"access_token": access_token, "refresh_token": refresh_token},
            principal_id=principal_id,
            issuer="google",
            account=account,
        )
        expires_at = datetime.now(UTC) + timedelta(seconds=int(expires_in))
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO credentials (
                    principal_id, issuer, account, scopes,
                    ciphertext, nonce, key_version, access_expires_at, updated_at
                )
                VALUES (%s, 'google', %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (principal_id, issuer, account) DO UPDATE
                SET scopes = EXCLUDED.scopes,
                    ciphertext = EXCLUDED.ciphertext,
                    nonce = EXCLUDED.nonce,
                    key_version = EXCLUDED.key_version,
                    access_expires_at = EXCLUDED.access_expires_at,
                    updated_at = now()
                RETURNING id
                """,
                (
                    principal_id,
                    account,
                    scopes,
                    blob.ciphertext,
                    blob.nonce,
                    blob.key_version,
                    expires_at,
                ),
            )
            row = await cur.fetchone()
            await conn.commit()
        assert row is not None
        return row["id"]

    async def get_valid_access_token(
        self,
        credential_id: UUID,
        *,
        space_id: UUID,
        refresh_fn: RefreshTokenFn,
        force_refresh: bool = False,
    ) -> str:
        """Return a valid decrypted access token, refreshing under a per-credential lock if needed."""
        # Fast path without lock if token is still fresh
        if not force_refresh:
            token = await self._read_if_fresh(credential_id)
            if token is not None:
                return token

        async with self._lock_for(credential_id):
            # Double-check inside lock so concurrent callers reuse the single refresh!
            if not force_refresh:
                token = await self._read_if_fresh(credential_id)
                if token is not None:
                    return token

            row = await self._fetch_row(credential_id)
            if row is None:
                raise CredentialRevokedError(f"Credential {credential_id} not found")

            decrypted = self.vault.decrypt(
                ciphertext=row["ciphertext"],
                nonce=row["nonce"],
                key_version=row["key_version"],
                principal_id=row["principal_id"],
                issuer=row["issuer"],
                account=row["account"],
            )
            refresh_token = decrypted.get("refresh_token")
            if not refresh_token:
                raise CredentialRevokedError("No refresh_token stored for credential")

            new_access, expires_in, maybe_new_refresh = await refresh_fn(refresh_token)
            updated_refresh = maybe_new_refresh or refresh_token
            new_blob = self.vault.encrypt(
                {"access_token": new_access, "refresh_token": updated_refresh},
                principal_id=row["principal_id"],
                issuer=row["issuer"],
                account=row["account"],
            )
            new_expires_at = datetime.now(UTC) + timedelta(seconds=int(expires_in))

            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    UPDATE credentials
                    SET ciphertext = %s,
                        nonce = %s,
                        key_version = %s,
                        access_expires_at = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (
                        new_blob.ciphertext,
                        new_blob.nonce,
                        new_blob.key_version,
                        new_expires_at,
                        credential_id,
                    ),
                )
                await conn.commit()

            await record_event(
                self.pool,
                principal_id=row["principal_id"],
                space_id=space_id,
                event_type="credential.refreshed",
                payload={"credential_id": str(credential_id), "issuer": row["issuer"]},
            )
            return new_access

    async def delete_credential(self, credential_id: UUID) -> None:
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM credentials WHERE id = %s", (credential_id,))
            await conn.commit()

    async def _fetch_row(self, credential_id: UUID) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, principal_id, issuer, account, scopes,
                       ciphertext, nonce, key_version, access_expires_at
                FROM credentials
                WHERE id = %s
                """,
                (credential_id,),
            )
            row = await cur.fetchone()
        return dict(row) if row else None

    async def _read_if_fresh(self, credential_id: UUID) -> str | None:
        row = await self._fetch_row(credential_id)
        if row is None:
            raise CredentialRevokedError(f"Credential {credential_id} not found")
        exp = row["access_expires_at"]
        if exp is None or exp <= datetime.now(UTC) + timedelta(seconds=REFRESH_SKEW_SECONDS):
            return None
        decrypted = self.vault.decrypt(
            ciphertext=row["ciphertext"],
            nonce=row["nonce"],
            key_version=row["key_version"],
            principal_id=row["principal_id"],
            issuer=row["issuer"],
            account=row["account"],
        )
        return decrypted.get("access_token")
