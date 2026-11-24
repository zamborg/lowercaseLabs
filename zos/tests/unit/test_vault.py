"""Unit and integration tests for M4.1 AES-GCM CredentialVault and per-credential refresh lock."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from cryptography.exceptions import InvalidTag

from zos.db.queries import init_principal_and_space
from zos.vault import CredentialStore, CredentialVault, generate_secret_key


def test_vault_round_trip_and_wrong_key_fails() -> None:
    key1 = generate_secret_key()
    key2 = generate_secret_key()
    v1 = CredentialVault(key1, key_version=1)
    v2 = CredentialVault(key2, key_version=1)
    pid = uuid.uuid4()

    blob = v1.encrypt(
        {"access_token": "at-secret", "refresh_token": "rt-secret"},
        principal_id=pid,
        issuer="google",
        account="zubin@example.com",
    )
    assert b"at-secret" not in blob.ciphertext
    assert b"rt-secret" not in blob.ciphertext

    plain = v1.decrypt(
        ciphertext=blob.ciphertext,
        nonce=blob.nonce,
        key_version=blob.key_version,
        principal_id=pid,
        issuer="google",
        account="zubin@example.com",
    )
    assert plain == {"access_token": "at-secret", "refresh_token": "rt-secret"}

    # Wrong key fails
    with pytest.raises(InvalidTag):
        v2.decrypt(
            ciphertext=blob.ciphertext,
            nonce=blob.nonce,
            key_version=blob.key_version,
            principal_id=pid,
            issuer="google",
            account="zubin@example.com",
        )

    # Wrong AAD (account mismatch) fails
    with pytest.raises(InvalidTag):
        v1.decrypt(
            ciphertext=blob.ciphertext,
            nonce=blob.nonce,
            key_version=blob.key_version,
            principal_id=pid,
            issuer="google",
            account="other@example.com",
        )


async def test_concurrent_refresh_calls_hit_upstream_once(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    vault = CredentialVault(generate_secret_key())
    store = CredentialStore(db_pool, vault)

    # Save an already-expired token (expires_in = -10)
    cred_id = await store.save_google_credential(
        principal_id=p_id,
        account="zubin@example.com",
        scopes=["https://www.googleapis.com/auth/calendar.events"],
        access_token="expired-token",
        refresh_token="rt-1",
        expires_in=-10,
    )

    refresh_calls = 0

    async def fake_refresh(rt: str) -> tuple[str, int, str | None]:
        nonlocal refresh_calls
        refresh_calls += 1
        await asyncio.sleep(0.1)
        return ("fresh-token-123", 3600, None)

    results = await asyncio.gather(
        *[store.get_valid_access_token(cred_id, space_id=s_id, refresh_fn=fake_refresh) for _ in range(8)]
    )
    assert results == ["fresh-token-123"] * 8
    assert refresh_calls == 1
