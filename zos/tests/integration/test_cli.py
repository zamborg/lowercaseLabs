"""Integration tests for M1.2 CLI commands via Typer CliRunner."""

from __future__ import annotations

import asyncio
import hashlib

import psycopg
from psycopg.rows import dict_row
from typer.testing import CliRunner

from tests.conftest import MIGRATIONS_DIR, get_test_database_url
from zos.cli import app
from zos.db import open_pool, run_migrations
from zos.db.queries import V0_TIER_SEEDS

runner = CliRunner()


def _parse_kv(output: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in output.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _reset_db(url: str) -> None:
    async def _prep() -> None:
        async with open_pool(url) as pool:
            await run_migrations(pool, MIGRATIONS_DIR)
            async with pool.connection() as conn:
                await conn.execute("TRUNCATE principals RESTART IDENTITY CASCADE")
                await conn.commit()

    asyncio.run(_prep())


def test_cli_init_client_token_policy_and_actuator(monkeypatch) -> None:
    url = get_test_database_url()
    _reset_db(url)
    monkeypatch.setenv("DATABASE_URL", url)

    # 1. zos migrate
    res = runner.invoke(app, ["migrate"])
    assert res.exit_code == 0, res.output

    # 2. zos init (idempotent)
    res = runner.invoke(app, ["init", "--principal", "zubin", "--display-name", "Zubin"])
    assert res.exit_code == 0, res.output
    assert "principal=zubin" in res.output
    assert f"tiers={len(V0_TIER_SEEDS)}" in res.output

    res2 = runner.invoke(app, ["init", "--principal", "zubin", "--display-name", "Zubin"])
    assert res2.exit_code == 0, res2.output

    # 3. zos client add NAME --bearer
    res = runner.invoke(app, ["client", "add", "codex", "--bearer"])
    assert res.exit_code == 0, res.output
    kv = _parse_kv(res.output)
    assert kv["token"].startswith("zos_b_")
    token_id = kv["token_id"]
    raw_token = kv["token"]

    # Verify SHA-256 hash stored in DB (not plaintext)
    with psycopg.connect(url, row_factory=dict_row) as conn:
        row = conn.execute(
            "SELECT token_hash, revoked_at FROM tokens WHERE id = %s", (token_id,)
        ).fetchone()
        assert row is not None
        assert bytes(row["token_hash"]) == hashlib.sha256(raw_token.encode()).digest()
        assert row["revoked_at"] is None

    # 4. zos policy set (global + client-specific)
    res = runner.invoke(app, ["policy", "set", "gmail_send", "deny"])
    assert res.exit_code == 0, res.output
    assert "client=*" in res.output

    res = runner.invoke(app, ["policy", "set", "--client", "codex", "gmail_send", "allow"])
    assert res.exit_code == 0, res.output
    assert "client=codex" in res.output

    # Updating global rule works cleanly despite NULL client_id
    res = runner.invoke(app, ["policy", "set", "gmail_send", "require_approval"])
    assert res.exit_code == 0, res.output

    with psycopg.connect(url, row_factory=dict_row) as conn:
        rows = conn.execute("SELECT client_id, tool_pattern, decision FROM client_policies").fetchall()
        assert len(rows) == 2

    # 5. zos token revoke
    res = runner.invoke(app, ["token", "revoke", token_id])
    assert res.exit_code == 0, res.output
    assert "Revoked 1 token(s)." in res.output

    with psycopg.connect(url, row_factory=dict_row) as conn:
        row = conn.execute("SELECT revoked_at FROM tokens WHERE id = %s", (token_id,)).fetchone()
        assert row is not None
        assert row["revoked_at"] is not None

    # 6. zos actuator add mac
    res = runner.invoke(app, ["actuator", "add", "mac"])
    assert res.exit_code == 0, res.output
    akv = _parse_kv(res.output)
    assert akv["token"].startswith("zos_a_")

    with psycopg.connect(url, row_factory=dict_row) as conn:
        arow = conn.execute(
            "SELECT kind, provider_id, client_id FROM tokens WHERE id = %s", (akv["token_id"],)
        ).fetchone()
        assert arow is not None
        assert arow["kind"] == "actuator"
        assert str(arow["provider_id"]) == akv["provider_id"]
        assert arow["client_id"] is None
