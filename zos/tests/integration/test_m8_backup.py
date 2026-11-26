"""M8.6 tests for backup recording, /admin Overview backup display, and LibreSSL encryption roundtrip."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import httpx
from argon2 import PasswordHasher
from typer.testing import CliRunner

from tests.conftest import MIGRATIONS_DIR, get_test_database_url
from zos.app import create_app
from zos.cli import app as cli_app
from zos.config import Settings
from zos.db import open_pool, run_migrations
from zos.db.queries import init_principal_and_space
from zos.vault import generate_secret_key


async def test_admin_overview_displays_last_backup_time_and_size(db_pool) -> None:
    await init_principal_and_space(db_pool, principal_slug="zubin")
    owner_pw = "backup-dash-pw"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8787",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost:8787",
            follow_redirects=False,
        ) as client,
    ):
        assert (await client.post("/admin/login", data={"passphrase": owner_pw})).status_code == 303
        r_before = await client.get("/admin")
        assert r_before.status_code == 200
        assert "Last backup" in r_before.text
        assert "Never" in r_before.text

        # Record a completed backup via AdminService
        from zos.admin_service import AdminService

        await AdminService(db_pool).record_backup(
            filename="zos-backup-20261124-031500Z.dump.enc",
            size_bytes=20480,
            completed_at="2026-11-24T03:15:00Z",
            retained_count=1,
        )

        r_after = await client.get("/admin")
        assert r_after.status_code == 200
        assert "20.0 KB" in r_after.text
        assert "2026-11-24T03:15:00Z" in r_after.text


def test_cli_backup_record_and_openssl_roundtrip(tmp_path: Path, monkeypatch) -> None:
    url = get_test_database_url()

    async def _reset() -> None:
        async with open_pool(url) as pool:
            await run_migrations(pool, MIGRATIONS_DIR)
            async with pool.connection() as conn:
                await conn.execute("TRUNCATE principals RESTART IDENTITY CASCADE")
                await conn.commit()

    asyncio.run(_reset())
    monkeypatch.setenv("DATABASE_URL", url)
    runner = CliRunner()
    assert runner.invoke(cli_app, ["init", "--principal", "zubin"]).exit_code == 0

    r = runner.invoke(
        cli_app,
        [
            "backup",
            "record",
            "--filename",
            "zos-backup-20261124-031500Z.dump.enc",
            "--size-bytes",
            "4096",
            "--completed-at",
            "2026-11-24T03:15:00Z",
            "--retained",
            "1",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "size_bytes=4096" in r.output

    # Verify macOS /usr/bin/openssl enc -aes-256-cbc -pbkdf2 -salt roundtrip
    plain = b"PGDMP-synthetic-backup-payload-12345"
    enc_path = tmp_path / "test.dump.enc"
    env = {**os.environ, "ZOS_BACKUP_KEY": "test-backup-key-secret"}
    p_enc = subprocess.run(
        [
            "/usr/bin/openssl",
            "enc",
            "-aes-256-cbc",
            "-pbkdf2",
            "-salt",
            "-pass",
            "env:ZOS_BACKUP_KEY",
            "-out",
            str(enc_path),
        ],
        input=plain,
        env=env,
        check=True,
    )
    assert p_enc.returncode == 0
    assert enc_path.read_bytes() != plain

    p_dec = subprocess.run(
        [
            "/usr/bin/openssl",
            "enc",
            "-d",
            "-aes-256-cbc",
            "-pbkdf2",
            "-pass",
            "env:ZOS_BACKUP_KEY",
            "-in",
            str(enc_path),
        ],
        env=env,
        capture_output=True,
        check=True,
    )
    assert p_dec.stdout == plain
