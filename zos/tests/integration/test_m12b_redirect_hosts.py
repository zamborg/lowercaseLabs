"""Integration tests for M12b: DB-backed redirect host allowlist & /admin/clients Allow/Dismiss."""

from __future__ import annotations

import re

import httpx
from argon2 import PasswordHasher

from tests.conftest import get_test_database_url
from zos.app import create_app
from zos.config import Settings
from zos.db.queries import init_principal_and_space
from zos.vault import generate_secret_key


def _extract_csrf(html_body: str) -> str:
    m = re.search(r'<input type="hidden" name="csrf_token" value="([^"]+)">', html_body)
    assert m is not None
    return m.group(1)


async def test_m12b_redirect_allowlist_seed_reject_allow_dismiss_and_audit(db_pool) -> None:
    await init_principal_and_space(db_pool, principal_slug="owner")
    owner_pw = "m12b-owner-pw"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="https://zos.example.com",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
        allowed_redirect_hosts=("seeded.example.org",),
        dcr_max_per_hour=50,
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://zos.example.com",
            follow_redirects=False,
        ) as client,
    ):
        # 1. Default seed (chatgpt.com, claude.ai) + env seed (seeded.example.org) work immediately
        for uri in (
            "https://chatgpt.com/connector_platform_oauth_redirect",
            "https://claude.ai/api/mcp/auth_callback",
            "https://seeded.example.org/oauth/cb",
        ):
            r = await client.post(
                "/oauth/register",
                json={"client_name": "AllowedApp", "redirect_uris": [uri], "token_endpoint_auth_method": "none"},
            )
            assert r.status_code == 201, f"{uri}: {r.text}"

        # 2. Unlisted HTTPS hosts are rejected and recorded (host + client_name + time only)
        r_rej1 = await client.post(
            "/oauth/register",
            json={
                "client_name": "Cursor Cloud",
                "redirect_uris": ["https://cursor.example/oauth/callback"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert r_rej1.status_code == 400
        assert r_rej1.json()["error"] == "invalid_redirect_uri"

        r_rej2 = await client.post(
            "/oauth/register",
            json={
                "client_name": "SpamBot",
                "redirect_uris": ["https://spam.example/cb"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert r_rej2.status_code == 400

        # Non-HTTPS / malformed schemes are rejected WITHOUT polluting blocked attempts table
        r_bad = await client.post(
            "/oauth/register",
            json={
                "client_name": "BadScheme",
                "redirect_uris": ["http://plain.example/cb"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert r_bad.status_code == 400

        # 3. Log into /admin/clients -> sees both blocked attempts with Allow / Dismiss buttons
        assert (await client.post("/admin/login", data={"passphrase": owner_pw})).status_code == 303
        r_page = await client.get("/admin/clients")
        assert r_page.status_code == 200
        assert "Cursor Cloud" in r_page.text
        assert "tried to connect from <code>cursor.example</code>" in r_page.text
        assert "SpamBot" in r_page.text
        assert "plain.example" not in r_page.text
        csrf = _extract_csrf(r_page.text)

        # CSRF enforced on Allow & Dismiss
        assert (
            await client.post("/admin/clients/redirect-hosts/allow", data={"host": "cursor.example"})
        ).status_code == 403

        # 4. Allow cursor.example -> clears blocked attempt, adds to allowlist, audits
        r_allow = await client.post(
            "/admin/clients/redirect-hosts/allow",
            data={"csrf_token": csrf, "host": "cursor.example"},
        )
        assert r_allow.status_code == 200
        assert "Allowed redirect host &#x27;cursor.example&#x27;" in r_allow.text
        assert "tried to connect from <code>cursor.example</code>" not in r_allow.text

        # Retry DCR for Cursor Cloud now SUCCEEDS!
        r_retry = await client.post(
            "/oauth/register",
            json={
                "client_name": "Cursor Cloud",
                "redirect_uris": ["https://cursor.example/oauth/callback"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert r_retry.status_code == 201

        # 5. Dismiss SpamBot attempt -> removes attempt, does NOT allow host, audits
        async with db_pool.connection() as conn:
            att = await (
                await conn.execute(
                    "SELECT id FROM dcr_rejected_attempts WHERE host = 'spam.example'"
                )
            ).fetchone()
            assert att is not None
            spam_att_id = str(att["id"])

        r_dis = await client.post(
            "/admin/clients/redirect-hosts/dismiss",
            data={"csrf_token": csrf, "attempt_id": spam_att_id},
        )
        assert r_dis.status_code == 200
        assert "Dismissed blocked attempt" in r_dis.text
        assert "SpamBot" not in r_dis.text

        # SpamBot still rejected on DCR
        r_still = await client.post(
            "/oauth/register",
            json={
                "client_name": "SpamBot",
                "redirect_uris": ["https://spam.example/cb"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert r_still.status_code == 400

    # Verify audit events
    async with db_pool.connection() as conn:
        evs = await (
            await conn.execute(
                """
                SELECT type, payload
                FROM events
                WHERE type IN ('redirect_host.allowed', 'redirect_host.dismissed')
                ORDER BY id ASC
                """
            )
        ).fetchall()
    assert [(e["type"], e["payload"]["host"]) for e in evs] == [
        ("redirect_host.allowed", "cursor.example"),
        ("redirect_host.dismissed", "spam.example"),
    ]
