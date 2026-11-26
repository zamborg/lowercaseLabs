"""Integration tests for M6 owner admin dashboard (/admin): auth, CSRF, rate limiting,
all 4 pages, audited mutations, activity privacy (no raw args), pagination/filtering, and CLI parity.
"""

from __future__ import annotations

import re
import time

import httpx
from argon2 import PasswordHasher

from tests.conftest import get_test_database_url
from zos.admin_service import AdminService
from zos.app import create_app
from zos.audit import args_digest, record_event
from zos.config import Settings
from zos.db.queries import init_principal_and_space
from zos.vault import generate_secret_key
from zos.web.admin_auth import (
    ADMIN_SESSION_COOKIE,
    AdminSession,
    AdminSessionManager,
    derive_session_signing_key,
)


def _extract_csrf(html_body: str) -> str:
    m = re.search(r'<input type="hidden" name="csrf_token" value="([^"]+)">', html_body)
    assert m is not None, "csrf_token hidden input not found"
    return m.group(1)


async def test_admin_auth_cookie_flags_expiry_tamper_rate_limit_and_csrf(db_pool) -> None:
    await init_principal_and_space(db_pool, principal_slug="zubin")
    owner_pw = "admin-pass-123"
    owner_hash = PasswordHasher().hash(owner_pw)
    secret_key = generate_secret_key()

    # 1. Verify HTTPS base URL sets Secure cookie flag
    https_cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="https://zos.example.com",
        zos_secret_key=secret_key,
        zos_owner_password_hash=owner_hash,
    )
    https_app = create_app(https_cfg, run_migrations_on_startup=False)
    async with (
        https_app.router.lifespan_context(https_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=https_app),
            base_url="https://zos.example.com",
            follow_redirects=False,
        ) as client,
    ):
        r_login = await client.post("/admin/login", data={"passphrase": owner_pw})
        assert r_login.status_code == 303
        set_cookie = r_login.headers["set-cookie"]
        assert f"{ADMIN_SESSION_COOKIE}=" in set_cookie
        assert "HttpOnly" in set_cookie
        assert "SameSite=strict" in set_cookie.lower().replace("samesite=strict", "SameSite=strict")
        assert "Secure" in set_cookie
        assert "Max-Age=43200" in set_cookie

    # 2. Full auth, rate-limit, expiry, tamper, CSRF, and security-header checks on localhost HTTP
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8787",
        zos_secret_key=secret_key,
        zos_owner_password_hash=owner_hash,
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
        # Unauthenticated -> redirect to /admin/login + security headers
        r_unauth = await client.get("/admin")
        assert r_unauth.status_code == 303
        assert r_unauth.headers["location"] == "/admin/login"
        assert r_unauth.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in r_unauth.headers["content-security-policy"]
        assert r_unauth.headers["cache-control"] == "no-store"

        # 5 bad logins -> 401; 6th -> 429 rate-limited with Retry-After
        for _ in range(5):
            r_bad = await client.post("/admin/login", data={"passphrase": "wrong"})
            assert r_bad.status_code == 401
        r_rl = await client.post("/admin/login", data={"passphrase": owner_pw})
        assert r_rl.status_code == 429
        assert "retry-after" in r_rl.headers

    # Separate app instance (fresh limiter) to test expired/tampered cookies & CSRF
    app2 = create_app(cfg, run_migrations_on_startup=False)
    async with (
        app2.router.lifespan_context(app2),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app2),
            base_url="http://localhost:8787",
            follow_redirects=False,
        ) as client,
    ):
        # Expired cookie rejected -> 303 to /admin/login
        mgr = AdminSessionManager(
            signing_key=derive_session_signing_key(
                secret_key=secret_key, owner_password_hash=owner_hash
            ),
            secure_cookie=False,
        )
        expired_cookie = mgr.sign(
            AdminSession(
                issued_at=int(time.time()) - 7200,
                expires_at=int(time.time()) - 10,
                csrf_token="x",
            )
        )
        r_exp = await client.get(
            "/admin", headers={"Cookie": f"{ADMIN_SESSION_COOKIE}={expired_cookie}"}
        )
        assert r_exp.status_code == 303
        assert r_exp.headers["location"] == "/admin/login"

        # Tampered cookie rejected
        r_tamp = await client.get(
            "/admin", headers={"Cookie": f"{ADMIN_SESSION_COOKIE}={expired_cookie}tamper"}
        )
        assert r_tamp.status_code == 303

        # Real login succeeds
        r_ok = await client.post("/admin/login", data={"passphrase": owner_pw})
        assert r_ok.status_code == 303
        assert "Secure" not in r_ok.headers["set-cookie"]

        r_ov = await client.get("/admin")
        assert r_ov.status_code == 200
        csrf = _extract_csrf(r_ov.text)

        # Every POST route rejects missing or wrong CSRF token with 403
        for path, payload in (
            ("/admin/clients/create", {"name": "c1", "trust_ceiling": "owner"}),
            ("/admin/clients/revoke", {"client_id": "00000000-0000-0000-0000-000000000000"}),
            ("/admin/policy/set", {"pattern": "mac_shell_run", "decision": "allow"}),
            ("/admin/policy/clear", {"pattern": "mac_shell_run"}),
            ("/admin/logout", {}),
        ):
            r_no = await client.post(path, data=payload)
            assert r_no.status_code == 403, f"expected 403 missing CSRF on {path}"
            r_bad_csrf = await client.post(path, data={**payload, "csrf_token": "wrong-csrf"})
            assert r_bad_csrf.status_code == 403, f"expected 403 bad CSRF on {path}"

        # Valid CSRF logout clears cookie
        r_out = await client.post("/admin/logout", data={"csrf_token": csrf})
        assert r_out.status_code == 303
        r_after = await client.get("/admin")
        assert r_after.status_code == 303


async def test_admin_pages_mutations_audit_privacy_pagination_and_cli_parity(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    owner_pw = "owner-dash-pass"
    owner_hash = PasswordHasher().hash(owner_pw)
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8787",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=owner_hash,
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
        r_login = await client.post("/admin/login", data={"passphrase": owner_pw})
        assert r_login.status_code == 303

        # 1. Overview page shows providers, backend, active sessions, and /connect/google
        r_ov = await client.get("/admin")
        assert r_ov.status_code == 200
        assert "gcal" in r_ov.text
        assert "mcp" in r_ov.text
        assert "/connect/google" in r_ov.text
        csrf = _extract_csrf(r_ov.text)

        # 2. Clients: create bearer client -> token shown ONCE, audited
        r_create = await client.post(
            "/admin/clients/create",
            data={"csrf_token": csrf, "name": "web-client", "trust_ceiling": "owner"},
        )
        assert r_create.status_code == 200
        tok_match = re.search(r"zos_b_[A-Za-z0-9_-]+", r_create.text)
        assert tok_match is not None
        raw_token = tok_match.group(0)

        # Subsequent GET /admin/clients must NOT show the token again
        r_clients = await client.get("/admin/clients")
        assert r_clients.status_code == 200
        assert "web-client" in r_clients.text
        assert raw_token not in r_clients.text

        # Mint second bearer token for same client
        r_tok2 = await client.post(
            "/admin/clients/token",
            data={"csrf_token": csrf, "client_name": "web-client"},
        )
        assert r_tok2.status_code == 200
        assert re.search(r"zos_b_[A-Za-z0-9_-]+", r_tok2.text) is not None

        # Look up client UUID
        async with db_pool.connection() as conn:
            c_row = await (
                await conn.execute("SELECT id FROM clients WHERE name = 'web-client'")
            ).fetchone()
            assert c_row is not None
            web_client_id = str(c_row["id"])

        # 3. Tools & Policy: view matrix, set override, clear override -> audited
        r_pol = await client.get("/admin/policy")
        assert r_pol.status_code == 200
        assert "zos_status" in r_pol.text
        assert "mac_shell_run" in r_pol.text
        assert "web-client" in r_pol.text

        r_set = await client.post(
            "/admin/policy/set",
            data={
                "csrf_token": csrf,
                "client_id": web_client_id,
                "pattern": "mac_shell_run",
                "decision": "allow",
            },
        )
        assert r_set.status_code == 200
        assert "Policy set" in r_set.text

        r_clr = await client.post(
            "/admin/policy/clear",
            data={"csrf_token": csrf, "client_id": web_client_id, "pattern": "mac_shell_run"},
        )
        assert r_clr.status_code == 200
        assert "Cleared override" in r_clr.text

        # 4. Revoke client via dashboard -> tokens revoked + 401 on /mcp + token.revoked event
        r_rev = await client.post(
            "/admin/clients/revoke",
            data={"csrf_token": csrf, "client_id": web_client_id},
        )
        assert r_rev.status_code == 200
        assert "Revoked 2 active token(s)." in r_rev.text

        r_mcp_revoked = await client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {raw_token}"},
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        )
        assert r_mcp_revoked.status_code == 401

        # 5. Activity page: never renders raw args, shows args_digest, supports filtering & cursor pagination
        secret_arg = "TOP_SECRET_RAW_ARG_NEVER_RENDER_98765"
        expected_hex_prefix = args_digest({"body": secret_arg}).hex()[:16]
        for i in range(4):
            await record_event(
                db_pool,
                principal_id=p_id,
                space_id=s_id,
                client_id=c_row["id"],
                event_type="tool.called",
                tool=f"memory_put_{i}",
                decision="allow",
                outcome="ok" if i % 2 == 0 else "denied",
                args={"body": secret_arg} if i == 0 else {"i": i},
            )

        r_act = await client.get("/admin/activity", params={"type": "tool.called", "limit": "2"})
        assert r_act.status_code == 200
        assert secret_arg not in r_act.text
        assert "memory_put_3" in r_act.text
        assert "memory_put_2" in r_act.text
        assert "memory_put_0" not in r_act.text  # on page 2

        # Follow cursor pagination to page 2
        next_match = re.search(r'href="(/admin/activity\?[^"]*before_id=\d+[^"]*)"', r_act.text)
        assert next_match is not None
        next_url = next_match.group(1).replace("&amp;", "&")
        r_act2 = await client.get(next_url)
        assert r_act2.status_code == 200
        assert secret_arg not in r_act2.text
        assert "memory_put_0" in r_act2.text
        assert expected_hex_prefix in r_act2.text

        # Outcome filter
        r_denied = await client.get(
            "/admin/activity", params={"type": "tool.called", "outcome": "denied"}
        )
        assert r_denied.status_code == 200
        assert "memory_put_3" in r_denied.text
        assert "memory_put_2" not in r_denied.text

        # 6. Service-layer parity with CLI: AdminService mutations match /admin audit event shapes
        svc = AdminService(db_pool)
        cli_res = await svc.register_bearer_client(name="cli-client", trust_ceiling="owner")
        await svc.set_policy(pattern="gmail_send", decision="deny", client_identifier="cli-client")
        await svc.clear_policy(pattern="gmail_send", client_identifier="cli-client")
        assert await svc.revoke_tokens(client_identifier=str(cli_res["client"]["id"])) == 1

        async with db_pool.connection() as conn:
            ev_types = [
                r["type"]
                for r in await (
                    await conn.execute("SELECT type FROM events ORDER BY id ASC")
                ).fetchall()
            ]
        for expected_type in (
            "client.registered",
            "token.issued",
            "policy.changed",
            "token.revoked",
            "tool.called",
        ):
            assert expected_type in ev_types
        # Both web-client and cli-client produced identical policy.changed event sequences
        async with db_pool.connection() as conn:
            web_pol = [
                (r["decision"], r["payload"]["action"])
                for r in await (
                    await conn.execute(
                        "SELECT decision, payload FROM events WHERE type = 'policy.changed' AND client_id = %s ORDER BY id ASC",
                        (c_row["id"],),
                    )
                ).fetchall()
            ]
            cli_pol = [
                (r["decision"], r["payload"]["action"])
                for r in await (
                    await conn.execute(
                        "SELECT decision, payload FROM events WHERE type = 'policy.changed' AND client_id = %s ORDER BY id ASC",
                        (cli_res["client"]["id"],),
                    )
                ).fetchall()
            ]
        assert web_pol == [("allow", "set"), ("cleared", "cleared")]
        assert cli_pol == [("deny", "set"), ("cleared", "cleared")]
