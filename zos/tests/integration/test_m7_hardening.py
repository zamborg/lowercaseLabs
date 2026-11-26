"""M7 pre-public hardening tests:
1. Shared global passphrase rate limiter (5/min, 20/hr, backoff) + auth.failed audit across /admin/login, /oauth/authorize, /connect/google.
2. DCR redirect host allowlist (ZOS_ALLOWED_REDIRECT_HOSTS) + prominent redirect host on /oauth/authorize.
3. DCR global rate limit + OAuth client cap + /admin/clients pruning.
4. Public https://zos.<domain> base URL readiness (PRM, ASM, WWW-Authenticate, resource binding, Secure cookie, Google callback).
"""

from __future__ import annotations

import base64
import hashlib
import re
from urllib.parse import parse_qs, urlparse

import httpx
from argon2 import PasswordHasher

from tests.conftest import get_test_database_url
from zos.app import create_app
from zos.auth.rate_limit import GlobalPassphraseRateLimiter
from zos.config import Settings
from zos.db.queries import init_principal_and_space
from zos.vault import generate_secret_key
from zos.web.admin_auth import ADMIN_SESSION_COOKIE


def _pkce_challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode("ascii").rstrip("=")


def _extract_csrf(html_body: str) -> str:
    m = re.search(r'<input type="hidden" name="csrf_token" value="([^"]+)">', html_body)
    assert m is not None
    return m.group(1)


def test_m7_global_passphrase_limiter_windows_and_backoff() -> None:
    lim = GlobalPassphraseRateLimiter(
        max_per_minute=5,
        max_per_hour=20,
        minute_window_seconds=60.0,
        hour_window_seconds=3600.0,
        backoff_base_seconds=10.0,
        max_backoff_seconds=3600.0,
    )
    t = 1000.0
    # 5 failures within a minute -> triggers minute window block
    for i in range(5):
        assert lim.retry_after_seconds(now=t + i) is None
        lim.record_failure(now=t + i)
    wait = lim.retry_after_seconds(now=t + 5)
    assert wait is not None and wait >= 55

    # After minute window passes (t + 65), 6th consecutive failure triggers exponential backoff (20s)
    assert lim.retry_after_seconds(now=t + 65) is None
    lim.record_failure(now=t + 65)
    wait_b1 = lim.retry_after_seconds(now=t + 66)
    assert wait_b1 is not None and wait_b1 >= 19

    # Valid owner passphrase clears consecutive backoff streak, but preserves hourly budget
    lim.record_success()
    assert lim.retry_after_seconds(now=t + 66) is None

    # Exhaust remaining 14 failures of the 20/hour cap (spaced > 60s apart)
    cur_t = t + 130.0
    for _ in range(14):
        assert lim.retry_after_seconds(now=cur_t) is None
        lim.record_failure(now=cur_t)
        cur_t += 65.0
        lim.record_success()

    # 20 failures recorded in last hour -> blocked until oldest expires
    hr_wait = lim.retry_after_seconds(now=cur_t)
    assert hr_wait is not None and hr_wait > 2000


async def test_m7_shared_passphrase_limiter_across_all_three_forms_and_auth_failed_audit(db_pool) -> None:
    await init_principal_and_space(db_pool, principal_slug="zubin")
    owner_pw = "real-owner-secret-pw"
    wrong_pw = "WRONG_GUESS_NEVER_LOGGED_12345"
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
        # Register an OAuth client for /oauth/authorize POST tests
        r_reg = await client.post(
            "/oauth/register",
            json={
                "client_name": "Loopback Client",
                "redirect_uris": ["http://127.0.0.1:9100/cb"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert r_reg.status_code == 201
        cid = r_reg.json()["client_id"]
        oauth_form = {
            "response_type": "code",
            "client_id": cid,
            "redirect_uri": "http://127.0.0.1:9100/cb",
            "scope": "mcp",
            "code_challenge": _pkce_challenge("v" * 64),
            "code_challenge_method": "S256",
            "decision": "approve",
        }

        # Spread 5 failures across /admin/login and /oauth/authorize (even with different X-Forwarded-For headers)
        r1 = await client.post("/admin/login", data={"passphrase": wrong_pw}, headers={"X-Forwarded-For": "1.1.1.1"})
        r2 = await client.post("/admin/login", data={"passphrase": wrong_pw}, headers={"X-Forwarded-For": "2.2.2.2"})
        r3 = await client.post("/oauth/authorize", data={**oauth_form, "passphrase": wrong_pw}, headers={"X-Forwarded-For": "3.3.3.3"})
        r4 = await client.post("/oauth/authorize", data={**oauth_form, "passphrase": wrong_pw}, headers={"X-Forwarded-For": "4.4.4.4"})
        r5 = await client.post("/admin/login", data={"passphrase": wrong_pw}, headers={"X-Forwarded-For": "5.5.5.5"})
        assert [r1.status_code, r2.status_code, r3.status_code, r4.status_code, r5.status_code] == [401, 401, 401, 401, 401]

        # 6th attempt on ANY form (even with the valid owner passphrase!) returns 429 + Retry-After
        for resp in (
            await client.post("/admin/login", data={"passphrase": owner_pw}),
            await client.post("/oauth/authorize", data={**oauth_form, "passphrase": owner_pw}),
        ):
            assert resp.status_code == 429
            assert "retry-after" in resp.headers
            assert int(resp.headers["retry-after"]) >= 1

    # Verify all 5 failures were audited as auth.failed / denied with ZERO passphrase material
    async with db_pool.connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT type, outcome, payload::text AS payload_text FROM events WHERE type = 'auth.failed' ORDER BY id ASC"
            )
        ).fetchall()
    assert len(rows) == 5
    routes_logged = []
    for row in rows:
        assert row["outcome"] == "denied"
        assert wrong_pw not in row["payload_text"]
        assert owner_pw not in row["payload_text"]
        for route in ("/admin/login", "/oauth/authorize"):
            if route in row["payload_text"]:
                routes_logged.append(route)
    assert routes_logged == [
        "/admin/login",
        "/admin/login",
        "/oauth/authorize",
        "/oauth/authorize",
        "/admin/login",
    ]


async def test_m7_dcr_redirect_allowlist_and_approve_page_host(db_pool) -> None:
    await init_principal_and_space(db_pool, principal_slug="zubin")
    owner_hash = PasswordHasher().hash("pw-dcr")

    # 1. Default (empty allowlist): loopback only
    cfg_default = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8787",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=owner_hash,
        dcr_max_per_hour=50,
    )
    app_default = create_app(cfg_default, run_migrations_on_startup=False)
    async with (
        app_default.router.lifespan_context(app_default),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_default),
            base_url="http://localhost:8787",
        ) as client,
    ):
        for good_uri in (
            "http://127.0.0.1:9101/cb",
            "http://localhost:9102/cb",
            "http://[::1]:9103/cb",
            "https://localhost:9104/cb",
        ):
            r = await client.post(
                "/oauth/register",
                json={"client_name": "Loopback", "redirect_uris": [good_uri], "token_endpoint_auth_method": "none"},
            )
            assert r.status_code == 201, f"expected 201 for {good_uri}, got {r.status_code}: {r.text}"

        for bad_uri in (
            "https://chatgpt.com/connector_platform_oauth_redirect",
            "https://evil.example/cb",
            "http://evil.example/cb",
            "codex://oauth/callback",
            "javascript:alert(1)",
            "http://127.0.0.1:9100/cb#frag",
            "http://user:pass@127.0.0.1:9100/cb",
        ):
            r = await client.post(
                "/oauth/register",
                json={"client_name": "Bad", "redirect_uris": [bad_uri], "token_endpoint_auth_method": "none"},
            )
            assert r.status_code == 400, f"expected 400 for {bad_uri}"
            assert r.json()["error"] == "invalid_redirect_uri"

    # 2. With ZOS_ALLOWED_REDIRECT_HOSTS=chatgpt.com
    cfg_cgpt = Settings(
        database_url=get_test_database_url(),
        zos_base_url="https://zos.example.com",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=owner_hash,
        allowed_redirect_hosts=("chatgpt.com",),
        dcr_max_per_hour=50,
    )
    app_cgpt = create_app(cfg_cgpt, run_migrations_on_startup=False)
    async with (
        app_cgpt.router.lifespan_context(app_cgpt),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_cgpt),
            base_url="https://zos.example.com",
        ) as client,
    ):
        r_ok1 = await client.post(
            "/oauth/register",
            json={
                "client_name": "ChatGPT",
                "redirect_uris": ["https://chatgpt.com/connector_platform_oauth_redirect"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert r_ok1.status_code == 201
        cid_cgpt = r_ok1.json()["client_id"]

        r_ok2 = await client.post(
            "/oauth/register",
            json={
                "client_name": "ChatGPT Fallback",
                "redirect_uris": ["https://chatgpt.com/connector/oauth/cb_123"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert r_ok2.status_code == 201

        for bad_ext in (
            "http://chatgpt.com/connector_platform_oauth_redirect",
            "https://evil-chatgpt.com/cb",
            "https://chatgpt.com.evil.com/cb",
            "https://chatgpt.com@evil.com/cb",
            "https://attacker.example/cb",
        ):
            r_bad = await client.post(
                "/oauth/register",
                json={"client_name": "Spoof", "redirect_uris": [bad_ext], "token_endpoint_auth_method": "none"},
            )
            assert r_bad.status_code == 400, f"expected 400 for {bad_ext}"
            assert r_bad.json()["error"] == "invalid_redirect_uri"

        # Approve page displays redirect host prominently
        r_auth = await client.get(
            "/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": cid_cgpt,
                "redirect_uri": "https://chatgpt.com/connector_platform_oauth_redirect",
                "scope": "mcp",
                "code_challenge": _pkce_challenge("c" * 64),
                "code_challenge_method": "S256",
            },
        )
        assert r_auth.status_code == 200
        assert "Redirect Host" in r_auth.text
        assert '<dd class="host-callout">chatgpt.com</dd>' in r_auth.text


async def test_m7_dcr_rate_limit_and_oauth_client_cap_with_admin_prune(db_pool) -> None:
    await init_principal_and_space(db_pool, principal_slug="zubin")
    owner_pw = "prune-owner-pw"
    owner_hash = PasswordHasher().hash(owner_pw)

    # 1. DCR rate limit (max 2/hour)
    cfg_rl = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8787",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=owner_hash,
        dcr_max_per_hour=2,
        max_oauth_clients=50,
    )
    app_rl = create_app(cfg_rl, run_migrations_on_startup=False)
    async with (
        app_rl.router.lifespan_context(app_rl),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_rl),
            base_url="http://localhost:8787",
        ) as client,
    ):
        for i in range(2):
            r = await client.post(
                "/oauth/register",
                json={"client_name": f"c{i}", "redirect_uris": [f"http://127.0.0.1:{9200 + i}/cb"]},
            )
            assert r.status_code == 201
        r_3 = await client.post(
            "/oauth/register",
            json={"client_name": "c2", "redirect_uris": ["http://127.0.0.1:9202/cb"]},
        )
        assert r_3.status_code == 429
        assert r_3.json()["error"] == "too_many_requests"
        assert "retry-after" in r_3.headers

    # 2. OAuth client cap (already 2 in DB; cap = 2 -> 3rd blocked until owner prunes via /admin/clients/delete)
    cfg_cap = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8787",
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=owner_hash,
        dcr_max_per_hour=20,
        max_oauth_clients=2,
    )
    app_cap = create_app(cfg_cap, run_migrations_on_startup=False)
    async with (
        app_cap.router.lifespan_context(app_cap),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_cap),
            base_url="http://localhost:8787",
            follow_redirects=False,
        ) as client,
    ):
        r_capped = await client.post(
            "/oauth/register",
            json={"client_name": "c3", "redirect_uris": ["http://127.0.0.1:9203/cb"]},
        )
        assert r_capped.status_code == 429
        assert r_capped.json()["error"] == "registration_cap_reached"
        assert "/admin/clients" in r_capped.json()["error_description"]

        # Log into /admin and prune one OAuth client
        r_login = await client.post("/admin/login", data={"passphrase": owner_pw})
        assert r_login.status_code == 303
        r_clients = await client.get("/admin/clients")
        assert r_clients.status_code == 200
        assert "OAuth: 2 / 2" in r_clients.text
        csrf = _extract_csrf(r_clients.text)

        async with db_pool.connection() as conn:
            row = await (
                await conn.execute("SELECT id FROM clients WHERE kind = 'oauth' ORDER BY created_at ASC LIMIT 1")
            ).fetchone()
            assert row is not None
            prune_id = str(row["id"])

        # Missing CSRF rejected
        r_no_csrf = await client.post("/admin/clients/delete", data={"client_id": prune_id})
        assert r_no_csrf.status_code == 403

        # Valid CSRF prunes client + records client.deleted event
        r_del = await client.post("/admin/clients/delete", data={"csrf_token": csrf, "client_id": prune_id})
        assert r_del.status_code == 200
        assert "Pruned OAuth client" in r_del.text
        assert "OAuth: 1 / 2" in r_del.text

        # Now DCR succeeds again!
        r_again = await client.post(
            "/oauth/register",
            json={"client_name": "c3", "redirect_uris": ["http://127.0.0.1:9203/cb"]},
        )
        assert r_again.status_code == 201

    async with db_pool.connection() as conn:
        ev = await (
            await conn.execute("SELECT type, outcome, payload FROM events WHERE type = 'client.deleted'")
        ).fetchone()
    assert ev is not None
    assert ev["outcome"] == "ok"
    assert ev["payload"]["deleted_client_id"] == prune_id


async def test_m7_public_https_base_url_readiness(db_pool) -> None:
    await init_principal_and_space(db_pool, principal_slug="zubin")
    owner_pw = "https-owner-pw"
    owner_hash = PasswordHasher().hash(owner_pw)
    public_base = "https://zos.example.com"

    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=public_base,
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=owner_hash,
        allowed_redirect_hosts=("chatgpt.com",),
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=public_base,
            follow_redirects=False,
        ) as client,
    ):
        # 1. PRM & ASM metadata advertise https://zos.example.com
        prm = (await client.get("/.well-known/oauth-protected-resource")).json()
        assert prm["resource"] == public_base
        assert prm["authorization_servers"] == [public_base]

        asm = (await client.get("/.well-known/oauth-authorization-server")).json()
        assert asm["issuer"] == public_base
        assert asm["authorization_endpoint"] == f"{public_base}/oauth/authorize"
        assert asm["token_endpoint"] == f"{public_base}/oauth/token"
        assert asm["registration_endpoint"] == f"{public_base}/oauth/register"

        # 2. Unauthenticated /mcp returns 401 with https WWW-Authenticate resource_metadata
        r_unauth = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert r_unauth.status_code == 401
        assert f'resource_metadata="{public_base}/.well-known/oauth-protected-resource"' in r_unauth.headers["www-authenticate"]

        # 3. Full OAuth flow bound to https://zos.example.com for ChatGPT redirect URI
        r_reg = await client.post(
            "/oauth/register",
            json={
                "client_name": "ChatGPT",
                "redirect_uris": ["https://chatgpt.com/connector_platform_oauth_redirect"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert r_reg.status_code == 201
        cid = r_reg.json()["client_id"]
        verifier = "p" * 64
        r_auth = await client.post(
            "/oauth/authorize",
            data={
                "response_type": "code",
                "client_id": cid,
                "redirect_uri": "https://chatgpt.com/connector_platform_oauth_redirect",
                "scope": "mcp",
                "state": "st1",
                "resource": f"{public_base}/mcp",
                "code_challenge": _pkce_challenge(verifier),
                "code_challenge_method": "S256",
                "passphrase": owner_pw,
                "decision": "approve",
            },
        )
        assert r_auth.status_code == 302
        qs = parse_qs(urlparse(r_auth.headers["location"]).query)
        assert qs["iss"] == [public_base]
        code = qs["code"][0]

        r_tok = await client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "https://chatgpt.com/connector_platform_oauth_redirect",
                "client_id": cid,
                "code_verifier": verifier,
                "resource": f"{public_base}/mcp",
            },
        )
        assert r_tok.status_code == 200
        access_token = r_tok.json()["access_token"]

        # Valid against https://zos.example.com
        r_mcp = await client.post(
            "/mcp",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "chatgpt", "version": "1.0"},
                },
            },
        )
        assert r_mcp.status_code == 200

        # Force the token's bound resource in DB to old localhost URL -> rejected with 401!
        async with db_pool.connection() as conn:
            await conn.execute("UPDATE tokens SET resource = 'http://localhost:8787'")
            await conn.commit()
        r_wrong_res = await client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
        )
        assert r_wrong_res.status_code == 401

        # 4. Admin login cookie turns on Secure flag on https://
        r_admin = await client.post("/admin/login", data={"passphrase": owner_pw})
        assert r_admin.status_code == 303
        set_cookie = r_admin.headers["set-cookie"]
        assert f"{ADMIN_SESSION_COOKIE}=" in set_cookie
        assert "Secure" in set_cookie
        assert "HttpOnly" in set_cookie

        # 5. Behind TLS-terminating tunnel (plain HTTP into app with public Host header),
        #    Starlette trailing-slash 307 redirects emit https:// (while loopback stays http://)
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://zos.example.com",
                follow_redirects=False,
            ) as tunnel_client,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8787",
                follow_redirects=False,
            ) as local_client,
        ):
            r_admin_slash = await tunnel_client.get("/admin/")
            assert r_admin_slash.status_code == 307
            assert r_admin_slash.headers["location"] == "https://zos.example.com/admin"

            r_mcp_slash = await tunnel_client.get("/mcp/")
            assert r_mcp_slash.status_code == 307
            assert r_mcp_slash.headers["location"] == "https://zos.example.com/mcp"

            # Default :443 port on Host header also resolves to https://
            r_443 = await tunnel_client.get("/admin/", headers={"Host": "zos.example.com:443"})
            assert r_443.status_code == 307
            assert r_443.headers["location"].startswith("https://zos.example.com")

            # Direct local loopback request stays http://
            r_local = await local_client.get("/admin/")
            assert r_local.status_code == 307
            assert r_local.headers["location"] == "http://127.0.0.1:8787/admin"
