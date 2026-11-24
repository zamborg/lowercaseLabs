"""M2.5 E2E tests: full MCP SDK OAuth 2.1 flow (discovery -> DCR -> authorize -> PKCE token -> tool call),
refresh token rotation, reuse family revocation, and revocation -> 401.
"""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, urlparse

import httpx
import httpx2
from argon2 import PasswordHasher
from mcp.client.auth import OAuthClientProvider
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from pydantic import AnyUrl

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.config import Settings
from zos.db.queries import init_principal_and_space, revoke_token


class InMemoryTokenStorage:
    def __init__(self) -> None:
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client_info = client_info


def _extract_hidden_inputs(html_body: str) -> dict[str, str]:
    pairs = re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">', html_body)
    return dict(pairs)


async def test_m2_full_oauth_flow_refresh_rotation_reuse_revocation_and_401(db_pool) -> None:
    await init_principal_and_space(db_pool, principal_slug="zubin")

    owner_passphrase = "correct-horse-battery-staple"
    owner_hash = PasswordHasher().hash(owner_passphrase)

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    mcp_url = f"{base_url}/mcp"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=base_url,
        zos_owner_password_hash=owner_hash,
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    storage = InMemoryTokenStorage()
    captured_callback: dict[str, str | None] = {}

    async def redirect_handler(authorize_url: str) -> None:
        async with httpx.AsyncClient(follow_redirects=False) as web:
            # 1. GET authorize page shows client name + redirect URI
            get_resp = await web.get(authorize_url)
            assert get_resp.status_code == 200
            assert "SDK Test Client" in get_resp.text
            assert "http://127.0.0.1:9999/cb" in get_resp.text

            hidden = _extract_hidden_inputs(get_resp.text)

            # 2. Wrong passphrase fails with 401
            bad_resp = await web.post(
                f"{base_url}/oauth/authorize",
                data={**hidden, "passphrase": "wrong-password", "decision": "approve"},
            )
            assert bad_resp.status_code == 401

            # 3. Correct passphrase redirects with code, state, iss
            ok_resp = await web.post(
                f"{base_url}/oauth/authorize",
                data={**hidden, "passphrase": owner_passphrase, "decision": "approve"},
            )
            assert ok_resp.status_code == 302
            loc = ok_resp.headers["location"]
            qs = parse_qs(urlparse(loc).query)
            captured_callback["code"] = qs["code"][0]
            captured_callback["state"] = qs.get("state", [None])[0]
            captured_callback["iss"] = qs.get("iss", [None])[0]

    async def callback_handler() -> AuthorizationCodeResult:
        assert captured_callback.get("code")
        return AuthorizationCodeResult(
            code=captured_callback["code"] or "",
            state=captured_callback.get("state"),
            iss=captured_callback.get("iss"),
        )

    oauth_provider = OAuthClientProvider(
        server_url=mcp_url,
        client_metadata=OAuthClientMetadata(
            client_name="SDK Test Client",
            redirect_uris=[AnyUrl("http://127.0.0.1:9999/cb")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        ),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )

    async with run_asgi_server(app, port=port):
        # 1. Full SDK OAuth flow from 401 discovery -> DCR -> authorize -> PKCE token -> MCP tool call
        async with (
            httpx2.AsyncClient(auth=oauth_provider) as http_client,
            streamable_http_client(mcp_url, http_client=http_client) as (read_s, write_s),
            ClientSession(read_s, write_s) as session,
        ):
            await session.initialize()
            put_res = await session.call_tool(
                "memory_put",
                {"key": "oauth-hello", "content": "Authenticated via OAuth 2.1!"},
            )
            assert not put_res.is_error
            get_res = await session.call_tool("memory_get", {"key": "oauth-hello"})
            assert json.loads(get_res.content[0].text)["found"] is True

        assert storage.tokens is not None
        assert storage.client_info is not None
        rt1 = storage.tokens.refresh_token
        at1 = storage.tokens.access_token
        oauth_client_id = storage.client_info.client_id
        assert rt1 and at1

        async with httpx.AsyncClient() as raw_http:
            # 2. Refresh token rotation: rt1 -> (at2, rt2)
            r_rot = await raw_http.post(
                f"{base_url}/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": rt1,
                    "client_id": oauth_client_id,
                    "resource": base_url,
                },
            )
            assert r_rot.status_code == 200, r_rot.text
            pair2 = r_rot.json()
            at2 = pair2["access_token"]
            rt2 = pair2["refresh_token"]
            assert at2 != at1 and rt2 != rt1

            # Verify at2 works against /mcp (initialize returns 200)
            r_ok = await raw_http.post(
                mcp_url,
                headers={
                    "Authorization": f"Bearer {at2}",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "probe", "version": "0.1"},
                    },
                },
            )
            assert r_ok.status_code == 200

            # 3. Reuse of already-rotated rt1 revokes the ENTIRE family (including at2 and rt2!)
            r_reuse = await raw_http.post(
                f"{base_url}/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": rt1,
                    "client_id": oauth_client_id,
                },
            )
            assert r_reuse.status_code == 400
            assert r_reuse.json()["error"] == "invalid_grant"

            # Now rt2 must fail refresh...
            r_rt2 = await raw_http.post(
                f"{base_url}/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": rt2,
                    "client_id": oauth_client_id,
                },
            )
            assert r_rt2.status_code == 400

            # ...and at2 must return 401 on /mcp with resource_metadata
            r_at2 = await raw_http.post(
                mcp_url,
                headers={"Authorization": f"Bearer {at2}"},
                json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
            )
            assert r_at2.status_code == 401
            assert "resource_metadata=" in r_at2.headers["www-authenticate"]

            # 4. Explicit revocation via revoke_token also -> 401
            # (at1 was in the same family so already revoked; verify 401 too)
            r_at1 = await raw_http.post(
                mcp_url,
                headers={"Authorization": f"Bearer {at1}"},
                json={"jsonrpc": "2.0", "id": 3, "method": "ping"},
            )
            assert r_at1.status_code == 401

    # Verify fresh revoke_token call returns 0 since family already revoked
    assert await revoke_token(db_pool, raw_token=at2) == 0


async def test_m2_oauth_edge_cases_dcr_pkce_code_reuse_expiry_and_family_isolation(db_pool) -> None:
    import base64
    import hashlib
    from datetime import UTC, datetime, timedelta

    await init_principal_and_space(db_pool, principal_slug="zubin")
    owner_pass = "edge-case-passphrase"
    owner_hash = PasswordHasher().hash(owner_pass)

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=base_url,
        zos_owner_password_hash=owner_hash,
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    async def authorize_and_get_code(
        web: httpx.AsyncClient,
        *,
        client_id: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> str:
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("ascii")).digest())
            .decode("ascii")
            .rstrip("=")
        )
        g = await web.get(
            f"{base_url}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "scope": "mcp",
                "resource": base_url,
            },
        )
        assert g.status_code == 200
        assert g.headers.get("x-frame-options") == "DENY"
        hidden = _extract_hidden_inputs(g.text)
        p = await web.post(
            f"{base_url}/oauth/authorize",
            data={**hidden, "passphrase": owner_pass, "decision": "approve"},
        )
        assert p.status_code == 302
        return parse_qs(urlparse(p.headers["location"]).query)["code"][0]

    async with run_asgi_server(app, port=port), httpx.AsyncClient(follow_redirects=False) as web:
        # 1. DCR validation: reject non-loopback http, reject non-'none' auth method, allow custom scheme & duplicate names
        r_http = await web.post(
            f"{base_url}/oauth/register",
            json={"client_name": "Evil", "redirect_uris": ["http://evil.example/cb"], "token_endpoint_auth_method": "none"},
        )
        assert r_http.status_code == 400

        r_method = await web.post(
            f"{base_url}/oauth/register",
            json={"client_name": "Secret", "redirect_uris": ["http://127.0.0.1:9001/cb"], "token_endpoint_auth_method": "client_secret_post"},
        )
        assert r_method.status_code == 400

        # Two OAuth clients with the SAME display name succeed (partial unique index applies to bearer only)
        r_c1 = await web.post(
            f"{base_url}/oauth/register",
            json={"client_name": "Claude Code", "redirect_uris": ["http://127.0.0.1:9001/cb"], "token_endpoint_auth_method": "none"},
        )
        r_c2 = await web.post(
            f"{base_url}/oauth/register",
            json={"client_name": "Claude Code", "redirect_uris": ["http://localhost:9002/cb"], "token_endpoint_auth_method": "none"},
        )
        assert r_c1.status_code == 201 and r_c2.status_code == 201
        cid1 = r_c1.json()["client_id"]
        cid2 = r_c2.json()["client_id"]
        assert cid1 != cid2

        # 2. Wrong PKCE code_verifier fails (and consumes the single-use code!)
        v_ok = "a" * 64
        code_bad_pkce = await authorize_and_get_code(
            web, client_id=cid1, redirect_uri="http://127.0.0.1:9001/cb", code_verifier=v_ok
        )
        r_bad_pkce = await web.post(
            f"{base_url}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code_bad_pkce,
                "redirect_uri": "http://127.0.0.1:9001/cb",
                "client_id": cid1,
                "code_verifier": "b" * 64,
            },
        )
        assert r_bad_pkce.status_code == 400

        # 3. Expired auth code fails
        code_exp = await authorize_and_get_code(
            web, client_id=cid1, redirect_uri="http://127.0.0.1:9001/cb", code_verifier=v_ok
        )
        async with db_pool.connection() as conn:
            await conn.execute(
                "UPDATE oauth_codes SET expires_at = %s",
                (datetime.now(UTC) - timedelta(seconds=5),),
            )
            await conn.commit()
        r_exp = await web.post(
            f"{base_url}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code_exp,
                "redirect_uri": "http://127.0.0.1:9001/cb",
                "client_id": cid1,
                "code_verifier": v_ok,
            },
        )
        assert r_exp.status_code == 400

        # 4. Single-use code replay fails; refresh family reuse in cid1 does NOT revoke cid2's family
        code1 = await authorize_and_get_code(
            web, client_id=cid1, redirect_uri="http://127.0.0.1:9001/cb", code_verifier=v_ok
        )
        t1 = await web.post(
            f"{base_url}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code1,
                "redirect_uri": "http://127.0.0.1:9001/cb",
                "client_id": cid1,
                "code_verifier": v_ok,
            },
        )
        assert t1.status_code == 200
        # Replay same code -> 400
        t1_replay = await web.post(
            f"{base_url}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code1,
                "redirect_uri": "http://127.0.0.1:9001/cb",
                "client_id": cid1,
                "code_verifier": v_ok,
            },
        )
        assert t1_replay.status_code == 400

        # Issue separate family for cid2
        code2 = await authorize_and_get_code(
            web, client_id=cid2, redirect_uri="http://localhost:9002/cb", code_verifier=v_ok
        )
        t2 = await web.post(
            f"{base_url}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code2,
                "redirect_uri": "http://localhost:9002/cb",
                "client_id": cid2,
                "code_verifier": v_ok,
            },
        )
        assert t2.status_code == 200
        cid2_at = t2.json()["access_token"]

        # Rotate cid1 rt1 -> rt2, then reuse rt1 -> revokes cid1 family ONLY
        rt1 = t1.json()["refresh_token"]
        rot1 = await web.post(
            f"{base_url}/oauth/token",
            data={"grant_type": "refresh_token", "refresh_token": rt1, "client_id": cid1},
        )
        assert rot1.status_code == 200
        reuse1 = await web.post(
            f"{base_url}/oauth/token",
            data={"grant_type": "refresh_token", "refresh_token": rt1, "client_id": cid1},
        )
        assert reuse1.status_code == 400

        # cid2's access token MUST still work on /mcp!
        r_cid2 = await web.post(
            f"{base_url}/mcp",
            headers={
                "Authorization": f"Bearer {cid2_at}",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "c2", "version": "0.1"},
                },
            },
        )
        assert r_cid2.status_code == 200
