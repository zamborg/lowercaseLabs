"""Integration tests for M9c async approvals for non-elicitation MCP clients."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

import httpx
import httpx2
import mcp_types as types
from argon2 import PasswordHasher
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.auth.context import CallContext
from zos.config import Settings
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.providers.base import ProviderTool, ToolProvider
from zos.vault import generate_secret_key


class FakeExternalShellProvider(ToolProvider):
    namespace = "demo"
    kind = "hosted"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def is_online(self) -> bool:
        return True

    async def list_tools(self) -> list[ProviderTool]:
        return [
            ProviderTool(
                name="run",
                description="Run a fake external action",
                input_schema={
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
                destructive_hint=True,
                idempotent_hint=False,
            )
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: CallContext,
    ) -> types.CallToolResult:
        self.calls.append(dict(arguments))
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"ran:{arguments.get('cmd', '')}")],
            is_error=False,
        )


def _extract_csrf(html_text: str) -> str:
    m = re.search(r'name="csrf_token"\s+value="([^"]+)"', html_text)
    assert m is not None, "CSRF token not found in HTML"
    return m.group(1)


def _extract_approval_id(text: str, base_url: str) -> str:
    escaped_base = re.escape(base_url.rstrip("/"))
    m = re.search(rf"{escaped_base}/admin/approvals/([A-Za-z0-9_-]+)", text)
    assert m is not None, f"Approval URL not found in: {text!r}"
    return m.group(1)


async def test_m9c_async_approvals_hold_approve_deny_csrf_encryption_and_audit(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]
    client = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="phone-claude")

    owner_pw = "owner-m9c-pass"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    provider = FakeExternalShellProvider()
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=base_url,
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
        approval_wait_seconds=5.0,
    )
    app = create_app(cfg, extra_providers=[provider], run_migrations_on_startup=False)

    xss_cmd = "echo <script>alert(1)</script> && uname -a"

    async with run_asgi_server(app, port=port):
        http_mcp = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client['token']}"})
        async with (
            http_mcp,
            httpx.AsyncClient(base_url=base_url, follow_redirects=False) as web,
            streamable_http_client(f"{base_url}/mcp", http_client=http_mcp) as (r, w),
            ClientSession(r, w, elicitation_callback=None) as sess,
        ):
            await sess.initialize()

            # Sign into /admin
            r_login = await web.post("/admin/login", data={"passphrase": owner_pw})
            assert r_login.status_code == 303

            # 1. Start held call (no elicitation) -> approve within wait window -> returns real result!
            call_task = asyncio.create_task(sess.call_tool("demo_run", {"cmd": xss_cmd}))

            # Wait until pending approval row appears in DB
            approval_id = None
            for _ in range(50):
                async with db_pool.connection() as conn:
                    cur = await conn.execute(
                        "SELECT id, args_ciphertext, args_nonce FROM pending_approvals WHERE status = 'pending'"
                    )
                    row = await cur.fetchone()
                if row is not None:
                    approval_id = str(row["id"])
                    # Encrypted at rest: ciphertext/nonce present and plaintext NOT stored
                    assert row["args_ciphertext"] is not None
                    assert row["args_nonce"] is not None
                    assert b"alert(1)" not in bytes(row["args_ciphertext"])
                    assert b"uname" not in bytes(row["args_ciphertext"])
                    break
                await asyncio.sleep(0.05)
            assert approval_id is not None

            # Overview shows pending count badge
            r_ov = await web.get("/admin")
            assert r_ov.status_code == 200
            assert "1 pending" in r_ov.text

            # /admin/approvals lists pending approval without leaking raw args
            r_list = await web.get("/admin/approvals")
            assert r_list.status_code == 200
            assert "phone-claude" in r_list.text
            assert "demo_run" in r_list.text
            assert f"/admin/approvals/{approval_id}" in r_list.text
            assert "uname -a" not in r_list.text

            # /admin/approvals/<id> shows decrypted, HTML-escaped args
            r_det = await web.get(f"/admin/approvals/{approval_id}")
            assert r_det.status_code == 200
            assert "&lt;script&gt;alert(1)&lt;/script&gt;" in r_det.text
            assert "<script>alert(1)</script>" not in r_det.text
            csrf = _extract_csrf(r_det.text)

            # CSRF enforced on both approve and deny (403 on bad/missing token)
            r_bad_approve = await web.post(
                f"/admin/approvals/{approval_id}/approve",
                data={"csrf_token": "wrong-token"},
            )
            assert r_bad_approve.status_code == 403
            r_bad_deny = await web.post(
                f"/admin/approvals/{approval_id}/deny",
                data={},
            )
            assert r_bad_deny.status_code == 403

            # Approve via CSRF-protected POST
            r_ok = await web.post(
                f"/admin/approvals/{approval_id}/approve",
                data={"csrf_token": csrf},
            )
            assert r_ok.status_code == 200
            assert "Approved" in r_ok.text

            # Held MCP tool call resolves with real execution result!
            res = await asyncio.wait_for(call_task, timeout=5.0)
            assert res.is_error is False
            assert res.content[0].text == f"ran:{xss_cmd}"
            assert provider.calls == [{"cmd": xss_cmd}]

            # Ciphertext purged in DB and status transitioned to consumed
            async with db_pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status, args_ciphertext, args_nonce FROM pending_approvals WHERE id = %s",
                    (approval_id,),
                )
                after_row = await cur.fetchone()
            assert after_row is not None
            assert after_row["status"] == "consumed"
            assert after_row["args_ciphertext"] is None
            assert after_row["args_nonce"] is None

            # 2. Deny within wait window -> returns deny error immediately and purges ciphertext
            deny_cmd = "rm -rf /tmp/never-run-this"
            deny_task = asyncio.create_task(sess.call_tool("demo_run", {"cmd": deny_cmd}))

            deny_id = None
            for _ in range(50):
                async with db_pool.connection() as conn:
                    cur = await conn.execute(
                        "SELECT id FROM pending_approvals WHERE status = 'pending'"
                    )
                    row = await cur.fetchone()
                if row is not None:
                    deny_id = str(row["id"])
                    break
                await asyncio.sleep(0.05)
            assert deny_id is not None

            r_deny = await web.post(
                f"/admin/approvals/{deny_id}/deny",
                data={"csrf_token": csrf},
            )
            assert r_deny.status_code == 200
            assert "Denied" in r_deny.text

            deny_res = await asyncio.wait_for(deny_task, timeout=5.0)
            assert deny_res.is_error is True
            assert "denied" in deny_res.content[0].text.lower()
            assert len(provider.calls) == 1  # Not executed!

            async with db_pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status, args_ciphertext, args_nonce FROM pending_approvals WHERE id = %s",
                    (deny_id,),
                )
                deny_row = await cur.fetchone()
            assert deny_row is not None
            assert deny_row["status"] == "denied"
            assert deny_row["args_ciphertext"] is None
            assert deny_row["args_nonce"] is None

    # 3. Audit trail check: approval.requested & approval.decided (outcome + latency), ZERO raw args
    async with db_pool.connection() as conn:
        cur = await conn.execute(
            "SELECT type, tool, decision, outcome, args_digest, payload "
            "FROM events WHERE type IN ('approval.requested', 'approval.decided') ORDER BY id ASC"
        )
        ev_rows = await cur.fetchall()

    assert len(ev_rows) == 4
    types_seq = [r["type"] for r in ev_rows]
    assert types_seq == [
        "approval.requested",
        "approval.decided",
        "approval.requested",
        "approval.decided",
    ]
    for r in ev_rows:
        assert r["args_digest"] is not None
        payload_str = str(r["payload"])
        assert "uname" not in payload_str
        assert "alert(1)" not in payload_str
        assert "never-run-this" not in payload_str

    decided_ok = ev_rows[1]
    assert decided_ok["outcome"] == "ok"
    assert decided_ok["payload"]["outcome"] == "approved"
    assert isinstance(decided_ok["payload"]["latency_ms"], int)
    assert decided_ok["payload"]["latency_ms"] >= 0

    decided_deny = ev_rows[3]
    assert decided_deny["outcome"] == "denied"
    assert decided_deny["payload"]["outcome"] == "denied"
    assert isinstance(decided_deny["payload"]["latency_ms"], int)


async def test_m9c_timeout_link_retry_single_use_isolation_and_expiry(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]
    client_a = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="chatgpt-phone")
    client_b = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="other-client")

    owner_pw = "owner-m9c-timeout"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    provider = FakeExternalShellProvider()
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=base_url,
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash(owner_pw),
        approval_wait_seconds=0.1,
    )
    app = create_app(cfg, extra_providers=[provider], run_migrations_on_startup=False)

    async with run_asgi_server(app, port=port):
        http_a = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_a['token']}"})
        http_b = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_b['token']}"})
        async with (
            http_a,
            http_b,
            httpx.AsyncClient(base_url=base_url, follow_redirects=False) as web,
            streamable_http_client(f"{base_url}/mcp", http_client=http_a) as (ra, wa),
            ClientSession(ra, wa, elicitation_callback=None) as sess_a,
            streamable_http_client(f"{base_url}/mcp", http_client=http_b) as (rb, wb),
            ClientSession(rb, wb, elicitation_callback=None) as sess_b,
        ):
            await sess_a.initialize()
            await sess_b.initialize()

            # 1. Timeout returns tool error containing clickable <ZOS_BASE_URL>/admin/approvals/<id>
            t_res = await sess_a.call_tool("demo_run", {"cmd": "uptime"})
            assert t_res.is_error is True
            aid = _extract_approval_id(t_res.content[0].text, base_url)
            assert provider.calls == []

            # Phone flow: unauthenticated GET /admin/approvals/<id> -> login?next=... -> lands on approval page!
            r_unauth = await web.get(f"/admin/approvals/{aid}")
            assert r_unauth.status_code == 303
            loc = r_unauth.headers["location"]
            assert loc == f"/admin/login?next=/admin/approvals/{aid}"

            r_login_get = await web.get(loc)
            assert r_login_get.status_code == 200
            assert f'value="/admin/approvals/{aid}"' in r_login_get.text

            r_login_post = await web.post(
                "/admin/login",
                data={"passphrase": owner_pw, "next": f"/admin/approvals/{aid}"},
            )
            assert r_login_post.status_code == 303
            assert urlparse(r_login_post.headers["location"]).path == f"/admin/approvals/{aid}"

            # Approve out-of-band
            r_det = await web.get(f"/admin/approvals/{aid}")
            assert r_det.status_code == 200
            assert "uptime" in r_det.text
            csrf = _extract_csrf(r_det.text)

            r_app = await web.post(f"/admin/approvals/{aid}/approve", data={"csrf_token": csrf})
            assert r_app.status_code == 200

            # Ciphertext purged immediately on approve even before retry
            async with db_pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status, args_ciphertext, args_nonce FROM pending_approvals WHERE id = %s",
                    (aid,),
                )
                st_row = await cur.fetchone()
            assert st_row is not None
            assert st_row["status"] == "approved"
            assert st_row["args_ciphertext"] is None
            assert st_row["args_nonce"] is None

            # 2. Wrong client CANNOT consume client_a's approval
            wrong_client_res = await sess_b.call_tool("demo_run", {"cmd": "uptime"})
            assert wrong_client_res.is_error is True
            wrong_client_aid = _extract_approval_id(wrong_client_res.content[0].text, base_url)
            assert wrong_client_aid != aid
            assert provider.calls == []

            # 3. Wrong args digest CANNOT consume client_a's approval
            wrong_args_res = await sess_a.call_tool("demo_run", {"cmd": "whoami"})
            assert wrong_args_res.is_error is True
            wrong_args_aid = _extract_approval_id(wrong_args_res.content[0].text, base_url)
            assert wrong_args_aid != aid
            assert provider.calls == []

            # 4. Exact retry by client_a consumes approval ONCE and executes!
            retry_ok = await sess_a.call_tool("demo_run", {"cmd": "uptime"})
            assert retry_ok.is_error is False
            assert retry_ok.content[0].text == "ran:uptime"
            assert provider.calls == [{"cmd": "uptime"}]

            # 5. Second retry of same call is re-gated (single-use atomic consumption)!
            retry_again = await sess_a.call_tool("demo_run", {"cmd": "uptime"})
            assert retry_again.is_error is True
            second_aid = _extract_approval_id(retry_again.content[0].text, base_url)
            assert second_aid != aid
            assert len(provider.calls) == 1

            # 6. Expired pending approval cannot be approved and sweep purges its ciphertext
            async with db_pool.connection() as conn:
                await conn.execute(
                    "UPDATE pending_approvals SET expires_at = now() - interval '1 minute' WHERE id = %s",
                    (second_aid,),
                )
                await conn.commit()

            r_exp = await web.post(f"/admin/approvals/{second_aid}/approve", data={"csrf_token": csrf})
            assert r_exp.status_code == 200
            assert "expired" in r_exp.text.lower()

            async with db_pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT status, args_ciphertext, args_nonce FROM pending_approvals WHERE id = %s",
                    (second_aid,),
                )
                exp_row = await cur.fetchone()
            assert exp_row is not None
            assert exp_row["status"] == "expired"
            assert exp_row["args_ciphertext"] is None
            assert exp_row["args_nonce"] is None

            # 7. Approved-but-expired approval CANNOT be consumed on retry!
            r_app_wrong = await web.post(
                f"/admin/approvals/{wrong_args_aid}/approve",
                data={"csrf_token": csrf},
            )
            assert r_app_wrong.status_code == 200
            async with db_pool.connection() as conn:
                await conn.execute(
                    "UPDATE pending_approvals SET expires_at = now() - interval '1 minute' WHERE id = %s",
                    (wrong_args_aid,),
                )
                await conn.commit()

            exp_retry = await sess_a.call_tool("demo_run", {"cmd": "whoami"})
            assert exp_retry.is_error is True
            new_aid = _extract_approval_id(exp_retry.content[0].text, base_url)
            assert new_aid != wrong_args_aid
            assert len(provider.calls) == 1


async def test_m9c_elicitation_client_still_uses_elicitation_path(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]
    client = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="claude-code")

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    provider = FakeExternalShellProvider()
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=base_url,
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=PasswordHasher().hash("owner-pw"),
        approval_wait_seconds=5.0,
    )
    app = create_app(cfg, extra_providers=[provider], run_migrations_on_startup=False)

    elicited: list[str] = []

    async def approve_cb(context: Any, params: types.ElicitRequestParams):
        elicited.append(params.message)
        return types.ElicitResult(action="accept", content={"approve": True})

    async with run_asgi_server(app, port=port):
        http_mcp = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client['token']}"})
        async with (
            http_mcp,
            streamable_http_client(f"{base_url}/mcp", http_client=http_mcp) as (r, w),
            ClientSession(r, w, elicitation_callback=approve_cb) as sess,
        ):
            await sess.initialize()
            res = await sess.call_tool("demo_run", {"cmd": "pwd"})
            assert res.is_error is False
            assert res.content[0].text == "ran:pwd"

    assert len(elicited) == 1
    assert "demo_run" in elicited[0]
    assert provider.calls == [{"cmd": "pwd"}]

    # No pending_approvals row created for elicitation-capable client
    async with db_pool.connection() as conn:
        cur = await conn.execute("SELECT COUNT(*) AS cnt FROM pending_approvals")
        row = await cur.fetchone()
    assert row is not None and int(row["cnt"]) == 0
