"""Tests for standalone ZOS Inbox module (MCP tools, caller verification, isolation, UI + CSRF)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx
import mcp_types as types
import psycopg
import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from zos_inbox.caller import (
    HTTP_CALLER_HEADER,
    HTTP_CALLER_SIGNATURE_HEADER,
    MCP_CALLER_META_KEY,
    sign_http_caller,
    sign_mcp_caller,
)
from zos_inbox.mcp_server import create_mcp_app
from zos_inbox.store import InboxStore, create_inbox_pool
from zos_inbox.ui import create_ui_app

DEFAULT_TEST_DB = "postgresql://zos:zos@127.0.0.1:55432/zos_test"
SECRET = "inbox-test-secret-0123456789"


def _db_url() -> str:
    return os.environ.get("ZOS_TEST_DATABASE_URL", DEFAULT_TEST_DB)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
async def inbox_store() -> AsyncIterator[InboxStore]:
    """Provision a clean temporary schema in test Postgres so tests never touch public."""
    schema = f"test_inbox_{uuid.uuid4().hex[:10]}"
    base = _db_url()
    async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")

    sep = "&" if "?" in base else "?"
    schema_url = f"{base}{sep}options=-csearch_path%3D{schema}"
    pool = create_inbox_pool(schema_url)
    await pool.open()
    store = InboxStore(pool)
    await store.init_schema()
    try:
        yield store
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_inbox_has_zero_zos_imports() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src" / "zos_inbox"
    for py_file in src_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "import zos\n" not in text
        assert "from zos." not in text
        assert "from zos " not in text


async def test_inbox_mcp_tools_auth_privacy_and_general_read_state(inbox_store: InboxStore) -> None:
    mcp_app = create_mcp_app(inbox_store, module_secret=SECRET, mcp_path="/mcp")
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(mcp_app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    base = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient() as hc:
            for _ in range(50):
                try:
                    if (await hc.get(f"{base}/healthz")).status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.05)

        def meta_for(client_name: str, tool: str, args: dict[str, Any]) -> types.RequestParamsMeta:
            env = sign_mcp_caller(
                SECRET,
                principal_id="p-owner",
                client_id=f"cid-{client_name}",
                client_name=client_name,
                tool=tool,
                args=args,
            )
            return cast(types.RequestParamsMeta, {MCP_CALLER_META_KEY: env})

        async with (
            streamable_http_client(f"{base}/mcp") as (read_s, write_s),
            ClientSession(read_s, write_s) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            assert {t.name for t in listed.tools} == {"send", "list", "read", "archive", "directory"}

            # 1. Unsigned (system) calls to send/list/read/archive MUST be refused
            for tool_name, args in (
                ("send", {"to": ["general"], "subject": "hi", "body": "no"}),
                ("list", {}),
                ("read", {"id": 1}),
                ("archive", {"id": 1}),
            ):
                r_un = await session.call_tool(tool_name, args)
                assert r_un.is_error
                assert "CALLER_REQUIRED" in r_un.content[0].text

            # 2. Client A sends to B and to general (from is strictly 'alpha')
            send_ab_args = {"to": ["beta"], "subject": "Direct note", "body": "Hi beta from alpha"}
            r_ab = await session.call_tool("send", send_ab_args, meta=meta_for("alpha", "send", send_ab_args))
            assert not r_ab.is_error
            msg_ab = json.loads(r_ab.content[0].text)["message"]
            assert msg_ab["from"] == "alpha"
            assert msg_ab["to"] == ["beta"]

            send_gen_args = {"to": ["general"], "subject": "Announcement", "body": "Hello all"}
            r_gen = await session.call_tool("send", send_gen_args, meta=meta_for("alpha", "send", send_gen_args))
            assert not r_gen.is_error
            msg_gen = json.loads(r_gen.content[0].text)["message"]
            assert msg_gen["from"] == "alpha"

            # Private message to alpha only (sent by owner via UI/store, not MCP)
            msg_pa = await inbox_store.send_message(
                sender="owner",
                to=["alpha"],
                subject="Private to alpha",
                body="Top secret for alpha",
            )

            # Reserved MCP caller names ('owner', 'general', any case/whitespace) MUST be refused
            for reserved_name in ("owner", "Owner ", "general", "GENERAL"):
                r_res_send = await session.call_tool(
                    "send",
                    {"to": ["alpha"], "subject": "Spoof", "body": "Nope"},
                    meta=meta_for(reserved_name, "send", {"to": ["alpha"], "subject": "Spoof", "body": "Nope"}),
                )
                assert r_res_send.is_error
                assert "CALLER_RESERVED" in r_res_send.content[0].text

                r_res_list = await session.call_tool(
                    "list",
                    {},
                    meta=meta_for(reserved_name, "list", {}),
                )
                assert r_res_list.is_error
                assert "CALLER_RESERVED" in r_res_list.content[0].text

            # 3. Beta lists its own mailbox -> sees Direct note
            r_blist = await session.call_tool("list", {}, meta=meta_for("beta", "list", {}))
            assert not r_blist.is_error
            blist = json.loads(r_blist.content[0].text)
            assert blist["mailbox"] == "beta"
            assert [m["id"] for m in blist["messages"]] == [msg_ab["id"]]

            # 4. Beta CANNOT list alpha's mailbox or read alpha's private message
            r_forbid_list = await session.call_tool(
                "list", {"mailbox": "alpha"}, meta=meta_for("beta", "list", {"mailbox": "alpha"})
            )
            assert r_forbid_list.is_error
            assert "PERMISSION_DENIED" in r_forbid_list.content[0].text

            r_forbid_read = await session.call_tool(
                "read", {"id": msg_pa["id"]}, meta=meta_for("beta", "read", {"id": msg_pa["id"]})
            )
            assert r_forbid_read.is_error
            assert "PERMISSION_DENIED" in r_forbid_read.content[0].text

            # 5. Beta reads direct note & general announcement -> marked read ONLY for beta
            r_read_ab = await session.call_tool(
                "read", {"id": msg_ab["id"]}, meta=meta_for("beta", "read", {"id": msg_ab["id"]})
            )
            assert not r_read_ab.is_error
            assert json.loads(r_read_ab.content[0].text)["message"]["read"] is True

            r_read_gen = await session.call_tool(
                "read", {"id": msg_gen["id"]}, meta=meta_for("beta", "read", {"id": msg_gen["id"]})
            )
            assert not r_read_gen.is_error

            # Beta's unread general list is now empty, while gamma still sees general message unread!
            r_bgen = await session.call_tool(
                "list", {"mailbox": "general", "unread_only": True},
                meta=meta_for("beta", "list", {"mailbox": "general", "unread_only": True}),
            )
            assert json.loads(r_bgen.content[0].text)["messages"] == []

            r_ggen = await session.call_tool(
                "list", {"mailbox": "general", "unread_only": True},
                meta=meta_for("gamma", "list", {"mailbox": "general", "unread_only": True}),
            )
            assert [m["id"] for m in json.loads(r_ggen.content[0].text)["messages"]] == [msg_gen["id"]]

            # 6. Archive hides message from subsequent list calls
            r_arch = await session.call_tool(
                "archive", {"id": msg_ab["id"]}, meta=meta_for("beta", "archive", {"id": msg_ab["id"]})
            )
            assert not r_arch.is_error
            r_blist_all = await session.call_tool(
                "list", {"unread_only": False}, meta=meta_for("beta", "list", {"unread_only": False})
            )
            assert json.loads(r_blist_all.content[0].text)["messages"] == []

            # 7. Directory shows known mailboxes and caller-scoped unread counts
            r_dir = await session.call_tool("directory", {}, meta=meta_for("gamma", "directory", {}))
            assert not r_dir.is_error
            dir_data = json.loads(r_dir.content[0].text)
            by_mb = {m["mailbox"]: m for m in dir_data["mailboxes"]}
            assert {"owner", "general", "alpha", "beta", "gamma"} <= set(by_mb)
            assert by_mb["general"]["unread_count"] == 1
    finally:
        srv.should_exit = True
        await task


async def test_inbox_ui_auth_xss_escaping_and_csrf(inbox_store: InboxStore) -> None:
    ui_app = create_ui_app(inbox_store, module_secret=SECRET)
    await inbox_store.send_message(
        sender="claude",
        to=["owner"],
        subject="<script>alert(1)</script> Status",
        body="Line 1\n<b>not bold</b>",
    )

    transport = httpx.ASGITransport(app=ui_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://inbox.local") as client:
        # 1. Unsigned request rejected with 403
        r_no = await client.get("/")
        assert r_no.status_code == 403

        hdr, sig = sign_http_caller(SECRET, principal_id="p-owner", client_id="owner", client_name="owner")
        auth_headers = {
            HTTP_CALLER_HEADER: hdr,
            HTTP_CALLER_SIGNATURE_HEADER: sig,
            "X-Forwarded-Prefix": "/admin/m/inbox",
        }

        # 2. Mailbox list renders with escaped HTML and unread count
        r_index = await client.get("/", headers=auth_headers)
        assert r_index.status_code == 200
        assert "<script>alert(1)</script>" not in r_index.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in r_index.text
        assert "/admin/m/inbox/compose" in r_index.text

        # 3. Compose page sets CSRF cookie scoped to X-Forwarded-Prefix; POST without CSRF fails 403
        r_comp = await client.get("/compose", headers=auth_headers)
        assert r_comp.status_code == 200
        set_cookie = r_comp.headers.get("set-cookie", "")
        assert "inbox_csrf=" in set_cookie and "Path=/admin/m/inbox" in set_cookie
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', r_comp.text)
        assert m is not None
        csrf = m.group(1)
        cookie_headers = {**auth_headers, "Cookie": f"inbox_csrf={csrf}"}

        r_bad_csrf = await client.post(
            "/send",
            headers=cookie_headers,
            data={"csrf_token": "wrong", "to": "general", "subject": "Hi", "body": "Test"},
            follow_redirects=False,
        )
        assert r_bad_csrf.status_code == 403

        # 4. Valid CSRF sends message as 'owner' and redirects under prefix
        r_send = await client.post(
            "/send",
            headers=cookie_headers,
            data={"csrf_token": csrf, "to": "claude, general", "subject": "From phone", "body": "Looks good!"},
            follow_redirects=False,
        )
        assert r_send.status_code == 303
        loc = r_send.headers["location"]
        assert loc.startswith("/admin/m/inbox/thread/")

        # 5. Thread view renders escaped content
        thread_path = loc[len("/admin/m/inbox") :]
        r_thread = await client.get(thread_path, headers=auth_headers)
        assert r_thread.status_code == 200
        assert "From phone" in r_thread.text
        assert "owner" in r_thread.text


async def test_owner_visibility_all_mailboxes_without_mutating_client_read_state(
    inbox_store: InboxStore,
) -> None:
    """Owner UI shows EVERY mailbox & client-to-client thread without altering clients' read state."""
    ui_app = create_ui_app(inbox_store, module_secret=SECRET)
    c2c = await inbox_store.send_message(
        sender="alpha",
        to=["beta"],
        subject="Private agent-to-agent handoff",
        body="Syncing notes between alpha and beta.",
    )

    transport = httpx.ASGITransport(app=ui_app)
    hdr, sig = sign_http_caller(SECRET, principal_id="p-owner", client_id="owner", client_name="owner")
    auth_headers = {
        HTTP_CALLER_HEADER: hdr,
        HTTP_CALLER_SIGNATURE_HEADER: sig,
        "X-Forwarded-Prefix": "/admin/m/inbox",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://inbox.local") as client:
        # Owner sees client-to-client message in All Mail AND in beta's mailbox
        r_all = await client.get("/all", headers=auth_headers)
        assert r_all.status_code == 200
        assert "Private agent-to-agent handoff" in r_all.text
        assert "alpha" in r_all.text and "beta" in r_all.text

        r_beta = await client.get("/mailbox/beta", headers=auth_headers)
        assert r_beta.status_code == 200
        assert "Private agent-to-agent handoff" in r_beta.text

        # Owner opens the client-to-client thread
        r_thread = await client.get(f"/thread/{c2c['thread_id']}?mb=beta", headers=auth_headers)
        assert r_thread.status_code == 200
        assert "Syncing notes between alpha and beta." in r_thread.text

    # Beta's unread state MUST remain untouched after owner viewed the thread!
    beta_unread = await inbox_store.list_messages(caller="beta", unread_only=True)
    assert [m["id"] for m in beta_unread["messages"]] == [c2c["id"]]
    assert beta_unread["messages"][0]["read"] is False
    dir_beta = {d["mailbox"]: d["unread_count"] for d in await inbox_store.directory(caller="beta")}
    assert dir_beta["beta"] == 1
