"""Tests for standalone ZOS Tasks module (store, MCP tools, caller auth, UI + CSRF, migrations)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import mcp_types as types
import psycopg
import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from zos_tasks.caller import (
    HTTP_CALLER_HEADER,
    HTTP_CALLER_SIGNATURE_HEADER,
    MCP_CALLER_META_KEY,
    sign_http_caller,
    sign_mcp_caller,
)
from zos_tasks.mcp_server import create_mcp_app
from zos_tasks.store import TasksStore, create_tasks_pool
from zos_tasks.ui import create_ui_app

DEFAULT_TEST_DB = "postgresql://zos:zos@127.0.0.1:55432/zos_test"
SECRET = "tasks-test-secret-0123456789"


def _db_url() -> str:
    return os.environ.get("ZOS_TEST_DATABASE_URL", DEFAULT_TEST_DB)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
async def tasks_store() -> AsyncIterator[TasksStore]:
    """Provision an isolated temporary Postgres schema so tests never touch public."""
    schema = f"test_tasks_{uuid.uuid4().hex[:10]}"
    base = _db_url()
    async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")

    sep = "&" if "?" in base else "?"
    schema_url = f"{base}{sep}options=-csearch_path%3D{schema}"
    pool = create_tasks_pool(schema_url)
    await pool.open()
    store = TasksStore(pool)
    await store.init_schema()
    try:
        yield store
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_tasks_has_zero_zos_imports() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src" / "zos_tasks"
    for py_file in src_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "import zos\n" not in text
        assert "from zos." not in text
        assert "from zos " not in text


async def test_tasks_store_crud_filters_cursor_and_migrations(tasks_store: TasksStore) -> None:
    # Migrations are idempotent
    assert await tasks_store.init_schema() == []

    t1 = await tasks_store.add_task(
        title="Renew passport",
        created_by="Claude",
        notes="Check photo requirements",
        due="2026-12-01",
        tags=["Travel", "admin", "travel"],
    )
    assert t1["status"] == "open"
    assert t1["due"] == "2026-12-01"
    assert t1["tags"] == ["admin", "travel"]
    assert t1["created_by"] == "Claude"
    assert t1["completed_at"] is None

    t2 = await tasks_store.add_task(
        title="Buy coffee beans",
        created_by="owner",
        due="2026-11-28",
        tags=["errands"],
    )
    t3 = await tasks_store.add_task(
        title="Read architecture doc",
        created_by="Codex",
        tags=["reading"],
    )

    # Get by id
    fetched = await tasks_store.get_task(t1["id"])
    assert fetched["title"] == "Renew passport"

    # Filter by tag
    by_tag = await tasks_store.list_tasks(tag="travel")
    assert [t["id"] for t in by_tag["tasks"]] == [t1["id"]]

    # Filter by due_before
    by_due = await tasks_store.list_tasks(due_before="2026-11-30")
    assert [t["id"] for t in by_due["tasks"]] == [t2["id"]]

    # Cursor pagination (limit=2 over 3 open tasks)
    page1 = await tasks_store.list_tasks(status="open", limit=2)
    assert [t["id"] for t in page1["tasks"]] == [t3["id"], t2["id"]]
    assert page1["next_cursor"] == str(t2["id"])

    page2 = await tasks_store.list_tasks(status="open", limit=2, cursor=page1["next_cursor"])
    assert [t["id"] for t in page2["tasks"]] == [t1["id"]]
    assert page2["next_cursor"] is None

    # Update & clear due date
    updated = await tasks_store.update_task(t2["id"], title="Buy espresso beans", due=None, tags=["coffee"])
    assert updated["title"] == "Buy espresso beans"
    assert updated["due"] is None
    assert updated["tags"] == ["coffee"]

    # Complete & reopen
    done = await tasks_store.complete_task(t1["id"])
    assert done["status"] == "done"
    assert done["completed_at"] is not None

    done_list = await tasks_store.list_tasks(status="done")
    assert [t["id"] for t in done_list["tasks"]] == [t1["id"]]

    reopened = await tasks_store.reopen_task(t1["id"])
    assert reopened["status"] == "open"
    assert reopened["completed_at"] is None

    # Delete (store/UI only)
    assert await tasks_store.delete_task(t3["id"]) is True
    with pytest.raises(KeyError):
        await tasks_store.get_task(t3["id"])


async def test_tasks_mcp_tools_descriptions_caller_auth_and_no_delete(tasks_store: TasksStore) -> None:
    mcp_app = create_mcp_app(tasks_store, module_secret=SECRET, mcp_path="/mcp")
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
            by_name = {t.name: t for t in listed.tools}
            # Exactly 5 tools — no delete over MCP!
            assert set(by_name) == {"list", "get", "add", "update", "complete"}
            assert "delete" not in by_name

            # Required "for the human owner" descriptions
            assert "for the human owner" in (by_name["add"].description or "")
            assert "Never use this to track your own agent work" in (by_name["add"].description or "")
            assert "human" in (by_name["list"].description or "").lower()

            # Unsigned call refused
            r_un = await session.call_tool("add", {"title": "Unauthorized"})
            assert r_un.is_error
            assert "CALLER_REQUIRED" in r_un.content[0].text

            # Reserved 'owner' caller refused over MCP
            r_res = await session.call_tool(
                "add",
                {"title": "Spoofed"},
                meta=meta_for("owner", "add", {"title": "Spoofed"}),
            )
            assert r_res.is_error
            assert "CALLER_RESERVED" in r_res.content[0].text

            # Signed MCP add records created_by = caller.client_name
            add_args = {"title": "Call dentist", "due": "2026-12-05", "tags": ["health"]}
            r_add = await session.call_tool("add", add_args, meta=meta_for("Claude", "add", add_args))
            assert not r_add.is_error
            created = json.loads(r_add.content[0].text)["task"]
            assert created["title"] == "Call dentist"
            assert created["created_by"] == "Claude"
            tid = created["id"]

            # Get, list, update, complete
            r_get = await session.call_tool("get", {"id": tid}, meta=meta_for("Claude", "get", {"id": tid}))
            assert not r_get.is_error
            assert json.loads(r_get.content[0].text)["task"]["id"] == tid

            upd_args = {"id": tid, "notes": "Ask about cleaning"}
            r_upd = await session.call_tool("update", upd_args, meta=meta_for("Codex", "update", upd_args))
            assert not r_upd.is_error
            assert json.loads(r_upd.content[0].text)["task"]["notes"] == "Ask about cleaning"

            r_comp = await session.call_tool("complete", {"id": tid}, meta=meta_for("Claude", "complete", {"id": tid}))
            assert not r_comp.is_error
            assert json.loads(r_comp.content[0].text)["task"]["status"] == "done"

            r_list = await session.call_tool(
                "list",
                {"status": "done"},
                meta=meta_for("Claude", "list", {"status": "done"}),
            )
            assert not r_list.is_error
            assert [t["id"] for t in json.loads(r_list.content[0].text)["tasks"]] == [tid]
    finally:
        srv.should_exit = True
        await task


async def test_tasks_ui_caller_signature_grouping_xss_and_crud(tasks_store: TasksStore) -> None:
    ui_app = create_ui_app(tasks_store, module_secret=SECRET)
    today = datetime.now(UTC).date()
    yesterday = (today - timedelta(days=1)).isoformat()
    tomorrow = (today + timedelta(days=2)).isoformat()

    overdue_t = await tasks_store.add_task(
        title="<script>alert(1)</script> Pay utility bill",
        created_by="Claude",
        due=yesterday,
        tags=["bills"],
    )
    await tasks_store.add_task(title="Pick up dry cleaning", created_by="owner", due=today.isoformat())
    await tasks_store.add_task(title="Book flights", created_by="Codex", due=tomorrow)
    await tasks_store.add_task(title="Organize desk", created_by="owner")

    transport = httpx.ASGITransport(app=ui_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tasks.local") as client:
        # 1. Unsigned UI request rejected with 403
        r_un = await client.get("/")
        assert r_un.status_code == 403

        # 2. Tampered signature rejected with 403
        hdr, sig = sign_http_caller(SECRET, principal_id="p-owner", client_id="owner", client_name="owner")
        r_bad = await client.get(
            "/",
            headers={HTTP_CALLER_HEADER: hdr, HTTP_CALLER_SIGNATURE_HEADER: "sha256=deadbeef"},
        )
        assert r_bad.status_code == 403

        auth_headers = {
            HTTP_CALLER_HEADER: hdr,
            HTTP_CALLER_SIGNATURE_HEADER: sig,
            "X-Forwarded-Prefix": "/admin/m/tasks",
        }

        # 3. Signed GET renders all 4 groups, escapes XSS, shows 'added by Claude'
        r_idx = await client.get("/", headers=auth_headers)
        assert r_idx.status_code == 200
        for heading in ("Overdue", "Today", "Upcoming", "No date"):
            assert heading in r_idx.text
        assert "<script>alert(1)</script>" not in r_idx.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in r_idx.text
        assert "added by Claude" in r_idx.text
        assert "added by Codex" in r_idx.text

        set_cookie = r_idx.headers.get("set-cookie", "")
        assert "tasks_csrf=" in set_cookie and "Path=/admin/m/tasks" in set_cookie
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', r_idx.text)
        assert m is not None
        csrf = m.group(1)
        cookie_headers = {**auth_headers, "Cookie": f"tasks_csrf={csrf}"}

        # 4. POST without valid CSRF fails 403
        r_no_csrf = await client.post(
            "/add",
            headers=cookie_headers,
            data={"csrf_token": "bad", "title": "Nope"},
            follow_redirects=False,
        )
        assert r_no_csrf.status_code == 403

        # 5. Add, complete, edit, and delete via UI
        r_add = await client.post(
            "/add",
            headers=cookie_headers,
            data={"csrf_token": csrf, "title": "Water plants", "tags": "home"},
            follow_redirects=False,
        )
        assert r_add.status_code == 303
        assert r_add.headers["location"].startswith("/admin/m/tasks/")

        r_comp = await client.post(
            "/complete",
            headers=cookie_headers,
            data={"csrf_token": csrf, "id": str(overdue_t["id"])},
            follow_redirects=False,
        )
        assert r_comp.status_code == 303

        r_after_comp = await client.get("/", headers=auth_headers)
        assert "Done" in r_after_comp.text
        assert "Water plants" in r_after_comp.text

        r_del = await client.post(
            "/delete",
            headers=cookie_headers,
            data={"csrf_token": csrf, "id": str(overdue_t["id"])},
            follow_redirects=False,
        )
        assert r_del.status_code == 303
        with pytest.raises(KeyError):
            await tasks_store.get_task(overdue_t["id"])
