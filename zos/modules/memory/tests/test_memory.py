"""Tests for standalone ZOS Memory module (MCP tools, FTS+trigram, pagination, UI + CSRF, import)."""

from __future__ import annotations

import asyncio
import io
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

from zos_memory.caller import (
    HTTP_CALLER_HEADER,
    HTTP_CALLER_SIGNATURE_HEADER,
    MCP_CALLER_META_KEY,
    sign_http_caller,
    sign_mcp_caller,
)
from zos_memory.import_cli import import_jsonl_stream
from zos_memory.mcp_server import create_mcp_app
from zos_memory.store import MemoryStore, create_memory_pool
from zos_memory.ui import create_ui_app

DEFAULT_TEST_DB = "postgresql://zos:zos@127.0.0.1:55432/zos_test"
SECRET = "memory-test-secret-0123456789"


def _db_url() -> str:
    return os.environ.get("ZOS_TEST_DATABASE_URL", DEFAULT_TEST_DB)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
async def memory_store() -> AsyncIterator[MemoryStore]:
    """Provision a clean temporary schema with pg_trgm access so tests never touch public."""
    schema = f"test_mem_{uuid.uuid4().hex[:10]}"
    base = _db_url()
    async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS extensions")
        cur = await conn.execute(
            """
            SELECT n.nspname
            FROM pg_extension e
            JOIN pg_namespace n ON n.oid = e.extnamespace
            WHERE e.extname = 'pg_trgm'
            """
        )
        row = await cur.fetchone()
        if row is None:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA extensions")
            ext_schema = "extensions"
        else:
            ext_schema = str(row[0])
        await conn.execute(f"CREATE SCHEMA {schema}")

    sep = "&" if "?" in base else "?"
    schema_url = f"{base}{sep}options=-csearch_path%3D{schema}%2C{ext_schema}"
    pool = create_memory_pool(schema_url)
    await pool.open()
    store = MemoryStore(pool)
    await store.init_schema()
    try:
        yield store
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_memory_has_zero_zos_imports() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src" / "zos_memory"
    for py_file in src_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "import zos\n" not in text
        assert "from zos." not in text
        assert "from zos " not in text


async def test_memory_mcp_put_get_search_list_delete_and_auth(memory_store: MemoryStore) -> None:
    mcp_app = create_mcp_app(memory_store, module_secret=SECRET, mcp_path="/mcp")
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
            assert set(by_name) == {"put", "get", "search", "list", "delete"}
            assert by_name["delete"].annotations is not None
            assert by_name["delete"].annotations.destructive_hint is True
            assert by_name["put"].annotations is not None
            assert by_name["put"].annotations.idempotent_hint is True

            # 1. Unsigned (system) calls MUST be refused
            for tool_name, args in (
                ("put", {"key": "k", "content": "c"}),
                ("get", {"key": "k"}),
                ("search", {"query": "k"}),
                ("list", {}),
                ("delete", {"key": "k"}),
            ):
                r_un = await session.call_tool(tool_name, args)
                assert r_un.is_error
                assert "CALLER_REQUIRED" in r_un.content[0].text

            # 2. Put + Get + Author attribution + Upsert preserving original author
            put_args = {
                "key": "pref/coffee",
                "content": "Prefers pour-over Ethiopian coffee in the morning",
                "tags": ["prefs", "food"],
            }
            r_put = await session.call_tool("put", put_args, meta=meta_for("claude", "put", put_args))
            assert not r_put.is_error
            put_data = json.loads(r_put.content[0].text)
            assert put_data["key"] == "pref/coffee"
            assert put_data["tags"] == ["food", "prefs"]
            assert put_data["author"] == "claude"

            get_args = {"key": "pref/coffee"}
            r_get = await session.call_tool("get", get_args, meta=meta_for("codex", "get", get_args))
            assert not r_get.is_error
            got = json.loads(r_get.content[0].text)
            assert got["found"] is True
            assert got["memory"]["content"] == "Prefers pour-over Ethiopian coffee in the morning"
            assert got["memory"]["author"] == "claude"

            # Missing key returns found=False
            r_miss = await session.call_tool(
                "get", {"key": "nope"}, meta=meta_for("codex", "get", {"key": "nope"})
            )
            assert json.loads(r_miss.content[0].text) == {"found": False, "key": "nope"}

            # Put second memory with different author + tags
            put2 = {
                "key": "project/zos",
                "content": "ZOS is a personal MCP gateway backed by PostgreSQL",
                "tags": ["work", "zos"],
            }
            await session.call_tool("put", put2, meta=meta_for("codex", "put", put2))

            # 3. FTS search + tag filter
            fts_args = {"query": "Ethiopian coffee", "tags": ["prefs"]}
            r_fts = await session.call_tool("search", fts_args, meta=meta_for("claude", "search", fts_args))
            fts_data = json.loads(r_fts.content[0].text)
            assert fts_data["mode"] == "fts"
            assert fts_data["count"] == 1
            assert fts_data["matches"][0]["key"] == "pref/coffee"
            assert "score" in fts_data["matches"][0]

            # 4. Trigram fallback for typo
            typo_args = {"query": "Ethiopan"}
            r_typo = await session.call_tool("search", typo_args, meta=meta_for("claude", "search", typo_args))
            typo_data = json.loads(r_typo.content[0].text)
            assert typo_data["mode"] == "trigram"
            assert typo_data["count"] == 1
            assert typo_data["matches"][0]["key"] == "pref/coffee"

            # 5. Empty query search returns recent
            rec_args = {"tags": ["work"]}
            r_rec = await session.call_tool("search", rec_args, meta=meta_for("claude", "search", rec_args))
            rec_data = json.loads(r_rec.content[0].text)
            assert rec_data["mode"] == "recent"
            assert [m["key"] for m in rec_data["matches"]] == ["project/zos"]

            # 6. memory_list returns summaries WITHOUT content, ordered updated_at desc, paginated
            p1_args = {"limit": 1}
            r_p1 = await session.call_tool("list", p1_args, meta=meta_for("claude", "list", p1_args))
            p1 = json.loads(r_p1.content[0].text)
            assert p1["count"] == 1
            assert "content" not in p1["items"][0]
            assert p1["items"][0]["key"] == "project/zos"
            assert p1["items"][0]["author"] == "codex"
            assert p1["next_cursor"] is not None

            p2_args = {"limit": 1, "cursor": p1["next_cursor"]}
            r_p2 = await session.call_tool("list", p2_args, meta=meta_for("claude", "list", p2_args))
            p2 = json.loads(r_p2.content[0].text)
            assert p2["count"] == 1
            assert p2["items"][0]["key"] == "pref/coffee"
            assert p2["next_cursor"] is None

            # Tag filter on list
            lt_args = {"tag": "zos"}
            r_lt = await session.call_tool("list", lt_args, meta=meta_for("claude", "list", lt_args))
            lt = json.loads(r_lt.content[0].text)
            assert [i["key"] for i in lt["items"]] == ["project/zos"]

            # Invalid cursor -> error
            bad_cur = {"cursor": "not-valid-base64"}
            r_bc = await session.call_tool("list", bad_cur, meta=meta_for("claude", "list", bad_cur))
            assert r_bc.is_error

            # 7. Delete
            del_args = {"key": "pref/coffee"}
            r_del = await session.call_tool("delete", del_args, meta=meta_for("claude", "delete", del_args))
            assert json.loads(r_del.content[0].text) == {"deleted": True, "key": "pref/coffee"}

            r_del2 = await session.call_tool("delete", del_args, meta=meta_for("claude", "delete", del_args))
            assert json.loads(r_del2.content[0].text) == {"deleted": False, "key": "pref/coffee"}
    finally:
        srv.should_exit = True
        await task


async def test_memory_ui_browse_search_xss_and_csrf_delete(memory_store: MemoryStore) -> None:
    ui_app = create_ui_app(memory_store, module_secret=SECRET)
    await memory_store.put_memory(
        key="note/xss",
        content="<script>alert(1)</script> Hello memory world",
        tags=["personal", "test"],
        client_id="cid-1",
        client_name="claude-desktop",
    )
    await memory_store.put_memory(
        key="work/plan",
        content="Ship M12a cleanly",
        tags=["work"],
        client_id="cid-2",
        client_name="codex",
    )

    transport = httpx.ASGITransport(app=ui_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://memory.local") as client:
        # Unsigned request refused 403
        assert (await client.get("/")).status_code == 403

        hdr, sig = sign_http_caller(SECRET, principal_id="p-owner", client_id="owner", client_name="owner")
        auth_headers = {
            HTTP_CALLER_HEADER: hdr,
            HTTP_CALLER_SIGNATURE_HEADER: sig,
            "X-Forwarded-Prefix": "/admin/m/memory",
        }

        # Index lists all memories with escaped content, author, and tag pills
        r_idx = await client.get("/", headers=auth_headers)
        assert r_idx.status_code == 200
        assert "<script>alert(1)</script>" not in r_idx.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in r_idx.text
        assert "claude-desktop" in r_idx.text
        assert "codex" in r_idx.text
        assert "2 memories" in r_idx.text

        # Search query and tag filter
        r_q = await client.get("/?q=M12a", headers=auth_headers)
        assert r_q.status_code == 200
        assert "work/plan" in r_q.text
        assert "note/xss" not in r_q.text

        r_tag = await client.get("/?tag=personal", headers=auth_headers)
        assert r_tag.status_code == 200
        assert "note/xss" in r_tag.text
        assert "work/plan" not in r_tag.text

        # Detail view shows full content, author, Delete button (no edit form/textarea)
        m_link = re.search(r'href="/admin/m/memory(/item/[^"]+)"', r_tag.text)
        assert m_link is not None
        r_det = await client.get(m_link.group(1), headers=auth_headers)
        assert r_det.status_code == 200
        assert "<textarea" not in r_det.text
        assert "Delete memory" in r_det.text
        assert "claude-desktop" in r_det.text
        assert "<script>alert(1)</script>" not in r_det.text

        set_cookie = r_det.headers.get("set-cookie", "")
        assert "memory_csrf=" in set_cookie and "Path=/admin/m/memory" in set_cookie
        m_csrf = re.search(r'name="csrf_token"\s+value="([^"]+)"', r_det.text)
        assert m_csrf is not None
        csrf = m_csrf.group(1)
        cookie_headers = {**auth_headers, "Cookie": f"memory_csrf={csrf}"}

        # POST /delete with bad CSRF -> 403
        r_bad = await client.post(
            "/delete",
            headers=cookie_headers,
            data={"csrf_token": "wrong", "key": "note/xss"},
            follow_redirects=False,
        )
        assert r_bad.status_code == 403
        assert await memory_store.get_memory(key="note/xss") is not None

        # POST /delete with valid CSRF deletes memory and redirects under prefix
        r_ok = await client.post(
            "/delete",
            headers=cookie_headers,
            data={"csrf_token": csrf, "key": "note/xss"},
            follow_redirects=False,
        )
        assert r_ok.status_code == 303
        assert r_ok.headers["location"].startswith("/admin/m/memory/")
        assert await memory_store.get_memory(key="note/xss") is None


async def test_memory_import_jsonl_idempotent_and_migrations_versioned(
    memory_store: MemoryStore,
) -> None:
    lines = "\n".join(
        [
            json.dumps({"key": "imp/1", "content": "First note", "tags": ["claude"]}),
            json.dumps({"key": "imp/2", "content": "Second note", "tags": ["chatgpt"]}),
        ]
    )
    s1 = await import_jsonl_stream(memory_store, io.StringIO(lines), source_name="cli-import")
    assert s1 == {"discovered": 2, "created": 2, "updated": 0, "unchanged": 0}

    s2 = await import_jsonl_stream(memory_store, io.StringIO(lines), source_name="cli-import")
    assert s2 == {"discovered": 2, "created": 0, "updated": 0, "unchanged": 2}

    lines_updated = json.dumps({"key": "imp/1", "content": "First note v2", "tags": ["claude"]})
    s3 = await import_jsonl_stream(memory_store, io.StringIO(lines_updated), source_name="cli-import")
    assert s3 == {"discovered": 1, "created": 0, "updated": 1, "unchanged": 0}

    # Migrations are idempotent and recorded in memory_schema_migrations
    assert await memory_store.init_schema() == []
    async with memory_store.pool.connection() as conn:
        rows = await (await conn.execute("SELECT version FROM memory_schema_migrations")).fetchall()
    assert [r["version"] for r in rows] == ["0001_init"]
