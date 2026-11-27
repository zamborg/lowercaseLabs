"""Tests for the standalone ZOS Files module (store, MCP tools, caller verification, UI + CSRF)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
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

from zos_files.caller import (
    HTTP_CALLER_HEADER,
    HTTP_CALLER_SIGNATURE_HEADER,
    MCP_CALLER_META_KEY,
    sign_http_caller,
    sign_mcp_caller,
)
from zos_files.mcp_server import create_mcp_app
from zos_files.store import (
    ConflictError,
    FilesStore,
    FileTooLargeError,
    PathError,
    QuotaExceededError,
    create_files_pool,
    decode_payload,
    normalize_path,
    normalize_prefix,
    sha256_hex,
)
from zos_files.ui import create_ui_app

DEFAULT_TEST_DB = "postgresql://zos:zos@127.0.0.1:55432/zos_test"
SECRET = "files-test-secret-0123456789"


def _db_url() -> str:
    return os.environ.get("ZOS_TEST_DATABASE_URL", DEFAULT_TEST_DB)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _make_store(quota_bytes: int = 1024 * 1024 * 1024) -> AsyncIterator[FilesStore]:
    schema = f"test_files_{uuid.uuid4().hex[:10]}"
    base = _db_url()
    async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in base else "?"
    pool = create_files_pool(f"{base}{sep}options=-csearch_path%3D{schema}")
    await pool.open()
    store = FilesStore(pool, quota_bytes=quota_bytes)
    await store.init_schema()
    try:
        yield store
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pytest.fixture
async def store() -> AsyncIterator[FilesStore]:
    async for s in _make_store():
        yield s


@pytest.fixture
async def small_store() -> AsyncIterator[FilesStore]:
    async for s in _make_store(quota_bytes=100):
        yield s


def test_files_has_zero_zos_imports() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src" / "zos_files"
    for py_file in src_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "import zos\n" not in text
        assert "from zos." not in text
        assert "from zos " not in text


# ── path normalization ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "/abs/path.md",
        "../escape.md",
        "a/../b.md",
        "a/./b.md",
        ".",
        "..",
        "a//b.md",
        "a/b/",
        "a\\b.md",
        "a/b\x00.md",
        "a/b\n.md",
        "a/\x7f.md",
        "x" * 513,
    ],
)
def test_normalize_path_rejects(bad: str) -> None:
    with pytest.raises(PathError):
        normalize_path(bad)


@pytest.mark.parametrize(
    "good", ["a.md", "skills/my-skill/SKILL.md", "a/b/c/d.txt", "x" * 512, "ünï/çødé.md", "a b/c d.md"]
)
def test_normalize_path_accepts(good: str) -> None:
    assert normalize_path(good) == good


def test_normalize_prefix() -> None:
    assert normalize_prefix(None) == ""
    assert normalize_prefix("") == ""
    assert normalize_prefix("/") == ""
    assert normalize_prefix("skills") == "skills/"
    assert normalize_prefix("skills/") == "skills/"
    assert normalize_prefix(" skills/a ") == "skills/a/"
    with pytest.raises(PathError):
        normalize_prefix("../x")
    with pytest.raises(PathError):
        normalize_prefix("/etc")


# ── store ────────────────────────────────────────────────────────────────────


async def test_text_and_binary_round_trip(store: FilesStore) -> None:
    text = "# Skill\nhéllo ✓\n"
    meta = await store.write("skills/a/SKILL.md", text.encode(), updated_by="alpha")
    assert meta["created"] is True
    assert meta["sha256"] == sha256_hex(text.encode())
    assert meta["content_type"] == "text/markdown"
    got_meta, data = await store.read("skills/a/SKILL.md")
    assert data.decode() == text
    assert got_meta["updated_by"] == "alpha"

    blob = bytes(range(256)) * 4
    await store.write("skills/a/blob.bin", blob, updated_by="alpha")
    _, got = await store.read("skills/a/blob.bin")
    assert got == blob

    # decode_payload round trips base64 and rejects junk
    assert decode_payload(base64.b64encode(blob).decode(), "base64") == blob
    with pytest.raises(ValueError, match="base64"):
        decode_payload("not base64!!", "base64")
    with pytest.raises(ValueError, match="encoding"):
        decode_payload("x", "latin-1")


async def test_overwrite_updates_metadata_and_usage(store: FilesStore) -> None:
    await store.write("a.txt", b"one", updated_by="alpha")
    meta = await store.write("a.txt", b"three", updated_by="beta")
    assert meta["created"] is False
    assert meta["updated_by"] == "beta"
    usage = await store.usage()
    assert usage["used_bytes"] == 5
    assert usage["file_count"] == 1


async def test_size_limit(store: FilesStore) -> None:
    ok = b"x" * store.max_file_bytes
    await store.write("big.bin", ok, updated_by="alpha")
    with pytest.raises(FileTooLargeError):
        await store.write("bigger.bin", ok + b"x", updated_by="alpha")
    with pytest.raises(FileTooLargeError):
        decode_payload("x" * (store.max_file_bytes + 1), "utf-8")
    with pytest.raises(FileTooLargeError):
        decode_payload("A" * (store.max_file_bytes * 2), "base64")


async def test_quota_enforced_and_overwrite_counts_net(small_store: FilesStore) -> None:
    await small_store.write("a.bin", b"x" * 60, updated_by="alpha")
    with pytest.raises(QuotaExceededError):
        await small_store.write("b.bin", b"x" * 41, updated_by="alpha")
    await small_store.write("b.bin", b"x" * 40, updated_by="alpha")  # exactly full
    # Overwriting a.bin with a smaller file frees room; growing it past quota is refused.
    await small_store.write("a.bin", b"x" * 10, updated_by="alpha")
    with pytest.raises(QuotaExceededError):
        await small_store.write("a.bin", b"x" * 61, updated_by="alpha")
    usage = await small_store.usage()
    assert usage["used_bytes"] == 50
    assert usage["quota_bytes"] == 100


async def test_if_sha256_conflict(store: FilesStore) -> None:
    first = await store.write("doc.md", b"v1", updated_by="alpha")
    ok = await store.write("doc.md", b"v2", updated_by="alpha", if_sha256=first["sha256"])
    assert ok["sha256"] == sha256_hex(b"v2")
    with pytest.raises(ConflictError) as ei:
        await store.write("doc.md", b"v3", updated_by="beta", if_sha256=first["sha256"])
    assert ei.value.current_sha256 == sha256_hex(b"v2")
    _, data = await store.read("doc.md")
    assert data == b"v2"
    # if_sha256 on a missing file conflicts too
    with pytest.raises(ConflictError) as ei2:
        await store.write("missing.md", b"x", updated_by="beta", if_sha256=sha256_hex(b"x"))
    assert ei2.value.current_sha256 is None


async def test_file_folder_collisions(store: FilesStore) -> None:
    await store.write("a/b", b"file", updated_by="alpha")
    with pytest.raises(ConflictError):
        await store.write("a/b/c", b"x", updated_by="alpha")
    await store.write("x/y/z.md", b"x", updated_by="alpha")
    with pytest.raises(ConflictError):
        await store.write("x/y", b"x", updated_by="alpha")


async def test_move_delete(store: FilesStore) -> None:
    await store.write("skills/a/SKILL.md", b"hi", updated_by="alpha")
    await store.write("skills/b/SKILL.md", b"other", updated_by="alpha")
    moved = await store.move("skills/a/SKILL.md", "skills/c/SKILL.md", updated_by="beta")
    assert moved["path"] == "skills/c/SKILL.md"
    assert moved["updated_by"] == "beta"
    with pytest.raises(KeyError):
        await store.read("skills/a/SKILL.md")
    with pytest.raises(ConflictError):
        await store.move("skills/c/SKILL.md", "skills/b/SKILL.md", updated_by="beta")
    with pytest.raises(KeyError):
        await store.move("nope.md", "other.md", updated_by="beta")
    with pytest.raises(ValueError, match="same"):
        await store.move("skills/c/SKILL.md", "skills/c/SKILL.md", updated_by="beta")
    with pytest.raises(PathError):
        await store.move("skills/c/SKILL.md", "../evil.md", updated_by="beta")
    with pytest.raises(ConflictError):  # 'skills/b/SKILL.md' is a file; can't nest under it
        await store.move("skills/c/SKILL.md", "skills/b/SKILL.md/x", updated_by="beta")

    deleted = await store.delete("skills/c/SKILL.md")
    assert deleted["path"] == "skills/c/SKILL.md"
    with pytest.raises(KeyError):
        await store.delete("skills/c/SKILL.md")
    assert (await store.usage())["used_bytes"] == 5


async def test_list_recursive_and_pagination(store: FilesStore) -> None:
    for p in ("README.md", "skills/a/SKILL.md", "skills/a/ref.md", "skills/b/SKILL.md", "other/x"):
        await store.write(p, b"1", updated_by="alpha")

    everything = await store.list()
    assert [e["path"] for e in everything["entries"]] == [
        "README.md",
        "other/x",
        "skills/a/SKILL.md",
        "skills/a/ref.md",
        "skills/b/SKILL.md",
    ]
    assert everything["next_cursor"] is None
    assert {"path", "size", "sha256", "updated_at", "updated_by"} <= everything["entries"][0].keys()

    scoped = await store.list(prefix="skills")
    assert [e["path"] for e in scoped["entries"]] == [
        "skills/a/SKILL.md",
        "skills/a/ref.md",
        "skills/b/SKILL.md",
    ]
    # 'skill' is not a folder prefix match for 'skills/'
    assert (await store.list(prefix="skill"))["entries"] == []

    seen: list[str] = []
    cursor = None
    while True:
        page = await store.list(prefix="skills/", limit=2, cursor=cursor)
        seen += [e["path"] for e in page["entries"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == [e["path"] for e in scoped["entries"]]


async def test_list_non_recursive_shows_children_and_folders(store: FilesStore) -> None:
    for p in ("README.md", "skills/a/SKILL.md", "skills/a/ref.md", "skills/top.md"):
        await store.write(p, b"12", updated_by="alpha")
    root = await store.list(recursive=False)
    by_path = {e["path"]: e for e in root["entries"]}
    assert set(by_path) == {"README.md", "skills/"}
    assert by_path["skills/"]["type"] == "folder"
    assert by_path["skills/"]["file_count"] == 3
    assert by_path["README.md"]["type"] == "file"

    sub = await store.list(prefix="skills", recursive=False)
    assert [(e["type"], e["path"]) for e in sub["entries"]] == [
        ("folder", "skills/a/"),
        ("file", "skills/top.md"),
    ]

    p1 = await store.list(recursive=False, limit=1)
    assert len(p1["entries"]) == 1 and p1["next_cursor"]
    p2 = await store.list(recursive=False, limit=1, cursor=p1["next_cursor"])
    assert p2["entries"][0]["path"] != p1["entries"][0]["path"]

    with pytest.raises(ValueError, match="limit"):
        await store.list(limit=0)


async def test_migrations_are_versioned_and_idempotent(store: FilesStore) -> None:
    assert await store.init_schema() == []
    async with store.pool.connection() as conn:
        rows = await (await conn.execute("SELECT version FROM files_schema_migrations")).fetchall()
    assert [r["version"] for r in rows] == ["0001_init"]


# ── MCP over Streamable HTTP ─────────────────────────────────────────────────


async def test_mcp_tools_end_to_end(store: FilesStore) -> None:
    port = _free_port()
    app = create_mcp_app(store, module_secret=SECRET)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        base = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient() as hc:
            for _ in range(100):
                try:
                    if (await hc.get(f"{base}/healthz")).status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.05)

        def meta_for(client: str, tool: str, args: dict[str, Any]) -> types.RequestParamsMeta:
            env = sign_mcp_caller(
                SECRET,
                principal_id="p-owner",
                client_id=f"cid-{client}",
                client_name=client,
                tool=tool,
                args=args,
            )
            return cast(types.RequestParamsMeta, {MCP_CALLER_META_KEY: env})

        async def call(session: ClientSession, client: str, tool: str, args: dict[str, Any]) -> Any:
            return await session.call_tool(tool, args, meta=meta_for(client, tool, args))

        def body(res: Any) -> dict[str, Any]:
            return cast(dict[str, Any], json.loads(res.content[0].text))

        async with (
            streamable_http_client(f"{base}/mcp") as (read_s, write_s),
            ClientSession(read_s, write_s) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            assert {t.name for t in listed.tools} == {"list", "read", "write", "delete", "move"}
            for t in listed.tools:
                assert t.description and "skills/<name>/SKILL.md" in t.description

            # Unsigned (system) calls are refused; reserved owner name refused.
            r = await session.call_tool("list", {})
            assert r.is_error and "CALLER_REQUIRED" in r.content[0].text
            r = await call(session, "owner", "list", {})
            assert r.is_error and "CALLER_RESERVED" in r.content[0].text

            # Tampered args are rejected by the caller signature.
            r = await session.call_tool(
                "read", {"path": "a.md"}, meta=meta_for("alpha", "read", {"path": "other.md"})
            )
            assert r.is_error and "CALLER_VERIFICATION_FAILED" in r.content[0].text

            # text write + read
            args = {"path": "skills/demo/SKILL.md", "content": "# Demo\n"}
            r = await call(session, "alpha", "write", args)
            assert not r.is_error
            written = body(r)["file"]
            assert written["updated_by"] == "alpha"
            r = await call(session, "beta", "read", {"path": "skills/demo/SKILL.md"})
            got = body(r)
            assert got["encoding"] == "utf-8" and got["content"] == "# Demo\n"
            assert got["sha256"] == written["sha256"]

            # binary write via base64, read comes back base64
            blob = bytes([0, 255, 1, 254, 2])
            enc = base64.b64encode(blob).decode()
            args = {"path": "skills/demo/blob.bin", "content": enc, "encoding": "base64"}
            assert not (await call(session, "alpha", "write", args)).is_error
            got = body(await call(session, "beta", "read", {"path": "skills/demo/blob.bin"}))
            assert got["encoding"] == "base64"
            assert base64.b64decode(got["content"]) == blob

            # a maximum-size binary file fits through the MCP transport (base64 > default 4 MiB cap)
            big = b"\xff" * (10 * 1024 * 1024)
            args = {
                "path": "big.bin",
                "content": base64.b64encode(big).decode(),
                "encoding": "base64",
            }
            r = await call(session, "alpha", "write", args)
            assert not r.is_error, r.content[0].text
            assert body(r)["file"]["size"] == len(big)
            assert not (await call(session, "alpha", "delete", {"path": "big.bin"})).is_error

            # if_sha256 conflict
            args = {"path": "skills/demo/SKILL.md", "content": "new", "if_sha256": "0" * 64}
            r = await call(session, "beta", "write", args)
            assert r.is_error and "CONFLICT" in r.content[0].text
            assert written["sha256"] in r.content[0].text
            args["if_sha256"] = written["sha256"]
            assert not (await call(session, "beta", "write", args)).is_error

            # path rejection and not found
            r = await call(session, "alpha", "write", {"path": "../x", "content": "x"})
            assert r.is_error and "INVALID_PATH" in r.content[0].text
            r = await call(session, "alpha", "read", {"path": "nope.md"})
            assert r.is_error and "NOT_FOUND" in r.content[0].text
            r = await call(session, "alpha", "write", {"path": "bad64", "content": "!!", "encoding": "base64"})
            assert r.is_error and "INVALID_ARGUMENT" in r.content[0].text

            # list + move + delete
            r = await call(session, "alpha", "list", {"prefix": "skills/", "recursive": False})
            assert [e["path"] for e in body(r)["entries"]] == ["skills/demo/"]
            r = await call(
                session, "alpha", "move", {"from": "skills/demo/blob.bin", "to": "skills/demo/b.bin"}
            )
            assert body(r)["file"]["path"] == "skills/demo/b.bin"
            r = await call(session, "alpha", "delete", {"path": "skills/demo/b.bin"})
            assert body(r)["deleted"] is True
            r = await call(session, "alpha", "list", {})
            assert [e["path"] for e in body(r)["entries"]] == ["skills/demo/SKILL.md"]
    finally:
        server.should_exit = True
        await task


async def test_mcp_size_and_quota_errors(small_store: FilesStore) -> None:
    port = _free_port()
    app = create_mcp_app(small_store, module_secret=SECRET)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        base = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient() as hc:
            for _ in range(100):
                try:
                    if (await hc.get(f"{base}/healthz")).status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.05)
        async with (
            streamable_http_client(f"{base}/mcp") as (read_s, write_s),
            ClientSession(read_s, write_s) as session,
        ):
            await session.initialize()

            async def write(path: str, content: str) -> Any:
                args = {"path": path, "content": content}
                env = sign_mcp_caller(
                    SECRET, principal_id="p", client_id="c", client_name="alpha", tool="write", args=args
                )
                return await session.call_tool(
                    "write", args, meta=cast(types.RequestParamsMeta, {MCP_CALLER_META_KEY: env})
                )

            r = await write("a.txt", "x" * 101)
            assert r.is_error and "QUOTA_EXCEEDED" in r.content[0].text
            r = await write("big.txt", "x" * (10 * 1024 * 1024 + 1))
            assert r.is_error and "FILE_TOO_LARGE" in r.content[0].text
    finally:
        server.should_exit = True
        await task


# ── Owner UI ─────────────────────────────────────────────────────────────────


def _owner_headers() -> dict[str, str]:
    hdr, sig = sign_http_caller(SECRET, principal_id="p-owner", client_id="owner", client_name="owner")
    return {
        HTTP_CALLER_HEADER: hdr,
        HTTP_CALLER_SIGNATURE_HEADER: sig,
        "X-Forwarded-Prefix": "/admin/m/files",
    }


async def test_ui_requires_caller_signature(store: FilesStore) -> None:
    app = create_ui_app(store, module_secret=SECRET)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://files.local"
    ) as client:
        assert (await client.get("/healthz")).status_code == 200
        for path in ("/", "/view?p=a.md", "/download?p=a.md"):
            assert (await client.get(path)).status_code == 403
        for path in ("/upload", "/delete", "/move"):
            assert (await client.post(path, data={"x": "y"})).status_code == 403

        # Forged signature / wrong secret
        hdr, _ = sign_http_caller("other-secret-0123456789", principal_id="p", client_id="owner", client_name="owner")
        forged = {HTTP_CALLER_HEADER: hdr, HTTP_CALLER_SIGNATURE_HEADER: "sha256=" + "0" * 64}
        assert (await client.get("/", headers=forged)).status_code == 403

        assert (await client.get("/", headers=_owner_headers())).status_code == 200


async def test_ui_browse_upload_view_move_delete(store: FilesStore) -> None:
    await store.write("skills/demo/SKILL.md", b"# <b>Demo</b>\n", updated_by="alpha")
    await store.write("skills/demo/blob.bin", b"\x00\x01\x02", updated_by="alpha")
    app = create_ui_app(store, module_secret=SECRET)
    headers = _owner_headers()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://files.local"
    ) as client:
        # Browse root and folder; breadcrumbs, usage, prefix-aware links
        r = await client.get("/", headers=headers)
        assert r.status_code == 200
        assert 'href="/admin/m/files/?p=skills"' in r.text
        assert "of 1.0 GiB used" in r.text
        assert 'name="viewport"' in r.text
        csrf = client.cookies.get("files_csrf")
        assert csrf
        headers = {**headers, "Cookie": f"files_csrf={csrf}"}

        r = await client.get("/?p=skills/demo", headers=headers)
        assert "SKILL.md" in r.text and "blob.bin" in r.text and "crumbs" in r.text

        # Owner viewing/downloading must not change updated_by / updated_at
        before = await store.stat("skills/demo/SKILL.md")
        r = await client.get("/view?p=skills/demo/SKILL.md", headers=headers)
        assert r.status_code == 200
        assert "&lt;b&gt;Demo&lt;/b&gt;" in r.text and "<b>Demo</b>" not in r.text
        r = await client.get("/view?p=skills/demo/blob.bin", headers=headers)
        assert "no preview" in r.text
        r = await client.get("/download?p=skills/demo/SKILL.md", headers=headers)
        assert r.status_code == 200 and r.content == b"# <b>Demo</b>\n"
        assert r.headers["content-disposition"].startswith("attachment")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert (await client.get("/view?p=nope.md", headers=headers)).status_code == 404
        assert (await store.stat("skills/demo/SKILL.md")) == before

        # CSRF required for mutations
        r = await client.post("/delete", data={"path": "skills/demo/SKILL.md"}, headers=headers)
        assert r.status_code == 403
        assert (await store.stat("skills/demo/SKILL.md"))["path"] == "skills/demo/SKILL.md"

        # Multipart upload (records updated_by=owner)
        r = await client.post(
            "/upload",
            data={"csrf_token": csrf, "dir": "skills/new"},
            files=[("files", ("SKILL.md", b"uploaded", "text/markdown")), ("files", ("img.png", b"\x89PNG", "image/png"))],
            headers=headers,
        )
        assert r.status_code == 303
        assert r.headers["location"].startswith("/admin/m/files/?p=skills/new")
        meta, data = await store.read("skills/new/SKILL.md")
        assert data == b"uploaded" and meta["updated_by"] == "owner"
        assert (await store.stat("skills/new/img.png"))["content_type"] == "image/png"

        # Move
        r = await client.post(
            "/move",
            data={"csrf_token": csrf, "from": "skills/new/SKILL.md", "to": "skills/renamed/SKILL.md"},
            headers=headers,
        )
        assert r.status_code == 303
        assert (await store.stat("skills/renamed/SKILL.md"))["updated_by"] == "owner"
        r = await client.post(
            "/move",
            data={"csrf_token": csrf, "from": "skills/renamed/SKILL.md", "to": "../x"},
            headers=headers,
        )
        assert r.status_code == 303 and "error=" in r.headers["location"]

        # Delete
        r = await client.post(
            "/delete", data={"csrf_token": csrf, "path": "skills/renamed/SKILL.md"}, headers=headers
        )
        assert r.status_code == 303
        with pytest.raises(KeyError):
            await store.stat("skills/renamed/SKILL.md")


async def test_ui_upload_limits(small_store: FilesStore) -> None:
    app = create_ui_app(small_store, module_secret=SECRET)
    headers = _owner_headers()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://files.local"
    ) as client:
        await client.get("/", headers=headers)
        csrf = client.cookies.get("files_csrf")
        headers = {**headers, "Cookie": f"files_csrf={csrf}"}

        # Over quota (100 bytes)
        r = await client.post(
            "/upload",
            data={"csrf_token": csrf, "dir": ""},
            files=[("files", ("big.bin", b"x" * 101, "application/octet-stream"))],
            headers=headers,
        )
        assert r.status_code == 303 and "quota" in r.headers["location"].lower()

        # Over per-file limit
        small_store.max_file_bytes = 10
        r = await client.post(
            "/upload",
            data={"csrf_token": csrf, "dir": ""},
            files=[("files", ("big.bin", b"x" * 11, "application/octet-stream"))],
            headers=headers,
        )
        assert r.status_code == 303 and "per-file" in r.headers["location"].lower().replace("%20", " ")
        assert (await small_store.usage())["file_count"] == 0


async def test_ui_upload_over_1mib_multipart_succeeds(store: FilesStore) -> None:
    """Multipart uploads larger than Starlette's 1 MiB spool threshold succeed up to 10 MiB."""
    app = create_ui_app(store, module_secret=SECRET)
    headers = _owner_headers()
    payload = b"A" * (2 * 1024 * 1024)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://files.local"
    ) as client:
        await client.get("/", headers=headers)
        csrf = client.cookies.get("files_csrf")
        headers = {**headers, "Cookie": f"files_csrf={csrf}"}
        r = await client.post(
            "/upload",
            data={"csrf_token": csrf, "dir": "large"},
            files=[("files", ("two-meg.bin", payload, "application/octet-stream"))],
            headers=headers,
        )
        assert r.status_code == 303 and "notice=" in r.headers["location"]
        meta, got = await store.read("large/two-meg.bin")
        assert meta["size"] == len(payload) and got == payload
