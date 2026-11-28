"""Tests for the standalone ZOS Files module 0.2.0 (POSIX-like store, migrations, 9 MCP tools, UI + CSRF)."""

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
    MIGRATIONS,
    ConflictError,
    FilesStore,
    FileTooLargeError,
    PathError,
    QuotaExceededError,
    create_files_pool,
    decode_payload,
    normalize_dir_path,
    normalize_entry_path,
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


def test_normalize_dir_entry_and_prefix() -> None:
    assert normalize_dir_path(None) == ""
    assert normalize_dir_path("") == ""
    assert normalize_dir_path("/") == ""
    assert normalize_dir_path("skills") == "skills"
    assert normalize_dir_path("skills/") == "skills"
    assert normalize_dir_path(" skills/a ") == "skills/a"
    assert normalize_prefix("skills") == "skills/"
    assert normalize_entry_path("skills/a/") == "skills/a"
    with pytest.raises(PathError):
        normalize_dir_path("../x")
    with pytest.raises(PathError):
        normalize_dir_path("/etc")
    with pytest.raises(PathError):
        normalize_entry_path("")
    with pytest.raises(PathError):
        normalize_entry_path("/")
    with pytest.raises(PathError):
        normalize_entry_path("a/b//")


# ── store: files, folders, POSIX semantics ───────────────────────────────────


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

    assert decode_payload(base64.b64encode(blob).decode(), "base64") == blob
    with pytest.raises(ValueError, match="base64"):
        decode_payload("not base64!!", "base64")
    with pytest.raises(ValueError, match="encoding"):
        decode_payload("x", "latin-1")


async def test_mkdir_p_on_write_and_empty_folder_persists(store: FilesStore) -> None:
    # Writing a deeply nested file auto-creates all missing parent folders.
    await store.write("skills/demo/scripts/run.py", b"print(1)\n", updated_by="alpha")
    for folder in ("skills", "skills/demo", "skills/demo/scripts"):
        st = await store.stat(folder)
        assert st["type"] == "folder"
        assert st["path"] == folder
        assert st["file_count"] == 1
        assert st["updated_by"] == "alpha"

    # Deleting the only file inside leaves the parent folders intact (empty folders persist).
    await store.delete("skills/demo/scripts/run.py")
    st_scripts = await store.stat("skills/demo/scripts")
    assert st_scripts["type"] == "folder"
    assert st_scripts["file_count"] == 0
    assert st_scripts["child_count"] == 0

    # Explicit mkdir is idempotent and empty folders appear in list and tree.
    m1 = await store.mkdir("notes/2026/drafts", updated_by="beta")
    assert m1["created"] is True
    assert m1["path"] == "notes/2026/drafts"
    m2 = await store.mkdir("notes/2026/drafts", updated_by="gamma")
    assert m2["created"] is False

    root = await store.list(path="")
    assert [e["path"] for e in root["entries"]] == ["notes", "skills"]
    assert all(e["type"] == "folder" for e in root["entries"])


async def test_overwrite_updates_metadata_and_usage(store: FilesStore) -> None:
    await store.write("a.txt", b"one", updated_by="alpha")
    meta = await store.write("a.txt", b"three", updated_by="beta")
    assert meta["created"] is False
    assert meta["updated_by"] == "beta"
    usage = await store.usage()
    assert usage["used_bytes"] == 5
    assert usage["file_count"] == 1
    assert usage["folder_count"] == 0


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
    await small_store.write("b.bin", b"x" * 40, updated_by="alpha")
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
    with pytest.raises(ConflictError) as ei2:
        await store.write("missing.md", b"x", updated_by="beta", if_sha256=sha256_hex(b"x"))
    assert ei2.value.current_sha256 is None


async def test_file_folder_collisions(store: FilesStore) -> None:
    await store.write("a/b", b"file", updated_by="alpha")
    with pytest.raises(ConflictError):
        await store.write("a/b/c", b"x", updated_by="alpha")
    with pytest.raises(ConflictError):
        await store.mkdir("a/b", updated_by="alpha")
    with pytest.raises(ConflictError):
        await store.mkdir("a/b/sub", updated_by="alpha")

    await store.mkdir("empty-dir", updated_by="alpha")
    with pytest.raises(ConflictError):
        await store.write("empty-dir", b"x", updated_by="alpha")


async def test_folder_move_semantics_and_guards(store: FilesStore) -> None:
    await store.write("skills/a/SKILL.md", b"hi", updated_by="alpha")
    await store.write("skills/a/scripts/run.sh", b"echo ok", updated_by="alpha")
    await store.mkdir("skills/a/empty-sub", updated_by="alpha")
    await store.write("skills/b/SKILL.md", b"other", updated_by="alpha")

    # Refuse moving into itself or its own subtree
    with pytest.raises(ValueError, match="same"):
        await store.move("skills/a", "skills/a", updated_by="beta")
    with pytest.raises(ValueError, match="subtree"):
        await store.move("skills/a", "skills/a/nested", updated_by="beta")

    # Refuse overwriting an existing folder or file
    with pytest.raises(ConflictError):
        await store.move("skills/a", "skills/b", updated_by="beta")
    with pytest.raises(ConflictError):
        await store.move("skills/a", "skills/b/SKILL.md", updated_by="beta")
    with pytest.raises(ConflictError):
        await store.move("skills/a", "skills/b/SKILL.md/child", updated_by="beta")

    # Move whole folder into a brand-new parent (auto-creates parent folders)
    moved = await store.move("skills/a", "archive/2026/skill-a", updated_by="beta")
    assert moved["type"] == "folder"
    assert moved["path"] == "archive/2026/skill-a"
    assert moved["moved_folders"] == 3  # a, a/scripts, a/empty-sub
    assert moved["moved_files"] == 2
    assert (await store.stat("archive/2026"))["type"] == "folder"
    assert (await store.stat("archive/2026/skill-a/empty-sub"))["type"] == "folder"
    _, data = await store.read("archive/2026/skill-a/scripts/run.sh")
    assert data == b"echo ok"
    with pytest.raises(KeyError):
        await store.stat("skills/a")
    with pytest.raises(KeyError):
        await store.read("skills/a/SKILL.md")


async def test_copy_file_and_folder_and_quota(small_store: FilesStore) -> None:
    await small_store.write("src/a.txt", b"x" * 30, updated_by="alpha")
    await small_store.write("src/sub/b.txt", b"y" * 25, updated_by="alpha")
    await small_store.mkdir("src/empty", updated_by="alpha")

    # Copying src (55 bytes) when 55 bytes already used → 110 > 100 quota refused!
    with pytest.raises(QuotaExceededError):
        await small_store.copy("src", "dst", updated_by="beta")
    with pytest.raises(KeyError):
        await small_store.stat("dst")

    # Shrink b.txt so total is 45 bytes; now copying src (45 bytes) → 90 <= 100 succeeds.
    await small_store.write("src/sub/b.txt", b"y" * 15, updated_by="alpha")
    copied = await small_store.copy("src", "backup/v1/src", updated_by="beta")
    assert copied["type"] == "folder"
    assert copied["copied_files"] == 2
    assert copied["copied_folders"] == 3  # src, src/sub, src/empty
    assert copied["copied_bytes"] == 45
    assert (await small_store.usage())["used_bytes"] == 90
    assert (await small_store.stat("backup/v1/src/empty"))["type"] == "folder"
    _, d = await small_store.read("backup/v1/src/a.txt")
    assert d == b"x" * 30

    # Refuse overwrite and refuse copy into own subtree
    with pytest.raises(ConflictError):
        await small_store.copy("src", "backup/v1/src", updated_by="beta")
    with pytest.raises(ValueError, match="subtree"):
        await small_store.copy("src", "src/inside", updated_by="beta")

    # Single-file copy quota check (90 + 30 > 100 refused; 10-byte file succeeds)
    with pytest.raises(QuotaExceededError):
        await small_store.copy("src/a.txt", "extra.txt", updated_by="beta")
    await small_store.write("tiny.txt", b"12345", updated_by="alpha")
    c_file = await small_store.copy("tiny.txt", "copies/tiny2.txt", updated_by="beta")
    assert c_file["type"] == "file"
    assert c_file["path"] == "copies/tiny2.txt"
    assert c_file["updated_by"] == "beta"


async def test_delete_empty_vs_non_empty_and_recursive(store: FilesStore) -> None:
    await store.write("proj/a.txt", b"1", updated_by="alpha")
    await store.write("proj/sub/b.txt", b"2", updated_by="alpha")
    await store.mkdir("proj/empty", updated_by="alpha")

    # Non-empty folder delete without recursive is refused
    with pytest.raises(ConflictError, match="not empty"):
        await store.delete("proj", recursive=False)
    assert (await store.stat("proj/a.txt"))["type"] == "file"

    # Empty folder deletes cleanly without recursive
    res_empty = await store.delete("proj/empty", recursive=False)
    assert res_empty["type"] == "folder"
    assert res_empty["deleted_files"] == 0
    assert res_empty["deleted_folders"] == 1
    with pytest.raises(KeyError):
        await store.stat("proj/empty")

    # Recursive delete removes whole subtree and returns exact counts
    res_rec = await store.delete("proj", recursive=True)
    assert res_rec["type"] == "folder"
    assert res_rec["deleted_files"] == 2
    assert res_rec["deleted_folders"] == 2  # proj, proj/sub
    assert (await store.usage())["file_count"] == 0
    assert (await store.usage())["folder_count"] == 0


async def test_folder_move_copy_delete_are_atomic_on_mid_subtree_failure(store: FilesStore) -> None:
    """Inject a failure mid-subtree in move, copy, and delete; verify nothing changes."""
    await store.write("tree/a.txt", b"aaa", updated_by="alpha")
    await store.write("tree/sub/b.txt", b"bbb", updated_by="alpha")
    await store.mkdir("tree/empty", updated_by="alpha")

    async def snapshot() -> tuple[list[str], list[str]]:
        async with store.pool.connection() as conn:
            f_rows = await (await conn.execute("SELECT path FROM files ORDER BY path")).fetchall()
            d_rows = await (await conn.execute("SELECT path FROM folders ORDER BY path")).fetchall()
        return [r["path"] for r in f_rows], [r["path"] for r in d_rows]

    before = await snapshot()

    async def boom(stage: str, conn: Any) -> None:
        raise RuntimeError(f"injected failure at {stage}")

    store._fail_hook = boom
    try:
        with pytest.raises(RuntimeError, match="move_mid_subtree"):
            await store.move("tree", "moved/tree", updated_by="beta")
        assert await snapshot() == before

        with pytest.raises(RuntimeError, match="copy_mid_subtree"):
            await store.copy("tree", "copied/tree", updated_by="beta")
        assert await snapshot() == before

        with pytest.raises(RuntimeError, match="delete_mid_subtree"):
            await store.delete("tree", recursive=True)
        assert await snapshot() == before
    finally:
        store._fail_hook = None


async def test_list_folders_first_pagination_and_tree_and_stat(store: FilesStore) -> None:
    await store.write("z-root.md", b"root", updated_by="alpha")
    await store.write("a-root.md", b"root2", updated_by="alpha")
    await store.write("skills/a/SKILL.md", b"skill-a", updated_by="alpha")
    await store.write("skills/a/scripts/deep/tool.py", b"print(1)", updated_by="alpha")
    await store.mkdir("zzz-empty", updated_by="beta")

    # Root ls: folders first (skills, zzz-empty), then files (a-root.md, z-root.md)
    root = await store.list(path="")
    assert [(e["type"], e["path"]) for e in root["entries"]] == [
        ("folder", "skills"),
        ("folder", "zzz-empty"),
        ("file", "a-root.md"),
        ("file", "z-root.md"),
    ]

    # Keyset pagination across folder->file boundary
    seen: list[str] = []
    cur = None
    while True:
        page = await store.list(path="", limit=1, cursor=cur)
        seen.extend(e["path"] for e in page["entries"])
        cur = page["next_cursor"]
        if cur is None:
            break
    assert seen == ["skills", "zzz-empty", "a-root.md", "z-root.md"]

    # Missing folder raises KeyError; listing a file raises ValueError
    with pytest.raises(KeyError):
        await store.list(path="nope")
    with pytest.raises(ValueError, match="is a file"):
        await store.list(path="a-root.md")

    # Stat on root, folder, and file
    st_root = await store.stat("")
    assert st_root["type"] == "folder" and st_root["child_count"] == 4 and st_root["file_count"] == 4
    st_skills = await store.stat("skills")
    assert st_skills["type"] == "folder"
    assert st_skills["child_count"] == 1
    assert st_skills["folder_count"] == 3  # a, a/scripts, a/scripts/deep
    assert st_skills["file_count"] == 2
    assert st_skills["total_size"] == len(b"skill-a") + len(b"print(1)")

    # Tree respects depth and max_entries
    t_shallow = await store.tree(path="", depth=2)
    assert t_shallow["has_deeper"] is True
    assert t_shallow["truncated"] is False
    assert "skills/a/scripts/deep/tool.py" not in [e["path"] for e in t_shallow["entries"]]
    assert "skills/" in t_shallow["text"] and "zzz-empty/" in t_shallow["text"]

    t_capped = await store.tree(path="", depth=5, max_entries=3)
    assert len(t_capped["entries"]) == 3
    assert t_capped["truncated"] is True


async def test_migration_from_010_creates_folder_rows_and_preserves_files() -> None:
    """Seed a 0.1.0 schema (files table only, no folders table) and upgrade to 0.2.0."""
    schema = f"test_files_mig_{uuid.uuid4().hex[:10]}"
    base = _db_url()
    async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in base else "?"
    pool = create_files_pool(f"{base}{sep}options=-csearch_path%3D{schema}")
    await pool.open()
    try:
        # Manually apply ONLY 0001_init (simulating an existing 0.1.0 deployment)
        v1_name, v1_sql = MIGRATIONS[0]
        async with pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE files_schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            for stmt in v1_sql.split(";"):
                if stmt.strip():
                    await conn.execute(stmt.strip())
            await conn.execute("INSERT INTO files_schema_migrations (version) VALUES (%s)", (v1_name,))
            # Seed files with nested paths and one root file
            await conn.execute(
                """
                INSERT INTO files (path, content, size, sha256, content_type, updated_by)
                VALUES
                    ('README.md', %s, 5, %s, 'text/markdown', 'owner'),
                    ('skills/alpha/SKILL.md', %s, 6, %s, 'text/markdown', 'claude'),
                    ('skills/alpha/scripts/run.py', %s, 4, %s, 'text/x-python', 'codex')
                """,
                (
                    b"hello",
                    sha256_hex(b"hello"),
                    b"# alph",
                    sha256_hex(b"# alph"),
                    b"pass",
                    sha256_hex(b"pass"),
                ),
            )
            await conn.commit()

        # Run 0.2.0 init_schema() migration
        migrated_store = FilesStore(pool)
        applied = await migrated_store.init_schema()
        assert applied == ["0002_folders"]
        assert await migrated_store.init_schema() == []  # idempotent

        # Verify all ancestor folders were backfilled and all files survived untouched
        for expected_dir in ("skills", "skills/alpha", "skills/alpha/scripts"):
            st = await migrated_store.stat(expected_dir)
            assert st["type"] == "folder"
        _, d1 = await migrated_store.read("README.md")
        _, d2 = await migrated_store.read("skills/alpha/SKILL.md")
        _, d3 = await migrated_store.read("skills/alpha/scripts/run.py")
        assert (d1, d2, d3) == (b"hello", b"# alph", b"pass")
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


async def test_migrations_are_versioned_and_idempotent(store: FilesStore) -> None:
    assert await store.init_schema() == []
    async with store.pool.connection() as conn:
        rows = await (
            await conn.execute("SELECT version FROM files_schema_migrations ORDER BY version")
        ).fetchall()
    assert [r["version"] for r in rows] == ["0001_init", "0002_folders"]


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
            by_name = {t.name: t for t in listed.tools}
            assert set(by_name) == {
                "list",
                "tree",
                "stat",
                "read",
                "write",
                "mkdir",
                "move",
                "copy",
                "delete",
            }
            for t in listed.tools:
                assert t.description and "skills/<name>/SKILL.md" in t.description
            assert by_name["delete"].annotations.destructive_hint is True

            # Unsigned / reserved / tampered caller checks
            r = await session.call_tool("list", {})
            assert r.is_error and "CALLER_REQUIRED" in r.content[0].text
            r = await call(session, "owner", "list", {})
            assert r.is_error and "CALLER_RESERVED" in r.content[0].text
            r = await session.call_tool(
                "read", {"path": "a.md"}, meta=meta_for("alpha", "read", {"path": "other.md"})
            )
            assert r.is_error and "CALLER_VERIFICATION_FAILED" in r.content[0].text

            # mkdir + write (text & binary) + stat + tree + list
            r = await call(session, "alpha", "mkdir", {"path": "skills/empty"})
            assert not r.is_error and body(r)["created"] is True

            r = await call(
                session, "alpha", "write", {"path": "skills/demo/SKILL.md", "content": "# Demo\n"}
            )
            assert not r.is_error
            written = body(r)["file"]
            assert written["updated_by"] == "alpha"

            blob = bytes([0, 255, 1, 254, 2])
            r = await call(
                session,
                "alpha",
                "write",
                {
                    "path": "skills/demo/blob.bin",
                    "content": base64.b64encode(blob).decode(),
                    "encoding": "base64",
                },
            )
            assert not r.is_error
            got = body(await call(session, "beta", "read", {"path": "skills/demo/blob.bin"}))
            assert got["encoding"] == "base64" and base64.b64decode(got["content"]) == blob

            # 10 MiB binary upload fits through Streamable HTTP transport
            big = b"\xff" * (10 * 1024 * 1024)
            r = await call(
                session,
                "alpha",
                "write",
                {"path": "big.bin", "content": base64.b64encode(big).decode(), "encoding": "base64"},
            )
            assert not r.is_error, r.content[0].text
            assert not (await call(session, "alpha", "delete", {"path": "big.bin"})).is_error

            # if_sha256 conflict + success
            r = await call(
                session,
                "beta",
                "write",
                {"path": "skills/demo/SKILL.md", "content": "new", "if_sha256": "0" * 64},
            )
            assert r.is_error and "CONFLICT" in r.content[0].text
            r = await call(
                session,
                "beta",
                "write",
                {"path": "skills/demo/SKILL.md", "content": "new", "if_sha256": written["sha256"]},
            )
            assert not r.is_error

            # tree & stat
            t = body(await call(session, "alpha", "tree", {"path": "skills", "depth": 3}))
            assert "demo/" in t["text"] and "empty/" in t["text"] and "SKILL.md" in t["text"]
            st = body(await call(session, "alpha", "stat", {"path": "skills/demo"}))
            assert st["type"] == "folder" and st["file_count"] == 2

            # copy folder + move folder + delete non-empty refused + recursive delete
            cp = body(
                await call(session, "alpha", "copy", {"from": "skills/demo", "to": "skills/demo-copy"})
            )
            assert cp["copied"] is True and cp["folder"]["copied_files"] == 2

            mv = body(
                await call(
                    session, "alpha", "move", {"from": "skills/demo-copy", "to": "archive/demo"}
                )
            )
            assert mv["moved"] is True and mv["folder"]["path"] == "archive/demo"

            r = await call(session, "alpha", "delete", {"path": "archive/demo"})
            assert r.is_error and "CONFLICT" in r.content[0].text

            dl = body(
                await call(session, "alpha", "delete", {"path": "archive/demo", "recursive": True})
            )
            assert dl["deleted"] is True and dl["folder"]["deleted_files"] == 2

            # Empty folder delete succeeds without recursive
            dl_empty = body(await call(session, "alpha", "delete", {"path": "skills/empty"}))
            assert dl_empty["deleted"] is True and dl_empty["folder"]["deleted_folders"] == 1
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

            async def call_t(tool: str, args: dict[str, Any]) -> Any:
                env = sign_mcp_caller(
                    SECRET, principal_id="p", client_id="c", client_name="alpha", tool=tool, args=args
                )
                return await session.call_tool(
                    tool, args, meta=cast(types.RequestParamsMeta, {MCP_CALLER_META_KEY: env})
                )

            r = await call_t("write", {"path": "a.txt", "content": "x" * 101})
            assert r.is_error and "QUOTA_EXCEEDED" in r.content[0].text
            r = await call_t("write", {"path": "big.txt", "content": "x" * (10 * 1024 * 1024 + 1)})
            assert r.is_error and "FILE_TOO_LARGE" in r.content[0].text
            assert not (await call_t("write", {"path": "half.txt", "content": "x" * 60})).is_error
            r = await call_t("copy", {"from": "half.txt", "to": "half2.txt"})
            assert r.is_error and "QUOTA_EXCEEDED" in r.content[0].text
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
        for path in ("/mkdir", "/upload", "/delete", "/move", "/copy"):
            assert (await client.post(path, data={"x": "y"})).status_code == 403

        hdr, _ = sign_http_caller("other-secret-0123456789", principal_id="p", client_id="owner", client_name="owner")
        forged = {HTTP_CALLER_HEADER: hdr, HTTP_CALLER_SIGNATURE_HEADER: "sha256=" + "0" * 64}
        assert (await client.get("/", headers=forged)).status_code == 403

        assert (await client.get("/", headers=_owner_headers())).status_code == 200


async def test_ui_file_manager_flows(store: FilesStore) -> None:
    """Test breadcrumbs, folders-first, new folder, upload, preview, move, copy, and folder delete confirm."""
    await store.write("skills/demo/SKILL.md", b"# <b>Demo</b>\n", updated_by="alpha")
    await store.write("skills/demo/blob.bin", b"\x00\x01\x02", updated_by="alpha")
    await store.write("a-top.txt", b"top", updated_by="alpha")
    app = create_ui_app(store, module_secret=SECRET)
    headers = _owner_headers()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://files.local"
    ) as client:
        # Root lists folders before files ('skills/' before 'a-top.txt')
        r = await client.get("/", headers=headers)
        assert r.status_code == 200
        assert 'href="/admin"' in r.text
        assert r.text.index("skills/") < r.text.index("a-top.txt")
        assert 'name="viewport"' in r.text
        csrf = client.cookies.get("files_csrf")
        assert csrf
        headers = {**headers, "Cookie": f"files_csrf={csrf}"}

        # CSRF required on all mutating endpoints
        for endpoint in ("/mkdir", "/move", "/copy", "/delete"):
            assert (await client.post(endpoint, data={"path": "a-top.txt"}, headers=headers)).status_code == 403

        # New folder via UI
        r = await client.post(
            "/mkdir",
            data={"csrf_token": csrf, "parent": "skills", "name": "scratch"},
            headers=headers,
        )
        assert r.status_code == 303 and "Created%20folder" in r.headers["location"]
        assert (await store.stat("skills/scratch"))["type"] == "folder"

        # Empty folder renders nicely with empty-folder delete confirm
        r = await client.get("/?p=skills/scratch", headers=headers)
        assert r.status_code == 200
        assert "This folder is empty." in r.text
        assert "Confirm delete empty folder" in r.text

        # Non-empty folder shows exact file count in its two-step delete confirm
        r = await client.get("/?p=skills/demo", headers=headers)
        assert r.status_code == 200
        assert "Contains 2 files" in r.text
        assert "Confirm delete folder (2 files)" in r.text
        assert 'name="recursive" value="true"' in r.text

        # Owner viewing & downloading does not change updated_by
        before = await store.stat("skills/demo/SKILL.md")
        r = await client.get("/view?p=skills/demo/SKILL.md", headers=headers)
        assert r.status_code == 200
        assert 'href="/admin"' in r.text
        assert "&lt;b&gt;Demo&lt;/b&gt;" in r.text and "<b>Demo</b>" not in r.text
        r = await client.get("/download?p=skills/demo/SKILL.md", headers=headers)
        assert r.status_code == 200 and r.content == b"# <b>Demo</b>\n"
        assert r.headers["content-disposition"].startswith("attachment")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert (await store.stat("skills/demo/SKILL.md")) == before

        # Copy file via UI
        r = await client.post(
            "/copy",
            data={
                "csrf_token": csrf,
                "kind": "file",
                "from": "skills/demo/SKILL.md",
                "to": "skills/demo/SKILL-copy.md",
            },
            headers=headers,
        )
        assert r.status_code == 303
        assert (await store.stat("skills/demo/SKILL-copy.md"))["updated_by"] == "owner"

        # Folder rename/move via UI
        r = await client.post(
            "/move",
            data={
                "csrf_token": csrf,
                "kind": "folder",
                "from": "skills/demo",
                "to": "skills/demo-v2",
            },
            headers=headers,
        )
        assert r.status_code == 303 and "p=skills/demo-v2" in r.headers["location"]
        assert (await store.stat("skills/demo-v2/SKILL.md"))["type"] == "file"
        with pytest.raises(KeyError):
            await store.stat("skills/demo")

        # Folder copy via UI
        r = await client.post(
            "/copy",
            data={
                "csrf_token": csrf,
                "kind": "folder",
                "from": "skills/demo-v2",
                "to": "skills/demo-backup",
            },
            headers=headers,
        )
        assert r.status_code == 303 and "p=skills/demo-backup" in r.headers["location"]
        assert (await store.stat("skills/demo-backup"))["file_count"] == 3

        # Folder delete via UI (recursive=true from two-step confirm)
        r = await client.post(
            "/delete",
            data={"csrf_token": csrf, "path": "skills/demo-backup", "recursive": "true"},
            headers=headers,
        )
        assert r.status_code == 303 and "Deleted%20folder" in r.headers["location"]
        with pytest.raises(KeyError):
            await store.stat("skills/demo-backup")

        # Upload into current folder
        r = await client.post(
            "/upload",
            data={"csrf_token": csrf, "dir": "skills/scratch/"},
            files=[("files", ("note.md", b"hi", "text/markdown"))],
            headers=headers,
        )
        assert r.status_code == 303
        meta, data = await store.read("skills/scratch/note.md")
        assert data == b"hi" and meta["updated_by"] == "owner"


async def test_ui_upload_limits(small_store: FilesStore) -> None:
    app = create_ui_app(small_store, module_secret=SECRET)
    headers = _owner_headers()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://files.local"
    ) as client:
        await client.get("/", headers=headers)
        csrf = client.cookies.get("files_csrf")
        headers = {**headers, "Cookie": f"files_csrf={csrf}"}

        r = await client.post(
            "/upload",
            data={"csrf_token": csrf, "dir": ""},
            files=[("files", ("big.bin", b"x" * 101, "application/octet-stream"))],
            headers=headers,
        )
        assert r.status_code == 303 and "quota" in r.headers["location"].lower()

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
