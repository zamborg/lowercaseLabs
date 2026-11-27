"""Core integration & E2E tests for the v3.2.2 Files module (catalog manifest, install, tools, tiers)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx2
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.catalog import ModuleManifest, get_module_catalog_entry, load_module_catalog
from zos.config import Settings
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.modules.host import InstalledModule
from zos.vault import generate_secret_key

FILES_SRC_DIR = Path(__file__).resolve().parents[2] / "modules" / "files" / "src"


def _import_files_module() -> tuple[Any, Any, Any]:
    src = str(FILES_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    from zos_files.mcp_server import create_mcp_app
    from zos_files.store import FilesStore, create_files_pool

    return create_files_pool, FilesStore, create_mcp_app


class _InProcessFilesHost:
    """Test ModuleHost that boots the real zos_files ASGI MCP server in-process."""

    kind = "docker"

    def __init__(self) -> None:
        self._servers: dict[str, tuple[uvicorn.Server, asyncio.Task[None], Any]] = {}

    async def install(
        self,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str] | None = None,
    ) -> InstalledModule:
        env = env or {}
        create_files_pool, FilesStore, create_mcp_app = _import_files_module()
        pool = create_files_pool(env["ZOS_MODULE_DATABASE_URL"])
        await pool.open()
        store = FilesStore(pool)
        await store.init_schema()

        mcp_app = create_mcp_app(
            store,
            module_secret=env["ZOS_MODULE_SECRET"],
            mcp_path=manifest.runtime.mcp_path,
        )
        port = free_port()
        srv = uvicorn.Server(uvicorn.Config(mcp_app, host="127.0.0.1", port=port, log_level="warning"))
        task = asyncio.create_task(srv.serve())
        self._servers[manifest.slug] = (srv, task, pool)

        async with httpx2.AsyncClient() as hc:
            for _ in range(50):
                try:
                    if (await hc.get(f"http://127.0.0.1:{port}/healthz")).status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.05)

        return InstalledModule(
            mcp_url=f"http://127.0.0.1:{port}{manifest.runtime.mcp_path}",
            ui_url=None,
            image_pin=f"inproc:{manifest.slug}",
            host="docker",
        )

    async def upgrade(
        self,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str] | None = None,
    ) -> InstalledModule:
        await self.remove(manifest.slug)
        return await self.install(manifest, config, env=env)

    async def stop(self, slug: str) -> None:
        await self.remove(slug)

    async def start(self, slug: str) -> None:
        return None

    async def remove(self, slug: str) -> None:
        entry = self._servers.pop(slug, None)
        if entry is None:
            return
        srv, task, pool = entry
        srv.should_exit = True
        with contextlib.suppress(Exception):
            await task
        with contextlib.suppress(Exception):
            await pool.close()

    async def inspect(self, slug: str) -> dict[str, Any]:
        return {"running": slug in self._servers}


def test_checked_in_files_module_manifest_is_valid() -> None:
    entries = load_module_catalog()
    assert any(e.slug == "files" for e in entries)

    entry = get_module_catalog_entry("files")
    assert entry is not None
    assert entry.manifest_version == 2
    assert entry.version == "0.1.0"
    assert entry.runtime.kind == "container"
    assert entry.runtime.build == "modules/files"
    assert entry.runtime.mcp_port == 8081
    assert entry.runtime.mcp_path == "/mcp"
    assert entry.runtime.ui_port == 8082
    assert entry.storage.postgres is True
    assert entry.network.egress is False
    assert entry.ui is not None and entry.ui.title == "Files" and entry.ui.path == "/"
    assert [c.name for c in entry.config] == ["FILES_QUOTA_BYTES"]
    assert entry.tiers.patterns == {
        "files_list": "read",
        "files_read": "read",
        "files_write": "write",
        "files_delete": "write",
        "files_move": "write",
    }

    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "modules" / "files" / "Dockerfile").is_file()
    assert (repo_root / "modules" / "files" / "pyproject.toml").is_file()


async def test_files_module_catalog_install_lists_tools_tiers_and_round_trips(db_pool) -> None:
    """Install files from catalog into core ZOS and verify tools, tiers, descriptions, and caller attribution."""
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    client = await create_bearer_client(
        db_pool,
        principal_id=p_id,
        space_id=s_id,
        name="Claude",
        trust_ceiling="owner",
    )

    host = _InProcessFilesHost()
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=base_url,
        zos_secret_key=generate_secret_key(),
    )
    app = create_app(cfg, run_migrations_on_startup=False, module_host=host)

    try:
        async with run_asgi_server(app, port=port):
            runtime = app.state.zos
            await runtime.module_manager.install("files", wait_ready=True)
            try:
                async with db_pool.connection() as conn:
                    cur = await conn.execute(
                        "SELECT tool_pattern, tier FROM tool_tiers WHERE principal_id = %s AND tool_pattern LIKE %s",
                        (p_id, "files_%"),
                    )
                    tiers = {r["tool_pattern"]: r["tier"] for r in await cur.fetchall()}
                assert tiers["files_list"] == "read"
                assert tiers["files_read"] == "read"
                assert tiers["files_write"] == "write"
                assert tiers["files_delete"] == "write"
                assert tiers["files_move"] == "write"

                http_c = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client['token']}"})
                async with (
                    http_c,
                    streamable_http_client(f"{base_url}/mcp", http_client=http_c) as (r, w),
                    ClientSession(r, w) as sess,
                ):
                    await sess.initialize()
                    listed = await sess.list_tools()
                    by_name = {t.name: t for t in listed.tools}

                    expected_tools = {
                        "files_list",
                        "files_read",
                        "files_write",
                        "files_delete",
                        "files_move",
                    }
                    assert expected_tools.issubset(by_name)

                    # Verify read vs write annotations derived from tiers
                    assert by_name["files_list"].annotations.read_only_hint is True
                    assert by_name["files_read"].annotations.read_only_hint is True
                    for write_tool in ("files_write", "files_delete", "files_move"):
                        assert by_name[write_tool].annotations.read_only_hint is False
                        assert by_name[write_tool].annotations.open_world_hint is False

                    for tool_name in expected_tools:
                        assert "skills/<name>/SKILL.md" in (by_name[tool_name].description or "")

                    # End-to-end skills round trip with signed caller attribution & optimistic concurrency
                    w_res = await sess.call_tool(
                        "files_write",
                        {
                            "path": "skills/demo/SKILL.md",
                            "content": "# Demo skill\n",
                        },
                    )
                    assert not w_res.is_error
                    written = json.loads(w_res.content[0].text)["file"]
                    assert written["path"] == "skills/demo/SKILL.md"
                    assert written["updated_by"] == "Claude"

                    l_res = await sess.call_tool("files_list", {"prefix": "skills/", "recursive": True})
                    assert not l_res.is_error
                    entries = json.loads(l_res.content[0].text)["entries"]
                    assert [e["path"] for e in entries] == ["skills/demo/SKILL.md"]
                    assert entries[0]["updated_by"] == "Claude"

                    r_res = await sess.call_tool("files_read", {"path": "skills/demo/SKILL.md"})
                    assert not r_res.is_error
                    read_back = json.loads(r_res.content[0].text)
                    assert read_back["encoding"] == "utf-8"
                    assert read_back["content"] == "# Demo skill\n"
                    assert read_back["sha256"] == written["sha256"]

                    m_res = await sess.call_tool(
                        "files_move",
                        {"from": "skills/demo/SKILL.md", "to": "skills/demo-v2/SKILL.md"},
                    )
                    assert not m_res.is_error
                    moved = json.loads(m_res.content[0].text)["file"]
                    assert moved["path"] == "skills/demo-v2/SKILL.md"
                    assert moved["updated_by"] == "Claude"

                    d_res = await sess.call_tool("files_delete", {"path": "skills/demo-v2/SKILL.md"})
                    assert not d_res.is_error
                    assert json.loads(d_res.content[0].text)["deleted"] is True
            finally:
                with contextlib.suppress(Exception):
                    await runtime.module_manager.uninstall("files", purge=True)
    finally:
        await host.remove("files")
