"""Core integration & E2E tests for the v3.2.1 Tasks module (catalog manifest, install, tools, tiers)."""

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

TASKS_SRC_DIR = Path(__file__).resolve().parents[2] / "modules" / "tasks" / "src"


def _import_tasks_module() -> tuple[Any, Any, Any]:
    src = str(TASKS_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    from zos_tasks.mcp_server import create_mcp_app
    from zos_tasks.store import TasksStore, create_tasks_pool

    return create_tasks_pool, TasksStore, create_mcp_app


class _InProcessTasksHost:
    """Test ModuleHost that boots the real zos_tasks ASGI MCP server in-process."""

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
        create_tasks_pool, TasksStore, create_mcp_app = _import_tasks_module()
        pool = create_tasks_pool(env["ZOS_MODULE_DATABASE_URL"])
        await pool.open()
        store = TasksStore(pool)
        await store.init_schema()

        mcp_app = create_mcp_app(store, module_secret=env["ZOS_MODULE_SECRET"], mcp_path=manifest.runtime.mcp_path)
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

    async def upgrade(self, manifest: ModuleManifest, config: dict[str, str], *, env: dict[str, str] | None = None) -> InstalledModule:
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


def test_checked_in_tasks_module_manifest_is_valid() -> None:
    entries = load_module_catalog()
    assert any(e.slug == "tasks" for e in entries)

    entry = get_module_catalog_entry("tasks")
    assert entry is not None
    assert entry.manifest_version == 2
    assert entry.version == "0.1.0"
    assert entry.runtime.kind == "container"
    assert entry.runtime.build == "modules/tasks"
    assert entry.runtime.mcp_port == 8081
    assert entry.runtime.mcp_path == "/mcp"
    assert entry.runtime.ui_port == 8082
    assert entry.storage.postgres is True
    assert entry.network.egress is False
    assert entry.ui is not None and entry.ui.title == "Tasks" and entry.ui.path == "/"
    assert entry.tiers.patterns == {
        "tasks_list": "read",
        "tasks_get": "read",
        "tasks_add": "write",
        "tasks_update": "write",
        "tasks_complete": "write",
    }

    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "modules" / "tasks" / "Dockerfile").is_file()
    assert (repo_root / "modules" / "tasks" / "pyproject.toml").is_file()


async def test_tasks_module_catalog_install_lists_tools_tiers_and_round_trips(db_pool) -> None:
    """Install tasks from catalog into core ZOS and verify tools, tiers, descriptions, and caller attribution."""
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

    host = _InProcessTasksHost()
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
            await runtime.module_manager.install("tasks", wait_ready=True)
            try:
                # Verify persisted tier rows match manifest
                async with db_pool.connection() as conn:
                    cur = await conn.execute(
                        "SELECT tool_pattern, tier FROM tool_tiers WHERE principal_id = %s AND tool_pattern LIKE %s",
                        (p_id, "tasks_%"),
                    )
                    tiers = {r["tool_pattern"]: r["tier"] for r in await cur.fetchall()}
                assert tiers["tasks_list"] == "read"
                assert tiers["tasks_get"] == "read"
                assert tiers["tasks_add"] == "write"
                assert tiers["tasks_update"] == "write"
                assert tiers["tasks_complete"] == "write"

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
                        "tasks_list",
                        "tasks_get",
                        "tasks_add",
                        "tasks_update",
                        "tasks_complete",
                    }
                    assert expected_tools.issubset(by_name)
                    assert "tasks_delete" not in by_name

                    # Verify read vs write annotations derived from tiers
                    assert by_name["tasks_list"].annotations.read_only_hint is True
                    assert by_name["tasks_get"].annotations.read_only_hint is True
                    assert by_name["tasks_add"].annotations.read_only_hint is False
                    assert by_name["tasks_add"].annotations.open_world_hint is False
                    assert "for the human owner" in (by_name["tasks_add"].description or "")

                    # Round trip through ZOS router with signed caller attribution
                    add_res = await sess.call_tool(
                        "tasks_add",
                        {"title": "File tax extension", "due": "2026-12-15", "tags": ["finance"]},
                    )
                    assert not add_res.is_error
                    added = json.loads(add_res.content[0].text)["task"]
                    assert added["created_by"] == "Claude"
                    assert added["status"] == "open"

                    list_res = await sess.call_tool("tasks_list", {"tag": "finance"})
                    assert not list_res.is_error
                    listed_tasks = json.loads(list_res.content[0].text)["tasks"]
                    assert [t["id"] for t in listed_tasks] == [added["id"]]

                    comp_res = await sess.call_tool("tasks_complete", {"id": added["id"]})
                    assert not comp_res.is_error
                    assert json.loads(comp_res.content[0].text)["task"]["status"] == "done"
            finally:
                with contextlib.suppress(Exception):
                    await runtime.module_manager.uninstall("tasks", purge=True)
    finally:
        await host.remove("tasks")
