import asyncio
import contextlib
from typing import Any

import httpx2
import uvicorn

from tests.e2e.helpers import free_port
from zos.catalog import ModuleManifest
from zos.modules.host import InstalledModule


class InProcessModuleHost:
    """Test ModuleHost that boots a module ASGI MCP server in-process."""

    kind = "docker"

    def __init__(self, load_module) -> None:
        self.load_module = load_module
        self._servers: dict[str, tuple[uvicorn.Server, asyncio.Task[None], Any]] = {}

    async def install(
        self,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str] | None = None,
    ) -> InstalledModule:
        env = env or {}
        create_pool, store_type, create_mcp_app = self.load_module()
        pool = create_pool(env["ZOS_MODULE_DATABASE_URL"])
        await pool.open()
        store = store_type(pool)
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

    async def uninstall(self, slug: str, *, purge: bool = False) -> None:
        await self.remove(slug)

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
