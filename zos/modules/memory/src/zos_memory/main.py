"""Entrypoint running Memory MCP server (`MCP_PORT`) and Owner UI (`UI_PORT`) together."""

from __future__ import annotations

import asyncio
import logging
import os

import uvicorn

from zos_memory.mcp_server import create_mcp_app
from zos_memory.store import MemoryStore, create_memory_pool
from zos_memory.ui import create_ui_app

logger = logging.getLogger(__name__)


async def _serve() -> None:
    database_url = os.environ.get("ZOS_MODULE_DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("ZOS_MODULE_DATABASE_URL is required")
    module_secret = os.environ.get("ZOS_MODULE_SECRET", "").strip()
    if not module_secret:
        raise RuntimeError("ZOS_MODULE_SECRET is required")

    mcp_port = int(os.environ.get("MCP_PORT", "8081"))
    mcp_path = os.environ.get("MCP_PATH", "/mcp").strip() or "/mcp"
    ui_port = int(os.environ.get("UI_PORT", "8082"))

    pool = create_memory_pool(database_url)
    await pool.open()
    try:
        store = MemoryStore(pool)
        await store.init_schema()

        mcp_app = create_mcp_app(store, module_secret=module_secret, mcp_path=mcp_path)
        ui_app = create_ui_app(store, module_secret=module_secret)

        mcp_srv = uvicorn.Server(
            uvicorn.Config(mcp_app, host="0.0.0.0", port=mcp_port, log_level="info")
        )
        ui_srv = uvicorn.Server(
            uvicorn.Config(ui_app, host="0.0.0.0", port=ui_port, log_level="info")
        )
        logger.info("Starting ZOS Memory module (MCP :%d%s, UI :%d)", mcp_port, mcp_path, ui_port)
        await asyncio.gather(mcp_srv.serve(), ui_srv.serve())
    finally:
        await pool.close()


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
