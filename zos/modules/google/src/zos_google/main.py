"""Shared entrypoint for `gmail` and `gcal` ZOS modules (selected via `ZOS_MODULE_SLUG`)."""

from __future__ import annotations

import asyncio
import logging
import os

import uvicorn

from zos_google.calendar import GoogleCalendarService
from zos_google.gmail import GmailService
from zos_google.mcp_server import create_mcp_app
from zos_google.oauth import DEFAULT_GCAL_MCP_URL, GoogleClient
from zos_google.store import GoogleTokenStore, create_google_pool
from zos_google.ui import create_ui_app

logger = logging.getLogger(__name__)


async def _serve() -> None:
    service = (
        os.environ.get("GOOGLE_SERVICE", "").strip().lower()
        or os.environ.get("ZOS_MODULE_SLUG", "").strip().lower()
    )
    if service not in ("gmail", "gcal"):
        raise RuntimeError(
            f"ZOS_MODULE_SLUG or GOOGLE_SERVICE must be 'gmail' or 'gcal', got {service!r}"
        )

    database_url = os.environ.get("ZOS_MODULE_DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("ZOS_MODULE_DATABASE_URL is required")
    module_secret = os.environ.get("ZOS_MODULE_SECRET", "").strip()
    if not module_secret:
        raise RuntimeError("ZOS_MODULE_SECRET is required")

    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    gcal_backend = os.environ.get("GCAL_BACKEND", "api").strip().lower() or "api"
    gcal_mcp_url = os.environ.get("GCAL_MCP_URL", DEFAULT_GCAL_MCP_URL).strip() or DEFAULT_GCAL_MCP_URL

    mcp_port = int(os.environ.get("MCP_PORT", "8081"))
    mcp_path = os.environ.get("MCP_PATH", "/mcp").strip() or "/mcp"
    ui_port = int(os.environ.get("UI_PORT", "8082"))

    pool = create_google_pool(database_url)
    await pool.open()
    try:
        store = GoogleTokenStore(pool, module_secret=module_secret, service=service)
        await store.init_schema()

        client = GoogleClient(
            store,
            service=service,
            client_id=client_id,
            client_secret=client_secret,
        )
        cal_service: GoogleCalendarService | None = None
        if service == "gmail":
            handler: GmailService | GoogleCalendarService = GmailService(client)
        else:
            cal_service = GoogleCalendarService(
                client,
                default_backend=gcal_backend,
                mcp_url=gcal_mcp_url,
            )
            handler = cal_service

        mcp_app = create_mcp_app(
            handler,
            service=service,
            module_secret=module_secret,
            mcp_path=mcp_path,
        )
        ui_app = create_ui_app(
            client,
            service=service,
            module_secret=module_secret,
            calendar_service=cal_service,
        )

        mcp_srv = uvicorn.Server(
            uvicorn.Config(mcp_app, host="0.0.0.0", port=mcp_port, log_level="info")
        )
        ui_srv = uvicorn.Server(
            uvicorn.Config(ui_app, host="0.0.0.0", port=ui_port, log_level="info")
        )
        logger.info(
            "Starting ZOS Google module service=%s (MCP :%d%s, UI :%d)",
            service,
            mcp_port,
            mcp_path,
            ui_port,
        )
        await asyncio.gather(mcp_srv.serve(), ui_srv.serve())
    finally:
        await pool.close()


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
