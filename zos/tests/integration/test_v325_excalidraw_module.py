"""Catalog installation, routed drawing edits and version conflicts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from tests.e2e.module_host import InProcessModuleHost
from zos.app import create_app
from zos.config import Settings
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.vault import generate_secret_key

EXCALIDRAW_SRC_DIR = Path(__file__).resolve().parents[2] / "modules" / "excalidraw" / "src"


def _import_excalidraw_module() -> tuple[Any, Any, Any]:
    src = str(EXCALIDRAW_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    from zos_excalidraw.mcp_server import create_mcp_app
    from zos_excalidraw.store import DrawingStore, create_pool

    return create_pool, DrawingStore, create_mcp_app


async def test_excalidraw_module_catalog_install_lists_tools_tiers_and_round_trips(db_pool) -> None:
    """Install from the catalog and exercise versioned edits through the core router."""
    async with db_pool.connection() as conn:
        await conn.execute("DROP SCHEMA IF EXISTS mod_excalidraw CASCADE")
        await conn.commit()

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

    host = InProcessModuleHost(_import_excalidraw_module)
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
            await runtime.module_manager.install("excalidraw", wait_ready=True)
            try:
                async with db_pool.connection() as conn:
                    cur = await conn.execute(
                        "SELECT tool_pattern, tier FROM tool_tiers WHERE principal_id = %s AND tool_pattern LIKE %s",
                        (p_id, "excalidraw_%"),
                    )
                    tiers = {r["tool_pattern"]: r["tier"] for r in await cur.fetchall()}

                read_tools = ("excalidraw_list", "excalidraw_get")
                write_tools = ("excalidraw_create", "excalidraw_update")
                for t in read_tools:
                    assert tiers[t] == "read"
                for t in write_tools:
                    assert tiers[t] == "write"

                http_c = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client['token']}"})
                async with (
                    http_c,
                    streamable_http_client(f"{base_url}/mcp", http_client=http_c) as (r, w),
                    ClientSession(r, w) as sess,
                ):
                    await sess.initialize()
                    listed = await sess.list_tools()
                    by_name = {t.name: t for t in listed.tools}

                    expected_tools = set(read_tools) | set(write_tools)
                    assert expected_tools.issubset(by_name)

                    for rt in read_tools:
                        assert by_name[rt].annotations.read_only_hint is True
                    for wt in write_tools:
                        assert by_name[wt].annotations.read_only_hint is False
                        assert by_name[wt].annotations.open_world_hint is False

                    created = await sess.call_tool("excalidraw_create", {"title": "Board"})
                    assert not created.is_error
                    drawing = json.loads(created.content[0].text)
                    assert drawing["updated_by"] == "Claude"
                    args = {"id": drawing["id"], "if_version": 1, "elements": [{"type":"diamond","label":"Start"}]}
                    updated = await sess.call_tool("excalidraw_update", args)
                    assert not updated.is_error
                    stale = await sess.call_tool("excalidraw_update", args)
                    assert stale.is_error
                    detail = json.loads(stale.content[0].text)
                    assert detail["current_version"] == 2
                    assert detail["summary"]["labels"] == ["Start"]
            finally:
                await runtime.module_manager.uninstall("excalidraw", purge=True)
    finally:
        await host.remove("excalidraw")
