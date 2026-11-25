"""M8.2 tests for MCP server InitializeResult `instructions` and ZOS_INSTRUCTIONS_FILE override."""

from __future__ import annotations

from pathlib import Path

import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.config import Settings
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.instructions import DEFAULT_MCP_INSTRUCTIONS


async def test_mcp_initialize_returns_default_instructions(db_pool) -> None:
    assert len(DEFAULT_MCP_INSTRUCTIONS) < 1200

    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    client_rec = await create_bearer_client(
        db_pool,
        principal_id=init["principal"]["id"],
        space_id=init["space"]["id"],
        name="inst-client",
    )

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    app = create_app(
        Settings(database_url=get_test_database_url(), zos_base_url=base),
        run_migrations_on_startup=False,
    )

    async with run_asgi_server(app, port=port):
        http_c = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_rec['token']}"})
        async with (
            http_c,
            streamable_http_client(f"{base}/mcp", http_client=http_c) as (r, w),
            ClientSession(r, w) as sess,
        ):
            init_res = await sess.initialize()
            assert init_res.instructions == DEFAULT_MCP_INSTRUCTIONS
            assert init_res.instructions is not None
            assert len(init_res.instructions) < 1200
            for expected in ("memory_search", "memory_list", "memory_put", "external"):
                assert expected in init_res.instructions


async def test_mcp_initialize_respects_instructions_file_override(db_pool, tmp_path: Path) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    client_rec = await create_bearer_client(
        db_pool,
        principal_id=init["principal"]["id"],
        space_id=init["space"]["id"],
        name="inst-override-client",
    )

    custom_file = tmp_path / "custom-instructions.txt"
    custom_text = "Custom ZOS routing instructions for testing."
    custom_file.write_text(custom_text + "\n", encoding="utf-8")

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    app = create_app(
        Settings(
            database_url=get_test_database_url(),
            zos_base_url=base,
            instructions_file=str(custom_file),
        ),
        run_migrations_on_startup=False,
    )

    async with run_asgi_server(app, port=port):
        http_c = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_rec['token']}"})
        async with (
            http_c,
            streamable_http_client(f"{base}/mcp", http_client=http_c) as (r, w),
            ClientSession(r, w) as sess,
        ):
            init_res = await sess.initialize()
            assert init_res.instructions == custom_text
