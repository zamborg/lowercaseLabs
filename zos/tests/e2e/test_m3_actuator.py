"""M3.6 E2E tests: core + WebSocket actuator dynamic visibility, list_changed, offline & mid-call errors."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import anyio
import httpx2
import mcp_types as types
import websockets
from mcp.client.session import ClientSession, IncomingMessage
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel.server import Server
from mcp.shared._context_streams import create_context_streams
from mcp.shared.message import SessionMessage
from mcp_types import jsonrpc_message_adapter

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.config import Settings
from zos.db.queries import (
    create_actuator_provider_and_token,
    create_bearer_client,
    init_principal_and_space,
    set_client_policy,
)


async def _run_fake_actuator(
    ws_url: str,
    token: str,
    *,
    started_call_event: asyncio.Event,
    hang_on_call: bool = False,
    tag: str = "v1",
) -> None:
    """Minimal WebSocket MCP server mimicking zos-actuator."""

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="messages_search",
                    description="Search messages",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                    annotations=types.ToolAnnotations(read_only_hint=True),
                ),
                types.Tool(
                    name="shell_run",
                    description="Run shell",
                    input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
                ),
            ]
        )

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name == "shell_run" and hang_on_call:
            started_call_event.set()
            await asyncio.sleep(60)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps({"tag": tag, "tool": params.name, "args": params.arguments or {}}),
                )
            ]
        )

    server = Server("fake-mac", version="0.1.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)

    async with websockets.connect(
        ws_url,
        additional_headers={"Authorization": f"Bearer {token}"},
    ) as ws:
        rw, rs = create_context_streams[SessionMessage | Exception](16)
        ws_stream, wr = create_context_streams[SessionMessage](16)

        async def reader() -> None:
            try:
                async for raw in ws:
                    text = raw if isinstance(raw, str) else raw.decode("utf-8")
                    msg = jsonrpc_message_adapter.validate_json(text, by_name=False)
                    await rw.send(SessionMessage(msg))
            except Exception:
                pass
            finally:
                with contextlib.suppress(Exception):
                    await rw.aclose()

        async def writer() -> None:
            try:
                async for sm in wr:
                    await ws.send(sm.message.model_dump_json(by_alias=True, exclude_none=True))
            except Exception:
                pass
            finally:
                with contextlib.suppress(Exception):
                    await wr.aclose()

        async with rw, rs, ws_stream, wr, anyio.create_task_group() as tg:
            tg.start_soon(reader)
            tg.start_soon(writer)
            try:
                await server.run(rs, ws_stream, server.create_initialization_options())
            finally:
                tg.cancel_scope.cancel()


async def test_m3_e2e_actuator_visibility_list_changed_offline_and_mid_call_drop(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    client_rec = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="agent-a")
    # Allow mac_shell_run for this client so we can test mid-call disconnect directly
    await set_client_policy(
        db_pool,
        principal_id=p_id,
        client_id=client_rec["client"]["id"],
        tool_pattern="mac_shell_run",
        decision="allow",
    )
    act = await create_actuator_provider_and_token(db_pool, principal_id=p_id, space_id=s_id, slug="mac")

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    mcp_url = f"{base_url}/mcp"
    ws_url = f"ws://127.0.0.1:{port}/actuator"
    cfg = Settings(database_url=get_test_database_url(), zos_base_url=base_url)
    app = create_app(cfg, run_migrations_on_startup=False)

    list_changed_count = 0
    list_changed_ev = anyio.Event()

    async def on_message(msg: IncomingMessage) -> None:
        nonlocal list_changed_count
        if isinstance(msg, types.ToolListChangedNotification):
            list_changed_count += 1
            list_changed_ev.set()

    async with run_asgi_server(app, port=port):
        http_client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_rec['token']}"})
        async with (
            http_client,
            streamable_http_client(mcp_url, http_client=http_client) as (rs, ws_s),
            ClientSession(rs, ws_s, message_handler=on_message) as session,
        ):
            await session.initialize()
            await asyncio.sleep(0.2)

            # 1. Before actuator connects: mac_* absent, stale call -> ACTUATOR_OFFLINE
            tools_before = {t.name for t in (await session.list_tools()).tools}
            assert "mac_messages_search" not in tools_before
            assert "mac_shell_run" not in tools_before

            stale = await session.call_tool("mac_messages_search", {"query": "hi"})
            assert stale.is_error
            assert "ACTUATOR_OFFLINE" in stale.content[0].text

            # 2. Connect actuator -> SDK client receives list_changed and mac_* tools appear
            list_changed_ev = anyio.Event()
            started_call = asyncio.Event()
            act_task = asyncio.create_task(
                _run_fake_actuator(ws_url, act["token"], started_call_event=started_call, tag="v1")
            )
            with anyio.fail_after(5):
                await list_changed_ev.wait()

            tools_online = {t.name: t for t in (await session.list_tools()).tools}
            assert "mac_messages_search" in tools_online
            assert "mac_shell_run" in tools_online
            assert tools_online["mac_messages_search"].annotations.read_only_hint is True
            assert tools_online["mac_shell_run"].annotations.open_world_hint is True

            # Call online actuator tool
            res_ok = await session.call_tool("mac_messages_search", {"query": "coffee"})
            assert not res_ok.is_error
            assert json.loads(res_ok.content[0].text)["tag"] == "v1"

            # 3. Disconnect actuator -> list_changed fires and mac_* tools disappear
            list_changed_ev = anyio.Event()
            act_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await act_task

            with anyio.fail_after(5):
                await list_changed_ev.wait()

            tools_after = {t.name for t in (await session.list_tools()).tools}
            assert "mac_messages_search" not in tools_after
            assert "mac_shell_run" not in tools_after

            # 4. Reconnect with hanging tool, drop socket mid-call -> ACTUATOR_DISCONNECTED_MID_CALL
            list_changed_ev = anyio.Event()
            started_call2 = asyncio.Event()
            act_task2 = asyncio.create_task(
                _run_fake_actuator(
                    ws_url, act["token"], started_call_event=started_call2, hang_on_call=True, tag="v2"
                )
            )
            with anyio.fail_after(5):
                await list_changed_ev.wait()

            call_task = asyncio.create_task(session.call_tool("mac_shell_run", {"command": "sleep 60"}))
            await asyncio.wait_for(started_call2.wait(), timeout=5.0)
            t0 = time.monotonic()
            act_task2.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await act_task2

            mid_res = await asyncio.wait_for(call_task, timeout=5.0)
            assert (time.monotonic() - t0) < 5.0
            assert mid_res.is_error
            assert "ACTUATOR_DISCONNECTED_MID_CALL" in mid_res.content[0].text


async def test_m3_actuator_connection_replacement_and_bad_token_rejected(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    client_rec = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="agent-rep")
    act = await create_actuator_provider_and_token(db_pool, principal_id=p_id, space_id=s_id, slug="mac")

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    mcp_url = f"{base_url}/mcp"
    ws_url = f"ws://127.0.0.1:{port}/actuator"
    cfg = Settings(database_url=get_test_database_url(), zos_base_url=base_url)
    app = create_app(cfg, run_migrations_on_startup=False)

    async with run_asgi_server(app, port=port):
        # 1. Invalid actuator token is rejected with close code 4401 (or handshake error)
        rejected = False
        try:
            async with websockets.connect(
                ws_url,
                additional_headers={"Authorization": "Bearer zos_a_invalid"},
            ) as ws:
                await asyncio.wait_for(ws.recv(), timeout=2.0)
        except Exception:
            rejected = True
        assert rejected

        # 2. Actuator v1 connects, then v2 replaces v1 cleanly without dropping mac offline
        ev1 = asyncio.Event()
        ev2 = asyncio.Event()
        t1 = asyncio.create_task(_run_fake_actuator(ws_url, act["token"], started_call_event=ev1, tag="v1"))
        try:
            http_client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_rec['token']}"})
            async with (
                http_client,
                streamable_http_client(mcp_url, http_client=http_client) as (rs, ws_s),
                ClientSession(rs, ws_s) as session,
            ):
                await session.initialize()
                # Wait for v1 online
                for _ in range(30):
                    names = {t.name for t in (await session.list_tools()).tools}
                    if "mac_messages_search" in names:
                        break
                    await asyncio.sleep(0.05)

                r1 = await session.call_tool("mac_messages_search", {"query": "q1"})
                assert json.loads(r1.content[0].text)["tag"] == "v1"

                # Connect v2 while v1 is still connected -> replaces v1
                t2 = asyncio.create_task(_run_fake_actuator(ws_url, act["token"], started_call_event=ev2, tag="v2"))
                try:
                    # Wait for v1 task to finish (closed by core with 4000)
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await asyncio.wait_for(t1, timeout=5.0)
                    # Calls must now route to v2 and mac must remain online!
                    r2 = await session.call_tool("mac_messages_search", {"query": "q2"})
                    assert not r2.is_error
                    assert json.loads(r2.content[0].text)["tag"] == "v2"
                finally:
                    t2.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await t2
        finally:
            t1.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t1
