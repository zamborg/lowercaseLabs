"""M0.2 throwaway hosting spike client verification script.

Verifies against a Streamable-HTTP MCP server:
1. Initialize and verify server advertises tools.listChanged = True
2. Call `ping` -> "pong"
3. List tools -> verify initial set
4. Call `toggle` -> receive notifications/tools/list_changed and verify `extra_tool` toggled
5. Stay idle for `--idle` seconds (default 120s for Fly verification)
6. Call `ping` and `toggle` again -> verify connection + notifications survive idle window
"""

from __future__ import annotations

import argparse
import asyncio
import time

import anyio
import mcp_types as types
from mcp.client.session import ClientSession, IncomingMessage
from mcp.client.streamable_http import streamable_http_client


async def run_check(url: str, idle_seconds: float) -> None:
    list_changed_events: list[float] = []
    list_changed_signal = anyio.Event()

    async def on_message(message: IncomingMessage) -> None:
        if isinstance(message, types.ToolListChangedNotification):
            list_changed_events.append(time.monotonic())
            list_changed_signal.set()
            print(f"[client] received notifications/tools/list_changed (count={len(list_changed_events)})")
        elif isinstance(message, Exception):
            print(f"[client] transport exception: {message!r}")

    print(f"[client] connecting to {url}")
    async with (
        streamable_http_client(url) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream, message_handler=on_message) as session,
    ):
        init = await session.initialize()
        print(
            f"[client] initialized server={init.server_info.name} "
            f"protocol={init.protocol_version} "
            f"listChanged={init.capabilities.tools.list_changed if init.capabilities.tools else None}"
        )
        assert init.capabilities.tools and init.capabilities.tools.list_changed, (
            "Server must advertise tools.listChanged=True"
        )

        # Give the GET SSE stream a brief moment to establish after notifications/initialized
        await asyncio.sleep(0.5)

        res = await session.call_tool("ping", {})
        text = res.content[0].text if res.content else ""
        print(f"[client] ping -> {text!r}")
        assert text == "pong"

        before_tools = {t.name for t in (await session.list_tools()).tools}
        print(f"[client] tools before toggle: {sorted(before_tools)}")

        list_changed_signal = anyio.Event()
        toggle_res = await session.call_tool("toggle", {})
        print(f"[client] toggle -> {toggle_res.content[0].text!r}")

        with anyio.fail_after(10):
            await list_changed_signal.wait()

        after_tools = {t.name for t in (await session.list_tools()).tools}
        print(f"[client] tools after toggle: {sorted(after_tools)}")
        assert ("extra_tool" in after_tools) != ("extra_tool" in before_tools), (
            f"extra_tool did not toggle: before={before_tools} after={after_tools}"
        )

        print(f"[client] sleeping idle for {idle_seconds:.1f}s...")
        t0 = time.monotonic()
        await asyncio.sleep(idle_seconds)
        elapsed = time.monotonic() - t0
        print(f"[client] woke after {elapsed:.1f}s idle; calling ping and toggle...")

        res2 = await session.call_tool("ping", {})
        text2 = res2.content[0].text if res2.content else ""
        print(f"[client] post-idle ping -> {text2!r}")
        assert text2 == "pong"

        list_changed_signal = anyio.Event()
        toggle_res2 = await session.call_tool("toggle", {})
        print(f"[client] post-idle toggle -> {toggle_res2.content[0].text!r}")

        with anyio.fail_after(10):
            await list_changed_signal.wait()

        final_tools = {t.name for t in (await session.list_tools()).tools}
        print(f"[client] tools after post-idle toggle: {sorted(final_tools)}")
        assert final_tools == before_tools, f"expected {before_tools}, got {final_tools}"

    print("[client] PASS: all hosting spike assertions succeeded.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/mcp")
    parser.add_argument("--idle", type=float, default=120.0)
    args = parser.parse_args()
    asyncio.run(run_check(args.url, args.idle))


if __name__ == "__main__":
    main()
