"""MCP stdio server integration test for zos-mac tool list and annotations."""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def test_zos_mac_stdio_mcp_tools_and_annotations(tmp_path: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "zos_mac",
            "--chat-db",
            str(tmp_path / "chat.db"),
            "--index-db",
            str(tmp_path / "messages.db"),
            "--tmux-socket",
            "zos-unit",
        ],
    )
    async with (
        stdio_client(params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as client,
    ):
        await client.initialize()
        listed = await client.list_tools()
        by_name = {t.name: t for t in listed.tools}
        assert set(by_name) == {"messages_search", "shell_run"}

        ms = by_name["messages_search"]
        assert ms.annotations is not None
        assert ms.annotations.read_only_hint is True
        assert ms.annotations.destructive_hint is False
        assert ms.annotations.open_world_hint is False

        sr = by_name["shell_run"]
        assert sr.annotations is not None
        assert sr.annotations.read_only_hint is False
        assert sr.annotations.open_world_hint is True
