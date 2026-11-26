"""Standalone macOS stdio MCP server exposing Messages search and persistent tmux shell.

Works directly with any stdio MCP client (e.g. Claude Desktop) or relayed to ZOS via `zos-device`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mcp_types as types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server

from zos_mac.messages import MessagesIndex
from zos_mac.shell import DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S, TmuxShellRunner

logger = logging.getLogger("zos_mac")

DEFAULT_CHAT_DB_PATH = Path("~/Library/Messages/chat.db")
DEFAULT_INDEX_DB_PATH = Path("~/.zos-actuator/messages.db")


@dataclass(frozen=True, slots=True)
class MacDeviceConfig:
    chat_db_path: Path = DEFAULT_CHAT_DB_PATH
    index_db_path: Path = DEFAULT_INDEX_DB_PATH
    tmux_socket: str = "zos"
    tmux_session: str = "zos"

    @classmethod
    def resolve(
        cls,
        *,
        config_path: Path | None = None,
        chat_db_path: Path | None = None,
        index_db_path: Path | None = None,
        tmux_socket: str | None = None,
        tmux_session: str | None = None,
    ) -> MacDeviceConfig:
        raw: dict[str, Any] = {}
        if config_path is not None:
            resolved = config_path.expanduser()
            if resolved.is_file():
                raw = tomllib.loads(resolved.read_text(encoding="utf-8"))

        final_chat_db = (
            chat_db_path
            if chat_db_path is not None
            else Path(str(raw.get("chat_db_path", str(DEFAULT_CHAT_DB_PATH))))
        ).expanduser()
        final_index_db = (
            index_db_path
            if index_db_path is not None
            else Path(str(raw.get("index_db_path", str(DEFAULT_INDEX_DB_PATH))))
        ).expanduser()
        final_socket = tmux_socket or str(raw.get("tmux_socket", "zos"))
        final_session = tmux_session or str(raw.get("tmux_session", "zos"))
        return cls(
            chat_db_path=final_chat_db,
            index_db_path=final_index_db,
            tmux_socket=final_socket,
            tmux_session=final_session,
        )


def build_mac_mcp_server(
    messages_index: MessagesIndex,
    shell_runner: TmuxShellRunner,
) -> Server[Any]:
    """Create the low-level stdio MCP server exposing messages_search and shell_run."""

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="messages_search",
                    description="Read-only full-text search over local macOS Messages (chat.db).",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search terms."},
                            "handle": {
                                "type": "string",
                                "description": "Optional phone, email, chat identifier, or contact name filter.",
                            },
                            "since": {"type": "string", "description": "Optional ISO 8601 minimum date (UTC)."},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
                        },
                        "required": ["query"],
                    },
                    annotations=types.ToolAnnotations(
                        read_only_hint=True,
                        destructive_hint=False,
                        open_world_hint=False,
                    ),
                ),
                types.Tool(
                    name="shell_run",
                    description="Run a command in a persistent local tmux shell session (`tmux -L zos`).",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command or script to execute."},
                            "timeout_s": {
                                "type": "number",
                                "minimum": 1,
                                "maximum": MAX_TIMEOUT_S,
                                "default": DEFAULT_TIMEOUT_S,
                            },
                            "reset": {
                                "type": "boolean",
                                "default": False,
                                "description": "If pane is busy, send Ctrl-C before running.",
                            },
                        },
                        "required": ["command"],
                    },
                    annotations=types.ToolAnnotations(
                        read_only_hint=False,
                        destructive_hint=False,
                        open_world_hint=True,
                    ),
                ),
            ]
        )

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = params.arguments or {}
        if params.name == "messages_search":
            try:
                query = str(args.get("query", ""))
                handle = args.get("handle")
                since = args.get("since")
                limit = int(args.get("limit", 20))
                result = await asyncio.to_thread(
                    messages_index.search,
                    query,
                    handle=str(handle) if handle is not None else None,
                    since=str(since) if since is not None else None,
                    limit=limit,
                )
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as exc:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"messages_search error: {exc}")],
                    is_error=True,
                )

        if params.name == "shell_run":
            try:
                command = str(args.get("command", ""))
                timeout_s = float(args.get("timeout_s", DEFAULT_TIMEOUT_S))
                reset = bool(args.get("reset", False))
                result = await shell_runner.run(command, timeout_s=timeout_s, reset=reset)
                is_err = not result.get("ok", False)
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(result, indent=2))],
                    is_error=is_err,
                )
            except Exception as exc:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"shell_run error: {exc}")],
                    is_error=True,
                )

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Unknown mac tool: {params.name}")],
            is_error=True,
        )

    return Server(
        "zos-mac",
        version="0.2.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def run_stdio(config: MacDeviceConfig) -> None:
    """Serve the macOS MCP server over stdin/stdout."""
    messages_index = MessagesIndex(config.chat_db_path, config.index_db_path)
    shell_runner = TmuxShellRunner(
        socket_name=config.tmux_socket,
        session_name=config.tmux_session,
    )
    server = build_mac_mcp_server(messages_index, shell_runner)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def parse_args(argv: Sequence[str] | None = None) -> MacDeviceConfig:
    parser = argparse.ArgumentParser(
        prog="zos-mac",
        description="Standalone macOS stdio MCP server (Messages search + tmux shell).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional TOML config file (reads chat_db_path, index_db_path, tmux_socket, tmux_session)",
    )
    parser.add_argument("--chat-db", type=Path, default=None, help="Override Messages chat.db path")
    parser.add_argument("--index-db", type=Path, default=None, help="Override local FTS5 index path")
    parser.add_argument("--tmux-socket", type=str, default=None, help="Override tmux socket name (default: zos)")
    parser.add_argument("--tmux-session", type=str, default=None, help="Override tmux session name (default: zos)")
    ns = parser.parse_args(argv)
    return MacDeviceConfig.resolve(
        config_path=ns.config,
        chat_db_path=ns.chat_db,
        index_db_path=ns.index_db,
        tmux_socket=ns.tmux_socket,
        tmux_session=ns.tmux_session,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=os.environ.get("ZOS_MAC_LOG_LEVEL", "WARNING"))
    cfg = parse_args(argv)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_stdio(cfg))


if __name__ == "__main__":
    main()
