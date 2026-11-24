"""Standalone macOS ZOS actuator daemon: dials out over WebSocket and serves MCP."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
import websockets
from mcp.server.lowlevel.server import Server
from mcp.shared._context_streams import create_context_streams
from mcp.shared.message import SessionMessage
from mcp_types import jsonrpc_message_adapter

from zos_actuator.messages import MessagesIndex
from zos_actuator.shell import DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S, TmuxShellRunner

logger = logging.getLogger("zos_actuator")

DEFAULT_CONFIG_PATH = Path("~/.zos-actuator/config.toml")
DEFAULT_CHAT_DB_PATH = Path("~/Library/Messages/chat.db")
DEFAULT_INDEX_DB_PATH = Path("~/.zos-actuator/messages.db")


@dataclass(frozen=True, slots=True)
class ActuatorConfig:
    url: str
    token: str
    chat_db_path: Path = DEFAULT_CHAT_DB_PATH
    index_db_path: Path = DEFAULT_INDEX_DB_PATH
    tmux_socket: str = "zos"
    tmux_session: str = "zos"

    @classmethod
    def from_file(cls, path: Path) -> ActuatorConfig:
        resolved = path.expanduser()
        raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
        url = str(raw.get("url", "")).strip()
        token = str(raw.get("token", "")).strip()
        if not url or not token:
            raise ValueError(f"Config {resolved} must define non-empty 'url' and 'token'")
        return cls(
            url=url,
            token=token,
            chat_db_path=Path(raw.get("chat_db_path", str(DEFAULT_CHAT_DB_PATH))).expanduser(),
            index_db_path=Path(raw.get("index_db_path", str(DEFAULT_INDEX_DB_PATH))).expanduser(),
            tmux_socket=str(raw.get("tmux_socket", "zos")),
            tmux_session=str(raw.get("tmux_session", "zos")),
        )


def build_actuator_mcp_server(
    messages_index: MessagesIndex,
    shell_runner: TmuxShellRunner,
) -> Server[Any]:
    """Create the low-level MCP server exposing exactly two V0 tools: messages_search and shell_run."""

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
                            "handle": {"type": "string", "description": "Optional phone/email/chat filter."},
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
            content=[types.TextContent(type="text", text=f"Unknown actuator tool: {params.name}")],
            is_error=True,
        )

    return Server(
        "zos-actuator",
        version="0.1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def run_once(
    url: str,
    token: str,
    server: Server[Any],
) -> None:
    """Connect once to ZOS /actuator WebSocket and serve MCP until disconnect."""
    async with websockets.connect(
        url,
        additional_headers={"Authorization": f"Bearer {token}"},
        ping_interval=20,
        ping_timeout=10,
        max_size=10 * 1024 * 1024,
    ) as ws:
        logger.info("Connected to ZOS actuator endpoint %s", url)
        read_writer, read_stream = create_context_streams[SessionMessage | Exception](32)
        write_stream, write_reader = create_context_streams[SessionMessage](32)

        async def ws_reader() -> None:
            try:
                async for raw_msg in ws:
                    text = raw_msg if isinstance(raw_msg, str) else raw_msg.decode("utf-8")
                    msg = jsonrpc_message_adapter.validate_json(text, by_name=False)
                    await read_writer.send(SessionMessage(msg))
            except Exception as exc:
                logger.debug("Actuator ws_reader ended: %s", exc)
            finally:
                with contextlib.suppress(Exception):
                    await read_writer.aclose()

        async def ws_writer() -> None:
            try:
                async for session_msg in write_reader:
                    payload = session_msg.message.model_dump_json(by_alias=True, exclude_none=True)
                    await ws.send(payload)
            except Exception as exc:
                logger.debug("Actuator ws_writer ended: %s", exc)
            finally:
                with contextlib.suppress(Exception):
                    await write_reader.aclose()

        async with (
            read_writer,
            read_stream,
            write_stream,
            write_reader,
            anyio.create_task_group() as tg,
        ):
            tg.start_soon(ws_reader)
            tg.start_soon(ws_writer)
            try:
                await server.run(read_stream, write_stream, server.create_initialization_options())
            finally:
                tg.cancel_scope.cancel()


async def run_forever(config: ActuatorConfig) -> None:
    """Maintain persistent outbound connection to ZOS with jittered exponential backoff."""
    messages_index = MessagesIndex(config.chat_db_path, config.index_db_path)
    shell_runner = TmuxShellRunner(
        socket_name=config.tmux_socket,
        session_name=config.tmux_session,
    )
    server = build_actuator_mcp_server(messages_index, shell_runner)

    attempt = 0
    while True:
        started_at = asyncio.get_running_loop().time()
        try:
            await run_once(config.url, config.token, server)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Actuator connection lost: %s", exc)

        elapsed = asyncio.get_running_loop().time() - started_at
        attempt = 0 if elapsed > 10.0 else min(attempt + 1, 6)

        base_delay = min(30.0, float(2 ** max(0, attempt - 1)))
        delay = base_delay * random.uniform(0.75, 1.25)
        logger.info("Reconnecting in %.2fs...", delay)
        await asyncio.sleep(delay)


def main() -> None:
    logging.basicConfig(level=os.environ.get("ZOS_ACTUATOR_LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="ZOS macOS actuator daemon")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("ZOS_ACTUATOR_CONFIG", str(DEFAULT_CONFIG_PATH))),
        help="Path to config.toml (default: ~/.zos-actuator/config.toml)",
    )
    args = parser.parse_args()
    cfg = ActuatorConfig.from_file(args.config)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_forever(cfg))


if __name__ == "__main__":
    main()
