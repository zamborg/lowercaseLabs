"""Standalone macOS stdio MCP server: Contacts, iMessage/SMS, and a persistent tmux shell.

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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mcp_types as types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server

from zos_mac.chatdb import (
    MAX_READ_LIMIT,
    MAX_THREADS_LIMIT,
    AmbiguousRecipientError,
    ChatStore,
)
from zos_mac.contacts import DEFAULT_ADDRESSBOOK_ROOT, MAX_CONTACT_RESULTS, ContactsIndex
from zos_mac.messages import MessagesIndex
from zos_mac.send import MAX_SEND_CHARS, VALID_SEND_BACKENDS, MessageSender, validate_send_backend
from zos_mac.shell import DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S, TmuxShellRunner

logger = logging.getLogger("zos_mac")

SERVER_VERSION = "0.3.1"
DEFAULT_CHAT_DB_PATH = Path("~/Library/Messages/chat.db")
DEFAULT_INDEX_DB_PATH = Path("~/.zos-actuator/messages.db")

TEXTING_WORKFLOW = (
    "To text someone by name: contacts_search -> (optionally messages_threads / messages_read "
    "for context) -> messages_send."
)


@dataclass(frozen=True, slots=True)
class MacDeviceConfig:
    chat_db_path: Path = DEFAULT_CHAT_DB_PATH
    index_db_path: Path = DEFAULT_INDEX_DB_PATH
    addressbook_root: Path = DEFAULT_ADDRESSBOOK_ROOT
    tmux_socket: str = "zos"
    tmux_session: str = "zos"
    send_backend: str = "auto"

    @classmethod
    def resolve(
        cls,
        *,
        config_path: Path | None = None,
        chat_db_path: Path | None = None,
        index_db_path: Path | None = None,
        addressbook_root: Path | None = None,
        tmux_socket: str | None = None,
        tmux_session: str | None = None,
        send_backend: str | None = None,
    ) -> MacDeviceConfig:
        raw: dict[str, Any] = {}
        if config_path is not None:
            resolved = config_path.expanduser()
            if resolved.is_file():
                raw = tomllib.loads(resolved.read_text(encoding="utf-8"))

        def pick(override: Path | None, key: str, default: Path) -> Path:
            return (override if override is not None else Path(str(raw.get(key, str(default))))).expanduser()

        raw_backend = send_backend if send_backend is not None else str(raw.get("send_backend", "auto"))
        return cls(
            chat_db_path=pick(chat_db_path, "chat_db_path", DEFAULT_CHAT_DB_PATH),
            index_db_path=pick(index_db_path, "index_db_path", DEFAULT_INDEX_DB_PATH),
            addressbook_root=pick(addressbook_root, "addressbook_root", DEFAULT_ADDRESSBOOK_ROOT),
            tmux_socket=tmux_socket or str(raw.get("tmux_socket", "zos")),
            tmux_session=tmux_session or str(raw.get("tmux_session", "zos")),
            send_backend=validate_send_backend(raw_backend),
        )


def _read_only() -> types.ToolAnnotations:
    return types.ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


def _tool_definitions() -> list[types.Tool]:
    return [
        types.Tool(
            name="contacts_search",
            description=(
                "Read-only fuzzy search of the Mac's Contacts (name, nickname, organization; "
                "tolerates typos; also accepts an exact phone or email). Returns contact ids with "
                f"labeled phones and emails. {TEXTING_WORKFLOW}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Name, nickname, company, phone, or email."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_CONTACT_RESULTS, "default": 10},
                },
                "required": ["query"],
            },
            annotations=_read_only(),
        ),
        types.Tool(
            name="contacts_get",
            description="Read-only: one contact by id (from contacts_search) with all labeled phones and emails.",
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Contact id from contacts_search."}},
                "required": ["id"],
            },
            annotations=_read_only(),
        ),
        types.Tool(
            name="messages_threads",
            description=(
                "Read-only: recent iMessage/SMS conversations, newest first. Each has chat_id, display "
                "name, participants (handle + contact name), last message time and snippet, is_group, "
                f"and unread_count. Use chat_id with messages_read or messages_send. {TEXTING_WORKFLOW}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_THREADS_LIMIT, "default": 20},
                    "since": {"type": "string", "description": "Optional ISO 8601 time; only threads active since then."},
                    "query": {
                        "type": "string",
                        "description": "Optional filter on group name, participant name, phone, or email.",
                    },
                },
            },
            annotations=_read_only(),
        ),
        types.Tool(
            name="messages_read",
            description=(
                "Read-only: one conversation in chronological order (sender is 'me' or the contact "
                "name). `chat` is a chat_id (from messages_threads/messages_search), a phone/email, or a "
                "contact id (1:1 chats with that person are merged). Paginate older messages by passing "
                "the returned `next_before` as `before`."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "chat": {"type": "string", "description": "chat_id, phone, email, or contact id."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_READ_LIMIT, "default": 50},
                    "before": {
                        "type": "string",
                        "description": "Cursor `next_before` from a previous call, or an ISO 8601 time.",
                    },
                },
                "required": ["chat"],
            },
            annotations=_read_only(),
        ),
        types.Tool(
            name="messages_search",
            description=(
                "Read-only full-text search over local macOS Messages (chat.db). Results include "
                "chat_id for chaining into messages_read."
            ),
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
            annotations=_read_only(),
        ),
        types.Tool(
            name="messages_send",
            description=(
                "Send an iMessage (or SMS) from the Mac's Messages app. Uses the macOS Shortcuts "
                "backend ('ZOS Send Message') when installed for 1:1 handles, or AppleScript as "
                "fallback (AppleScript is required for group chat_id targets). `to` is a phone "
                "number, an email, a chat_id (required for group chats), or a contact id from "
                "contacts_search. For a contact with several handles the one from the most recent "
                "conversation is used; if that is ambiguous an error lists the options (never guesses). "
                "Returns status `sent` (confirmed in chat.db) or `unconfirmed`, plus `chat_id` and "
                f"`backend`. {TEXTING_WORKFLOW}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Phone, email, chat_id, or contact id."},
                    "text": {"type": "string", "maxLength": MAX_SEND_CHARS, "description": "Message text."},
                },
                "required": ["to", "text"],
            },
            # No readOnlyHint: ZOS defaults this tool to the `external` tier (approval required).
            annotations=types.ToolAnnotations(destructive_hint=False, open_world_hint=True),
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


def _json_result(payload: Any, *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2, ensure_ascii=False))],
        is_error=is_error,
    )


def _error(tool: str, exc: Exception) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=f"{tool} error: {exc}")],
        is_error=True,
    )


def build_mac_mcp_server(
    messages_index: MessagesIndex,
    shell_runner: TmuxShellRunner,
    *,
    contacts: ContactsIndex | None = None,
    chat_store: ChatStore | None = None,
    sender: MessageSender | None = None,
) -> Server[Any]:
    """Create the low-level stdio MCP server exposing Contacts, Messages, and shell tools."""
    contacts_idx = contacts if contacts is not None else messages_index.contacts
    store = chat_store if chat_store is not None else ChatStore(messages_index.chat_db_path, contacts_idx)
    msg_sender = sender if sender is not None else MessageSender(store)

    def contacts_search(args: dict[str, Any]) -> dict[str, Any]:
        hits = contacts_idx.search(str(args.get("query", "")), limit=int(args.get("limit", 10)))
        out: dict[str, Any] = {"count": len(hits), "contacts": [c.to_dict() for c, _s in hits]}
        if contacts_idx.last_hint:
            out["contacts_hint"] = contacts_idx.last_hint
        return out

    def contacts_get(args: dict[str, Any]) -> dict[str, Any]:
        cid = str(args.get("id", "")).strip()
        if not cid:
            raise ValueError("id must be non-empty")
        contact = contacts_idx.get(cid)
        if contact is None:
            hint = f" ({contacts_idx.last_hint})" if contacts_idx.last_hint else ""
            raise LookupError(f"No contact with id {cid!r}; use contacts_search.{hint}")
        return contact.to_dict()

    def messages_threads(args: dict[str, Any]) -> dict[str, Any]:
        since = args.get("since")
        query = args.get("query")
        return store.threads(
            limit=int(args.get("limit", 20)),
            since=str(since) if since else None,
            query=str(query) if query else None,
        )

    def messages_read(args: dict[str, Any]) -> dict[str, Any]:
        before = args.get("before")
        return store.read(
            str(args.get("chat", "")),
            limit=int(args.get("limit", 50)),
            before=str(before) if before else None,
        )

    def messages_search(args: dict[str, Any]) -> dict[str, Any]:
        handle = args.get("handle")
        since = args.get("since")
        return messages_index.search(
            str(args.get("query", "")),
            handle=str(handle) if handle is not None else None,
            since=str(since) if since is not None else None,
            limit=int(args.get("limit", 20)),
        )

    sync_tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "contacts_search": contacts_search,
        "contacts_get": contacts_get,
        "messages_threads": messages_threads,
        "messages_read": messages_read,
        "messages_search": messages_search,
    }

    async def messages_send(args: dict[str, Any]) -> types.CallToolResult:
        try:
            result = await msg_sender.send(args.get("to"), args.get("text"))
        except AmbiguousRecipientError as exc:
            return _json_result({"error": str(exc), "options": exc.options}, is_error=True)
        return _json_result(result)

    async def shell_run(args: dict[str, Any]) -> types.CallToolResult:
        result = await shell_runner.run(
            str(args.get("command", "")),
            timeout_s=float(args.get("timeout_s", DEFAULT_TIMEOUT_S)),
            reset=bool(args.get("reset", False)),
        )
        return _json_result(result, is_error=not result.get("ok", False))

    async_tools: dict[str, Callable[[dict[str, Any]], Awaitable[types.CallToolResult]]] = {
        "messages_send": messages_send,
        "shell_run": shell_run,
    }

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=_tool_definitions())

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = dict(params.arguments or {})
        try:
            if params.name in sync_tools:
                return _json_result(await asyncio.to_thread(sync_tools[params.name], args))
            if params.name in async_tools:
                return await async_tools[params.name](args)
        except Exception as exc:
            return _error(params.name, exc)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Unknown mac tool: {params.name}")],
            is_error=True,
        )

    return Server(
        "zos-mac",
        version=SERVER_VERSION,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def run_stdio(config: MacDeviceConfig) -> None:
    """Serve the macOS MCP server over stdin/stdout."""
    contacts = ContactsIndex(config.addressbook_root)
    messages_index = MessagesIndex(config.chat_db_path, config.index_db_path, contacts=contacts)
    chat_store = ChatStore(config.chat_db_path, contacts)
    sender = MessageSender(chat_store, send_backend=config.send_backend)
    shell_runner = TmuxShellRunner(
        socket_name=config.tmux_socket,
        session_name=config.tmux_session,
    )
    server = build_mac_mcp_server(
        messages_index,
        shell_runner,
        contacts=contacts,
        chat_store=chat_store,
        sender=sender,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def parse_args(argv: Sequence[str] | None = None) -> MacDeviceConfig:
    parser = argparse.ArgumentParser(
        prog="zos-mac",
        description="Standalone macOS stdio MCP server (Contacts, iMessage/SMS, tmux shell).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional TOML config file (reads chat_db_path, index_db_path, addressbook_root, "
            "tmux_socket, tmux_session, send_backend)"
        ),
    )
    parser.add_argument("--chat-db", type=Path, default=None, help="Override Messages chat.db path")
    parser.add_argument("--index-db", type=Path, default=None, help="Override local FTS5 index path")
    parser.add_argument(
        "--addressbook-root", type=Path, default=None, help="Override AddressBook directory"
    )
    parser.add_argument("--tmux-socket", type=str, default=None, help="Override tmux socket name (default: zos)")
    parser.add_argument("--tmux-session", type=str, default=None, help="Override tmux session name (default: zos)")
    parser.add_argument(
        "--send-backend",
        choices=VALID_SEND_BACKENDS,
        default=None,
        help="Message send backend: auto (default), shortcuts, or applescript",
    )
    ns = parser.parse_args(argv)
    return MacDeviceConfig.resolve(
        config_path=ns.config,
        chat_db_path=ns.chat_db,
        index_db_path=ns.index_db,
        addressbook_root=ns.addressbook_root,
        tmux_socket=ns.tmux_socket,
        tmux_session=ns.tmux_session,
        send_backend=ns.send_backend,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=os.environ.get("ZOS_MAC_LOG_LEVEL", "WARNING"))
    cfg = parse_args(argv)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_stdio(cfg))


if __name__ == "__main__":
    main()
