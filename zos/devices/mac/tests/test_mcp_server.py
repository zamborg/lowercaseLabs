"""MCP server tests: tool list, hints (read-only except send/shell), and calls over fixtures."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mcp import Client
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from conftest import JARED_ID, MacFixture
from zos_mac.chatdb import ChatStore
from zos_mac.main import build_mac_mcp_server
from zos_mac.messages import MessagesIndex
from zos_mac.send import MessageSender
from zos_mac.shell import TmuxShellRunner

READ_ONLY_TOOLS = {
    "contacts_search",
    "contacts_get",
    "messages_threads",
    "messages_read",
    "messages_search",
}
ALL_TOOLS = READ_ONLY_TOOLS | {"messages_send", "shell_run"}


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
            "--addressbook-root",
            str(tmp_path / "AddressBook"),
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
        assert set(by_name) == ALL_TOOLS

        for name in READ_ONLY_TOOLS:
            ann = by_name[name].annotations
            assert ann is not None
            assert ann.read_only_hint is True, name
            assert ann.destructive_hint is False
            assert ann.open_world_hint is False

        send = by_name["messages_send"].annotations
        assert send is not None
        assert send.read_only_hint is None  # unset -> ZOS defaults to `external` (approval)
        assert "contacts_search" in (by_name["messages_send"].description or "")
        assert "Shortcuts" in (by_name["messages_send"].description or "")

        sr = by_name["shell_run"]
        assert sr.annotations is not None
        assert sr.annotations.read_only_hint is False
        assert sr.annotations.open_world_hint is True


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


async def test_in_process_text_jared_workflow(mac_fixture: MacFixture) -> None:
    calls: list[list[str]] = []

    async def runner(argv: Sequence[str]) -> tuple[int, str, str]:
        calls.append(list(argv))
        return 0, "", ""

    store = ChatStore(mac_fixture.chat_db, mac_fixture.contacts)
    server = build_mac_mcp_server(
        MessagesIndex(mac_fixture.chat_db, mac_fixture.index_db, contacts=mac_fixture.contacts),
        TmuxShellRunner(socket_name="zos-unit-never-used"),
        contacts=mac_fixture.contacts,
        chat_store=store,
        sender=MessageSender(
            store,
            send_backend="applescript",
            runner=runner,
            confirm_timeout_s=0.1,
            poll_interval_s=0.05,
        ),
    )
    async with Client(server) as client:
        found = _payload(await client.call_tool("contacts_search", {"query": "jared"}))
        assert found["contacts"][0]["id"] == JARED_ID

        got = _payload(await client.call_tool("contacts_get", {"id": JARED_ID}))
        assert got["display_name"] == "Jared Dunn"

        threads = _payload(await client.call_tool("messages_threads", {"limit": 3}))
        assert threads["count"] == 3

        convo = _payload(await client.call_tool("messages_read", {"chat": JARED_ID, "limit": 2}))
        assert convo["has_more"] is True

        sent = _payload(await client.call_tool("messages_send", {"to": JARED_ID, "text": "hey"}))
        assert sent["to"] == "+14155550101"
        assert sent["backend"] == "applescript"
        assert sent["status"] == "unconfirmed"  # fake runner records no chat.db row
        assert len(calls) == 1

        ambiguous = await client.call_tool("messages_send", {"to": "SRC-A:3", "text": "hey"})
        assert ambiguous.is_error is True
        assert len(_payload(ambiguous)["options"]) == 2
        assert len(calls) == 1

        missing = await client.call_tool("contacts_get", {"id": "nope"})
        assert missing.is_error is True
        empty = await client.call_tool("messages_send", {"to": "", "text": "hey"})
        assert empty.is_error is True
