"""Unit tests for M1.4 tool naming, provider registry, sessions, and routing."""

from __future__ import annotations

import json
import uuid
from typing import Any

import mcp_types as types
import pytest

from zos.auth.context import CallContext
from zos.providers.base import ProviderKind, ProviderTool
from zos.router import (
    MAX_TOOL_NAME_LENGTH,
    ProviderRegistry,
    SessionTracker,
    ToolRouter,
    format_tool_name,
    validate_prefix,
)


def _fake_ctx() -> CallContext:
    return CallContext(
        principal_id=uuid.uuid4(),
        principal_slug="zubin",
        space_id=uuid.uuid4(),
        space_slug="main",
        client_id=uuid.uuid4(),
        client_name="test-client",
        client_kind="bearer",
        trust_ceiling="owner",
        token_id=uuid.uuid4(),
    )


class FakeProvider:
    def __init__(
        self,
        namespace: str,
        kind: ProviderKind = "hosted",
        online: bool = True,
        tools: list[ProviderTool] | None = None,
    ) -> None:
        self._namespace = namespace
        self._kind = kind
        self._online = online
        self._tools = tools or []
        self.provider_id = uuid.uuid4()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def kind(self) -> ProviderKind:
        return self._kind

    def is_online(self) -> bool:
        return self._online

    async def list_tools(self) -> list[ProviderTool]:
        return list(self._tools)

    async def call_tool(self, name: str, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        self.calls.append((name, args))
        return types.CallToolResult(content=[types.TextContent(type="text", text=f"{self._namespace}:{name}:{args}")])


class FakeConnection:
    def __init__(self) -> None:
        self.notified = 0

    async def send_tool_list_changed(self) -> None:
        self.notified += 1


def test_tool_naming_validation_and_48_char_cap() -> None:
    assert format_tool_name("memory", "search") == "memory_search"
    assert format_tool_name("gcal", "list_events") == "gcal_list_events"

    # Reject underscore or reserved prefix
    with pytest.raises(ValueError):
        validate_prefix("bad_prefix")
    with pytest.raises(ValueError):
        validate_prefix("zos")
    with pytest.raises(ValueError):
        format_tool_name("memory", "bad name!")

    # Overflow > 48 chars: deterministic truncate + 6-char hash
    long_tool = "very_long_tool_name_that_definitely_exceeds_forty_eight_chars"
    name1 = format_tool_name("provider", long_tool)
    name2 = format_tool_name("provider", long_tool)
    name_other = format_tool_name("provider", long_tool + "_v2")
    assert len(name1) == MAX_TOOL_NAME_LENGTH
    assert name1 == name2
    assert name1 != name_other
    assert name1.startswith("provider_")


async def test_router_lists_online_only_routes_calls_and_reports_offline() -> None:
    reg = ProviderRegistry()
    mem = FakeProvider(
        "memory",
        kind="hosted",
        online=True,
        tools=[ProviderTool(name="get", description="Get memory")],
    )
    mac = FakeProvider(
        "mac",
        kind="actuator",
        online=False,
        tools=[ProviderTool(name="shell_run", description="Run shell")],
    )
    reg.register(mem)
    reg.register(mac)

    router = ToolRouter(reg)
    ctx = _fake_ctx()

    # Only online providers + zos_status appear
    listed = await router.list_tools(ctx)
    names = [t.name for t in listed]
    assert names == ["zos_status", "memory_get"]

    # zos_status reports both providers and online state
    status_res = await router.call_tool("zos_status", {}, ctx)
    assert not status_res.is_error
    status_data = json.loads(status_res.content[0].text)
    assert status_data["providers"] == [
        {"namespace": "mac", "kind": "actuator", "online": False},
        {"namespace": "memory", "kind": "hosted", "online": True},
    ]

    # Routing to online provider works
    get_res = await router.call_tool("memory_get", {"key": "k"}, ctx)
    assert not get_res.is_error
    assert mem.calls == [("get", {"key": "k"})]

    # Calling offline actuator tool returns ACTUATOR_OFFLINE
    off_res = await router.call_tool("mac_shell_run", {"command": "ls"}, ctx)
    assert off_res.is_error
    assert "ACTUATOR_OFFLINE" in off_res.content[0].text

    # Calling unknown tool returns error
    unk_res = await router.call_tool("nope_tool", {}, ctx)
    assert unk_res.is_error
    assert "Unknown tool" in unk_res.content[0].text

    # Bringing mac online exposes mac_shell_run
    mac._online = True
    names2 = [t.name for t in await router.list_tools(ctx)]
    assert names2 == ["zos_status", "mac_shell_run", "memory_get"]


async def test_session_tracker_broadcasts_list_changed() -> None:
    tracker = SessionTracker()
    c1, c2 = FakeConnection(), FakeConnection()
    tracker.register(c1)
    tracker.register(c2)
    assert tracker.count == 2

    count = await tracker.broadcast_tools_changed()
    assert count == 2
    assert c1.notified == 1 and c2.notified == 1

    tracker.unregister(c1)
    assert tracker.count == 1
    assert await tracker.broadcast_tools_changed() == 1
    assert c1.notified == 1 and c2.notified == 2
