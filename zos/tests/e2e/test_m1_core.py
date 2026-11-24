"""M1.10 E2E test: two real MCP SDK Streamable HTTP clients sharing memory, annotations, elicitation, and audit."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx2
import mcp_types as types
from mcp.client.session import ClientRequestContext, ClientSession
from mcp.client.streamable_http import streamable_http_client

from tests.conftest import get_test_database_url
from tests.e2e.helpers import run_asgi_server
from zos.app import create_app
from zos.auth.context import CallContext
from zos.config import Settings
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.providers.base import ProviderTool


class FakeExternalProvider:
    namespace = "demo"
    kind = "upstream"
    provider_id = None

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def is_online(self) -> bool:
        return True

    async def list_tools(self) -> list[ProviderTool]:
        return [
            ProviderTool(
                name="action",
                description="Fake external action requiring approval.",
                input_schema={"type": "object", "properties": {"target": {"type": "string"}}},
                destructive_hint=True,
            )
        ]

    async def call_tool(self, name: str, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        self.calls.append(args)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"executed:{args.get('target')}")]
        )


async def test_m1_e2e_shared_memory_annotations_elicitation_and_audit(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    client_a = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="claude-code")
    client_b = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="codex")

    fake_ext = FakeExternalProvider()
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://127.0.0.1:0",
    )
    app = create_app(cfg, extra_providers=[fake_ext], run_migrations_on_startup=False)

    elicited_messages: list[str] = []

    async def approve_callback(context: ClientRequestContext, params: types.ElicitRequestParams):
        elicited_messages.append(params.message)
        return types.ElicitResult(action="accept", content={"approve": True})

    async with run_asgi_server(app) as base_url:
        mcp_url = f"{base_url}/mcp"
        http_a = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_a['token']}"})
        http_b = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_b['token']}"})

        async with (
            http_a,
            http_b,
            streamable_http_client(mcp_url, http_client=http_a) as (ra, wa),
            ClientSession(ra, wa, elicitation_callback=approve_callback) as sess_a,
            streamable_http_client(mcp_url, http_client=http_b) as (rb, wb),
            ClientSession(rb, wb, elicitation_callback=None) as sess_b,
        ):
            await sess_a.initialize()
            await sess_b.initialize()

            # 1. Verify tools & annotations
            tools_a = {t.name: t for t in (await sess_a.list_tools()).tools}
            assert "zos_status" in tools_a
            assert "memory_put" in tools_a
            assert "memory_get" in tools_a
            assert "memory_search" in tools_a
            assert "memory_delete" in tools_a
            assert "demo_action" in tools_a

            assert tools_a["memory_get"].annotations is not None
            assert tools_a["memory_get"].annotations.read_only_hint is True
            assert tools_a["memory_get"].annotations.open_world_hint is False

            assert tools_a["memory_delete"].annotations is not None
            assert tools_a["memory_delete"].annotations.destructive_hint is True
            assert tools_a["memory_delete"].annotations.read_only_hint is False

            assert tools_a["demo_action"].annotations is not None
            assert tools_a["demo_action"].annotations.open_world_hint is True
            assert tools_a["demo_action"].annotations.destructive_hint is True

            # 2. Client A puts memory; Client B gets & searches it
            key = f"shared-{uuid.uuid4().hex[:8]}"
            put_res = await sess_a.call_tool(
                "memory_put",
                {"key": key, "content": "Unifying memories across Claude Code and Codex.", "tags": ["marquee"]},
            )
            assert not put_res.is_error

            get_res = await sess_b.call_tool("memory_get", {"key": key})
            assert not get_res.is_error
            get_payload = json.loads(get_res.content[0].text)
            assert get_payload["found"] is True
            assert "Claude Code and Codex" in get_payload["memory"]["content"]

            search_res = await sess_b.call_tool("memory_search", {"query": "Unifying memories"})
            assert not search_res.is_error
            search_payload = json.loads(search_res.content[0].text)
            assert any(m["key"] == key for m in search_payload["matches"])

            # 3. Fake external tool triggers elicitation for Client A (approved)
            ext_a = await sess_a.call_tool("demo_action", {"target": "alpha"})
            assert not ext_a.is_error
            assert ext_a.content[0].text == "executed:alpha"
            assert len(elicited_messages) == 1
            assert "demo_action" in elicited_messages[0]
            assert '{"target":"alpha"}' in elicited_messages[0]

            # 4. Client B (no elicitation capability) is denied with CLI override message
            ext_b = await sess_b.call_tool("demo_action", {"target": "beta"})
            assert ext_b.is_error
            assert "zos policy set --client codex demo_action allow" in ext_b.content[0].text
            assert fake_ext.calls == [{"target": "alpha"}]

    # 5. Audit events exist for every tool call (5 calls total)
    async with db_pool.connection() as conn:
        cur = await conn.execute(
            "SELECT client_id, tool, decision, outcome, args_digest "
            "FROM events WHERE type = 'tool.called' ORDER BY id ASC"
        )
        events = await cur.fetchall()

    assert len(events) == 5
    summary = [(e["tool"], e["decision"], e["outcome"]) for e in events]
    assert summary == [
        ("memory_put", "allow", "ok"),
        ("memory_get", "allow", "ok"),
        ("memory_search", "allow", "ok"),
        ("demo_action", "require_approval", "ok"),
        ("demo_action", "require_approval", "denied"),
    ]
    assert all(e["args_digest"] is not None for e in events)


async def test_m1_e2e_untrusted_trust_ceiling_denies_external_without_elicitation(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    untrusted = await create_bearer_client(
        db_pool,
        principal_id=p_id,
        space_id=s_id,
        name="untrusted-agent",
        trust_ceiling="untrusted",
    )

    fake_ext = FakeExternalProvider()
    cfg = Settings(database_url=get_test_database_url(), zos_base_url="http://127.0.0.1:0")
    app = create_app(cfg, extra_providers=[fake_ext], run_migrations_on_startup=False)

    elicit_called = False

    async def should_not_be_called(context: ClientRequestContext, params: types.ElicitRequestParams):
        nonlocal elicit_called
        elicit_called = True
        return types.ElicitResult(action="accept", content={"approve": True})

    async with run_asgi_server(app) as base_url:
        mcp_url = f"{base_url}/mcp"
        http_c = httpx2.AsyncClient(headers={"Authorization": f"Bearer {untrusted['token']}"})
        async with (
            http_c,
            streamable_http_client(mcp_url, http_client=http_c) as (r, w),
            ClientSession(r, w, elicitation_callback=should_not_be_called) as sess,
        ):
            await sess.initialize()

            # External tool resolves to deny for untrusted -> hidden from tools/list
            tool_names = {t.name for t in (await sess.list_tools()).tools}
            assert "demo_action" not in tool_names
            assert "memory_put" in tool_names

            # Direct call to external tool is denied immediately without invoking elicitation
            res = await sess.call_tool("demo_action", {"target": "nope"})
            assert res.is_error
            assert "Permission denied" in res.content[0].text
            assert elicit_called is False
            assert fake_ext.calls == []

            # Read/write tools remain allowed
            put_res = await sess.call_tool("memory_put", {"key": "u-k", "content": "ok"})
            assert not put_res.is_error

    async with db_pool.connection() as conn:
        cur = await conn.execute(
            "SELECT tool, decision, outcome FROM events WHERE type = 'tool.called' ORDER BY id ASC"
        )
        rows = [(r["tool"], r["decision"], r["outcome"]) for r in await cur.fetchall()]
    assert rows == [("demo_action", "deny", "denied"), ("memory_put", "allow", "ok")]
