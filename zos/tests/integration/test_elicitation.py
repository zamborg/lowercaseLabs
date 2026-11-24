"""Integration tests for M1.6 MCP elicitation confirmation with real SDK ClientSession."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

import anyio
import mcp_types as types
from mcp.client.session import ClientRequestContext, ClientSession
from mcp.server.lowlevel.server import Server
from mcp.shared._context_streams import create_context_streams
from mcp.shared.message import SessionMessage

from zos.audit import canonical_json
from zos.auth.context import CallContext
from zos.policy import build_tool_annotations, request_tool_confirmation
from zos.providers.base import ProviderTool
from zos.router import PolicyEvaluation, ProviderRegistry, ToolRouter


def _ctx() -> CallContext:
    return CallContext(
        principal_id=uuid.uuid4(),
        principal_slug="zubin",
        space_id=uuid.uuid4(),
        space_slug="main",
        client_id=uuid.uuid4(),
        client_name="claude-code",
        client_kind="bearer",
        trust_ceiling="owner",
        token_id=uuid.uuid4(),
    )


class ExternalFakeProvider:
    namespace = "ext"
    kind = "upstream"
    provider_id = uuid.uuid4()

    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []

    def is_online(self) -> bool:
        return True

    async def list_tools(self) -> list[ProviderTool]:
        return [
            ProviderTool(
                name="send",
                description="External send tool",
                input_schema={"type": "object", "properties": {"body": {"type": "string"}}},
            )
        ]

    async def call_tool(self, name: str, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        self.executed.append(args)
        return types.CallToolResult(content=[types.TextContent(type="text", text="SENT")])


@asynccontextmanager
async def _paired_session(router: ToolRouter, ctx: CallContext, *, elicitation_callback=None):
    async def on_list_tools(rctx, params):
        return types.ListToolsResult(tools=await router.list_tools(ctx))

    async def on_call_tool(rctx, params: types.CallToolRequestParams):
        return await router.call_tool(
            params.name,
            params.arguments or {},
            ctx,
            server_session=rctx.session,
            request_id=rctx.request_id,
        )

    server = Server("zos-test", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    c2s_send, c2s_recv = create_context_streams[SessionMessage | Exception](16)
    s2c_send, s2c_recv = create_context_streams[SessionMessage | Exception](16)

    async def run_server():
        await server.run(c2s_recv, s2c_send, server.create_initialization_options())  # type: ignore[arg-type]

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        async with ClientSession(
            s2c_recv,
            c2s_send,  # type: ignore[arg-type]
            elicitation_callback=elicitation_callback,
        ) as session:
            await session.initialize()
            yield session
        tg.cancel_scope.cancel()


def _make_router(provider: ExternalFakeProvider) -> ToolRouter:
    reg = ProviderRegistry()
    reg.register(provider)

    async def eval_policy(ctx, tool_name, destructive_hint, idempotent_hint):
        tier = "external" if tool_name == "ext_send" else "read"
        decision = "require_approval" if tier == "external" else "allow"
        return PolicyEvaluation(
            tier=tier,
            decision=decision,
            annotations=build_tool_annotations(tier, destructive_hint=destructive_hint),
        )

    return ToolRouter(
        reg,
        evaluate_policy=eval_policy,
        confirm_call=request_tool_confirmation,
    )


async def test_elicitation_accept_true_executes_with_untruncated_args() -> None:
    provider = ExternalFakeProvider()
    router = _make_router(provider)
    seen_messages: list[str] = []
    seen_schemas: list[dict[str, Any]] = []

    async def accept_cb(context: ClientRequestContext, params: types.ElicitRequestParams):
        seen_messages.append(params.message)
        seen_schemas.append(params.requested_schema)  # type: ignore[union-attr]
        return types.ElicitResult(action="accept", content={"approve": True})

    long_payload = {"z_key": "x" * 2500, "a_key": 42}
    async with _paired_session(router, _ctx(), elicitation_callback=accept_cb) as session:
        res = await session.call_tool("ext_send", long_payload)

    assert not res.is_error
    assert res.content[0].text == "SENT"
    assert provider.executed == [long_payload]
    assert len(seen_messages) == 1
    assert "ext_send" in seen_messages[0]
    # Exact, untruncated canonical JSON
    assert canonical_json(long_payload) in seen_messages[0]
    assert seen_schemas[0]["properties"]["approve"]["type"] == "boolean"


async def test_elicitation_accept_false_and_decline_and_cancel_deny() -> None:
    for result_obj in (
        types.ElicitResult(action="accept", content={"approve": False}),
        types.ElicitResult(action="decline"),
        types.ElicitResult(action="cancel"),
    ):
        provider = ExternalFakeProvider()
        router = _make_router(provider)

        async def cb(context: ClientRequestContext, params: types.ElicitRequestParams, _r=result_obj):
            return _r

        async with _paired_session(router, _ctx(), elicitation_callback=cb) as session:
            res = await session.call_tool("ext_send", {"body": "hi"})

        assert res.is_error
        assert provider.executed == []


async def test_no_elicitation_capability_denies_with_policy_override_hint() -> None:
    provider = ExternalFakeProvider()
    router = _make_router(provider)
    ctx = _ctx()

    async with _paired_session(router, ctx, elicitation_callback=None) as session:
        res = await session.call_tool("ext_send", {"body": "hi"})

    assert res.is_error
    assert provider.executed == []
    text = res.content[0].text
    assert "zos policy set --client claude-code ext_send allow" in text
