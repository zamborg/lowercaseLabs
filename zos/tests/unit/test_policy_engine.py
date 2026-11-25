"""Unit tests for M10a policy split: visibility vs allowance engine chain & built-in guards."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import mcp_types as types
import pytest

from zos.auth.context import CallContext, TrustCeiling
from zos.policy import (
    EngineChain,
    FunctionPolicyEngine,
    PolicyClient,
    PolicyInput,
    PolicySnapshot,
    PolicyTool,
    TableEngine,
    Verdict,
    load_policy_engine,
    load_policy_engines,
    visible,
)
from zos.policy.guards import deny_rm_in_shell
from zos.providers.base import ProviderKind, ProviderTool
from zos.router import ProviderRegistry, ToolRouter


def _ctx(*, trust: TrustCeiling = "owner", name: str = "agent") -> CallContext:
    return CallContext(
        principal_id=uuid.uuid4(),
        principal_slug="owner",
        space_id=uuid.uuid4(),
        space_slug="main",
        client_id=uuid.uuid4(),
        client_name=name,
        client_kind="bearer",
        trust_ceiling=trust,
        token_id=uuid.uuid4(),
    )


def _policy_input(
    *,
    tool: str = "mac_shell_run",
    tier: str = "external",
    args: dict[str, Any] | None = None,
    trust: TrustCeiling = "owner",
) -> PolicyInput:
    return PolicyInput(
        principal_id=uuid.uuid4(),
        client=PolicyClient(id=uuid.uuid4(), name="agent", trust_ceiling=trust),
        tool=PolicyTool(exposed_name=tool, tier=tier),  # type: ignore[arg-type]
        args=args or {},
    )


class _StubProvider:
    def __init__(self, namespace: str, tools: list[ProviderTool]) -> None:
        self._namespace = namespace
        self._tools = tools
        self.provider_id = uuid.uuid4()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def kind(self) -> ProviderKind:
        return "hosted"

    def is_online(self) -> bool:
        return True

    async def list_tools(self) -> list[ProviderTool]:
        return list(self._tools)

    async def call_tool(self, name: str, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        self.calls.append((name, args))
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])


@pytest.mark.parametrize(
    ("tool", "command", "expected"),
    [
        ("mac_shell_run", "rm -rf /tmp/foo", "deny"),
        ("mac_shell_run", "sudo rm /etc/hosts", "deny"),
        ("mac_shell_run", "x;rm foo", "deny"),
        ("mac_shell_run", r"\rm foo", "deny"),
        ("linux_shell_run", "/bin/rm -f file", "deny"),
        ("mac_shell_run", "echo farm", "pass"),
        ("mac_shell_run", "ls -la farm/bar", "pass"),
        ("mac_shell_run", "docker run --rm alpine", "pass"),
        ("mac_shell_run", "echo my_rm_file", "pass"),
        ("memory_put", "rm -rf /", "pass"),
    ],
)
async def test_deny_rm_in_shell_guard(tool: str, command: str, expected: Verdict) -> None:
    inp = _policy_input(tool=tool, tier="external", args={"command": command})
    assert await deny_rm_in_shell.decide(inp) == expected
    assert deny_rm_in_shell.name == "deny_rm_in_shell"


def test_load_policy_engines_imports_builtin_guard() -> None:
    loaded = load_policy_engines(["zos.policy.guards:deny_rm_in_shell"])
    assert len(loaded) == 1
    assert loaded[0].name == "deny_rm_in_shell"

    with pytest.raises(ValueError):
        load_policy_engine("invalid_no_colon")


async def test_engine_chain_order_and_pass_fallthrough() -> None:
    seen: list[str] = []

    async def first(inp: PolicyInput) -> Verdict:
        seen.append("first")
        return "pass"

    async def second(inp: PolicyInput) -> Verdict:
        seen.append("second")
        return "deny"

    async def third(inp: PolicyInput) -> Verdict:
        seen.append("third")
        return "allow"

    snap = PolicySnapshot(tiers={"mac_shell_run": "write"}, client_rules={}, global_rules={})
    chain = EngineChain(
        [
            FunctionPolicyEngine("first", first),
            FunctionPolicyEngine("second", second),
            FunctionPolicyEngine("third", third),
        ]
    )
    res = await chain.evaluate(_policy_input(tool="mac_shell_run", tier="write"), table_engine=TableEngine(snap))
    assert seen == ["first", "second"]
    assert res.verdict == "deny"
    assert res.decision == "deny"
    assert res.engine == "second"
    assert res.error is None


async def test_engine_chain_falls_through_to_table_engine() -> None:
    snap = PolicySnapshot(tiers={"memory_get": "read"}, client_rules={}, global_rules={})
    chain = EngineChain([FunctionPolicyEngine("noop", lambda _: "pass")])
    res = await chain.evaluate(_policy_input(tool="memory_get", tier="read"), table_engine=TableEngine(snap))
    assert res.verdict == "allow"
    assert res.decision == "allow"
    assert res.engine == "table"


async def test_engine_timeout_and_exception_become_escalate_and_clamp_for_untrusted() -> None:
    async def slow(_: PolicyInput) -> Verdict:
        await asyncio.sleep(5.0)
        return "allow"

    async def broken(_: PolicyInput) -> Verdict:
        raise RuntimeError("judge crashed")

    snap = PolicySnapshot(tiers={"memory_get": "read"}, client_rules={}, global_rules={})

    # Timeout -> escalate for owner
    timeout_chain = EngineChain([FunctionPolicyEngine("slow_judge", slow)], timeout_s=0.02)
    res_to = await timeout_chain.evaluate(_policy_input(tool="memory_get", tier="read", trust="owner"), table_engine=TableEngine(snap))
    assert res_to.verdict == "escalate"
    assert res_to.decision == "require_approval"
    assert res_to.engine == "slow_judge"
    assert res_to.error == "timeout"

    # Timeout -> clamped to deny for untrusted
    res_to_untrusted = await timeout_chain.evaluate(
        _policy_input(tool="memory_get", tier="read", trust="untrusted"),
        table_engine=TableEngine(snap),
    )
    assert res_to_untrusted.verdict == "deny"
    assert res_to_untrusted.decision == "deny"
    assert res_to_untrusted.engine == "slow_judge"
    assert res_to_untrusted.error == "timeout"

    # Exception -> escalate for owner, deny for untrusted
    err_chain = EngineChain([FunctionPolicyEngine("broken_judge", broken)], timeout_s=1.0)
    res_err = await err_chain.evaluate(_policy_input(tool="memory_get", tier="read", trust="owner"), table_engine=TableEngine(snap))
    assert res_err.verdict == "escalate"
    assert res_err.decision == "require_approval"
    assert res_err.engine == "broken_judge"
    assert res_err.error == "RuntimeError"

    res_err_untrusted = await err_chain.evaluate(
        _policy_input(tool="memory_get", tier="read", trust="untrusted"),
        table_engine=TableEngine(snap),
    )
    assert res_err_untrusted.verdict == "deny"
    assert res_err_untrusted.decision == "deny"
    assert res_err_untrusted.engine == "broken_judge"


async def test_untrusted_clamp_applies_to_explicit_engine_escalate() -> None:
    snap = PolicySnapshot(tiers={"memory_get": "read"}, client_rules={}, global_rules={})
    chain = EngineChain([FunctionPolicyEngine("cautious", lambda _: "escalate")])

    res_owner = await chain.evaluate(_policy_input(tool="memory_get", tier="read", trust="owner"), table_engine=TableEngine(snap))
    assert res_owner.decision == "require_approval"
    assert res_owner.engine == "cautious"

    res_untrusted = await chain.evaluate(
        _policy_input(tool="memory_get", tier="read", trust="untrusted"),
        table_engine=TableEngine(snap),
    )
    assert res_untrusted.verdict == "deny"
    assert res_untrusted.decision == "deny"
    assert res_untrusted.engine == "cautious"


async def test_visibility_and_allowance_separation_in_router() -> None:
    reg = ProviderRegistry()
    prov = _StubProvider(
        "mac",
        [
            ProviderTool(name="shell_run", description="Shell"),
            ProviderTool(name="hidden_tool", description="Hidden"),
        ],
    )
    reg.register(prov)

    # Table: mac_shell_run is allowed for this client; mac_hidden_tool is denied;
    # mac_ext_tool (not denied, external tier) escalates in table.
    snap = PolicySnapshot(
        tiers={"mac_shell_run": "external", "mac_hidden_tool": "read"},
        client_rules={"mac_shell_run": "allow", "mac_hidden_tool": "deny"},
        global_rules={},
    )
    engine_invocations: list[str] = []

    async def recording_guard(inp: PolicyInput) -> Verdict:
        engine_invocations.append(inp.tool.exposed_name)
        return await deny_rm_in_shell.decide(inp)

    audits: list[tuple[str, str, str, dict[str, Any] | None]] = []

    async def record_audit(
        ctx: CallContext,
        provider_id: Any,
        tool: str,
        decision: str,
        outcome: str,
        args: dict[str, Any],
        payload: dict[str, Any] | None,
    ) -> None:
        audits.append((tool, decision, outcome, payload))

    async def load_snap(_: CallContext) -> PolicySnapshot:
        return snap

    router = ToolRouter(
        reg,
        load_policy_snapshot=load_snap,
        policy_chain=EngineChain([FunctionPolicyEngine("deny_rm_in_shell", recording_guard)]),
        record_audit=record_audit,
    )
    owner = _ctx(trust="owner")

    # Visibility hides mac_hidden_tool from list_tools
    assert visible(snap, owner, "mac_shell_run") is True
    assert visible(snap, owner, "mac_hidden_tool") is False
    listed = [t.name for t in await router.list_tools(owner)]
    assert "mac_shell_run" in listed
    assert "mac_hidden_tool" not in listed

    # Calling hidden tool is rejected at visibility layer and NEVER reaches the engine chain
    hidden_res = await router.call_tool("mac_hidden_tool", {}, owner)
    assert hidden_res.is_error
    assert engine_invocations == []
    assert audits[-1][1] == "deny"
    assert audits[-1][2] == "denied"
    assert audits[-1][3] is not None
    assert audits[-1][3]["engine"] == "visibility"

    # Guard denies mac_shell_run with rm even though table would allow it
    rm_res = await router.call_tool("mac_shell_run", {"command": "rm -rf /tmp/x"}, owner)
    assert rm_res.is_error
    assert engine_invocations == ["mac_shell_run"]
    assert prov.calls == []
    assert audits[-1][1] == "deny"
    assert audits[-1][2] == "denied"
    assert audits[-1][3] is not None
    assert audits[-1][3]["engine"] == "deny_rm_in_shell"

    # Safe command passes guard and falls through to TableEngine (allow)
    ok_res = await router.call_tool("mac_shell_run", {"command": "echo farm"}, owner)
    assert not ok_res.is_error
    assert prov.calls == [("shell_run", {"command": "echo farm"})]
    assert audits[-1][1] == "allow"
    assert audits[-1][2] == "ok"
    assert audits[-1][3] is not None
    assert audits[-1][3]["engine"] == "table"


async def test_engine_can_allow_tool_table_would_escalate() -> None:
    """An allowance engine (like a future LLM judge) can allow a tool the table would escalate."""
    reg = ProviderRegistry()
    prov = _StubProvider("mac", [ProviderTool(name="shell_run", description="Shell")])
    reg.register(prov)

    # External tier -> table defaults to require_approval (escalate), visible to owner
    snap = PolicySnapshot(tiers={"mac_shell_run": "external"}, client_rules={}, global_rules={})
    confirm_called = False

    async def confirm(*_args: Any) -> tuple[bool, str | None]:
        nonlocal confirm_called
        confirm_called = True
        return False, "should not be called"

    audits: list[tuple[str, str, str, dict[str, Any] | None]] = []

    async def record_audit(
        ctx: CallContext,
        provider_id: Any,
        tool: str,
        decision: str,
        outcome: str,
        args: dict[str, Any],
        payload: dict[str, Any] | None,
    ) -> None:
        audits.append((tool, decision, outcome, payload))

    async def judge(inp: PolicyInput) -> Verdict:
        if inp.tool.exposed_name == "mac_shell_run" and inp.args.get("command") == "uname -a":
            return "allow"
        return "pass"

    async def load_snap(_: CallContext) -> PolicySnapshot:
        return snap

    router = ToolRouter(
        reg,
        load_policy_snapshot=load_snap,
        policy_chain=EngineChain([FunctionPolicyEngine("mock_judge", judge)]),
        confirm_call=confirm,
        record_audit=record_audit,
    )
    owner = _ctx(trust="owner")
    untrusted = _ctx(trust="untrusted")

    # Owner: tool is visible (table=require_approval != deny), judge allows it without escalation!
    res = await router.call_tool("mac_shell_run", {"command": "uname -a"}, owner)
    assert not res.is_error
    assert confirm_called is False
    assert prov.calls == [("shell_run", {"command": "uname -a"})]
    assert audits[-1][1] == "allow"
    assert audits[-1][3] is not None
    assert audits[-1][3]["engine"] == "mock_judge"

    # Untrusted: external tool is hidden at visibility layer and NEVER reaches judge!
    assert visible(snap, untrusted, "mac_shell_run") is False
    res_u = await router.call_tool("mac_shell_run", {"command": "uname -a"}, untrusted)
    assert res_u.is_error
    assert audits[-1][1] == "deny"
    assert audits[-1][3] is not None
    assert audits[-1][3]["engine"] == "visibility"
