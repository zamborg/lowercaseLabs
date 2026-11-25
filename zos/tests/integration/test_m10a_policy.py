"""Integration tests for M10a: engine chain wiring, config parsing, and audit payload `engine`."""

from __future__ import annotations

import uuid
from typing import Any

import mcp_types as types
import pytest

from tests.conftest import get_test_database_url
from zos.admin_service import AdminService
from zos.app import create_app
from zos.auth.context import CallContext
from zos.config import Settings
from zos.db.queries import init_principal_and_space
from zos.policy import FunctionPolicyEngine, PolicyInput, Verdict
from zos.policy.guards import deny_rm_in_shell
from zos.providers.base import ProviderKind, ProviderTool


class _ShellStubProvider:
    def __init__(self) -> None:
        self.provider_id = None
        self.calls: list[str] = []

    @property
    def namespace(self) -> str:
        return "stub"

    @property
    def kind(self) -> ProviderKind:
        return "hosted"

    def is_online(self) -> bool:
        return True

    async def list_tools(self) -> list[ProviderTool]:
        return [ProviderTool(name="shell_run", description="Stub shell")]

    async def call_tool(self, name: str, args: dict[str, Any], ctx: CallContext) -> types.CallToolResult:
        self.calls.append(str(args.get("command", "")))
        return types.CallToolResult(content=[types.TextContent(type="text", text="ran")])


def test_settings_from_env_parses_policy_engine_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://zos:zos@localhost:5432/zos")
    monkeypatch.setenv("ZOS_POLICY_ENGINES", "zos.policy.guards:deny_rm_in_shell , ")
    monkeypatch.setenv("ZOS_POLICY_ENGINE_TIMEOUT_S", "2.5")
    cfg = Settings.from_env(load_env_file=False)
    assert cfg.policy_engines == ("zos.policy.guards:deny_rm_in_shell",)
    assert cfg.policy_engine_timeout_seconds == 2.5

    monkeypatch.delenv("ZOS_POLICY_ENGINES", raising=False)
    monkeypatch.delenv("ZOS_POLICY_ENGINE_TIMEOUT_S", raising=False)
    default_cfg = Settings.from_env(load_env_file=False)
    assert default_cfg.policy_engines == ()
    assert default_cfg.policy_engine_timeout_seconds == 10.0


async def test_audit_events_record_engine_name_across_layers(db_pool: Any) -> None:
    db_url = get_test_database_url()
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    principal_id = init["principal"]["id"]
    space_id = init["space"]["id"]

    svc = AdminService(db_pool)
    reg = await svc.register_bearer_client(name="m10a-client", trust_ceiling="owner")
    client_id = reg["client"]["id"]
    # Allow stub_shell_run in table so guard can override it on rm
    await svc.set_policy(pattern="stub_shell_run", decision="allow", client_identifier="m10a-client")
    # Explicitly deny memory_delete so visibility hides it
    await svc.set_policy(pattern="memory_delete", decision="deny", client_identifier="m10a-client")

    stub = _ShellStubProvider()
    seen_by_judge: list[str] = []

    async def judge(inp: PolicyInput) -> Verdict:
        seen_by_judge.append(inp.tool.exposed_name)
        if inp.tool.exposed_name == "stub_shell_run" and inp.args.get("command") == "safe-judge":
            return "allow"
        return "pass"

    cfg = Settings(
        database_url=db_url,
        policy_engines=("zos.policy.guards:deny_rm_in_shell",),
        policy_engine_timeout_seconds=5.0,
    )
    app = create_app(
        cfg,
        extra_providers=[stub],
        policy_engines=[deny_rm_in_shell, FunctionPolicyEngine("judge_v1", judge)],
    )

    ctx = CallContext(
        principal_id=principal_id,
        principal_slug="owner",
        space_id=space_id,
        space_slug="main",
        client_id=client_id,
        client_name="m10a-client",
        client_kind="bearer",
        trust_ceiling="owner",
        token_id=uuid.uuid4(),
    )

    async with app.router.lifespan_context(app):
        rt = app.state.zos
        # 1) Hidden tool -> rejected by visibility, never hits judge
        r_hidden = await rt.router.call_tool("memory_delete", {"key": "k"}, ctx)
        assert r_hidden.is_error
        assert "memory_delete" not in seen_by_judge

        # 2) Guard denies rm command even though table allows stub_shell_run
        r_rm = await rt.router.call_tool("stub_shell_run", {"command": "rm -rf /tmp/test"}, ctx)
        assert r_rm.is_error
        assert stub.calls == []

        # 3) Judge explicitly allows safe-judge
        r_judge = await rt.router.call_tool("stub_shell_run", {"command": "safe-judge"}, ctx)
        assert not r_judge.is_error

        # 4) Ordinary command falls through to TableEngine
        r_table = await rt.router.call_tool("stub_shell_run", {"command": "echo farm"}, ctx)
        assert not r_table.is_error

        async with rt.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """
                    SELECT tool, decision, outcome, payload
                    FROM events
                    WHERE type = 'tool.called' AND client_id = %s
                    ORDER BY id ASC
                    """,
                    (client_id,),
                )
            ).fetchall()

    assert [(r["tool"], r["decision"], r["outcome"], r["payload"]["engine"]) for r in rows] == [
        ("memory_delete", "deny", "denied", "visibility"),
        ("stub_shell_run", "deny", "denied", "deny_rm_in_shell"),
        ("stub_shell_run", "allow", "ok", "judge_v1"),
        ("stub_shell_run", "allow", "ok", "table"),
    ]
