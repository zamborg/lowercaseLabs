"""Tool router: merges online provider tools, applies policy/confirmation, routes calls, audits."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

import mcp_types as types
from mcp.server.session import ServerSession

from zos.auth.context import CallContext
from zos.policy.decisions import PolicyEvaluation, PolicySnapshot
from zos.policy.engine import (
    EngineChain,
    PolicyClient,
    PolicyInput,
    PolicyTool,
    TableEngine,
    visible,
)
from zos.policy.tiers import Tier
from zos.providers.base import (
    DownstreamMCPContext,
    ProviderTool,
    ToolProvider,
    current_downstream_mcp,
)
from zos.router.naming import format_tool_name, split_prefix
from zos.router.registry import ProviderRegistry
from zos.router.sessions import SessionTracker
from zos.router.status import ZOS_STATUS_TOOL_NAME, render_status_result, zos_status_tool

logger = logging.getLogger(__name__)


PolicyEvaluator = Callable[
    [CallContext, str, bool | None, bool | None],
    Awaitable[PolicyEvaluation],
]
PolicySnapshotLoader = Callable[[CallContext], Awaitable[PolicySnapshot]]
ConfirmationHandler = Callable[
    [ServerSession | None, Any, CallContext, str, dict[str, Any]],
    Awaitable[tuple[bool, str | None]],
]
AuditRecorder = Callable[
    [CallContext, UUID | None, str, str, str, dict[str, Any], dict[str, Any] | None],
    Awaitable[None],
]


class ToolRouter:
    """Merges tools from online providers, enforces policy & confirmation, and routes calls."""

    def __init__(
        self,
        registry: ProviderRegistry,
        sessions: SessionTracker | None = None,
        *,
        evaluate_policy: PolicyEvaluator | None = None,
        load_policy_snapshot: PolicySnapshotLoader | None = None,
        policy_chain: EngineChain | None = None,
        confirm_call: ConfirmationHandler | None = None,
        record_audit: AuditRecorder | None = None,
    ) -> None:
        self.registry = registry
        self.sessions = sessions or SessionTracker()
        self._evaluate_policy = evaluate_policy
        self._load_policy_snapshot = load_policy_snapshot
        self._policy_chain = policy_chain or EngineChain()
        self._confirm_call = confirm_call
        self._record_audit = record_audit

    async def _load_snapshot(self, ctx: CallContext) -> PolicySnapshot | None:
        if self._load_policy_snapshot is None:
            return None
        return await self._load_policy_snapshot(ctx)

    async def _eval_policy(
        self,
        ctx: CallContext,
        exposed_name: str,
        *,
        destructive_hint: bool | None = None,
        idempotent_hint: bool | None = None,
        snapshot: PolicySnapshot | None = None,
    ) -> PolicyEvaluation:
        if snapshot is not None:
            return snapshot.evaluate(
                exposed_name,
                trust_ceiling=ctx.trust_ceiling,
                destructive_hint=destructive_hint,
                idempotent_hint=idempotent_hint,
            )
        if self._load_policy_snapshot is not None:
            snap = await self._load_policy_snapshot(ctx)
            return snap.evaluate(
                exposed_name,
                trust_ceiling=ctx.trust_ceiling,
                destructive_hint=destructive_hint,
                idempotent_hint=idempotent_hint,
            )
        if self._evaluate_policy is not None:
            return await self._evaluate_policy(ctx, exposed_name, destructive_hint, idempotent_hint)
        is_read = exposed_name == ZOS_STATUS_TOOL_NAME
        resolved_destructive = False if is_read else (True if destructive_hint is None else bool(destructive_hint))
        return PolicyEvaluation(
            tier="read" if is_read else "external",
            decision="allow",
            annotations=types.ToolAnnotations(
                read_only_hint=is_read,
                destructive_hint=resolved_destructive,
                idempotent_hint=idempotent_hint,
                open_world_hint=not is_read,
            ),
        )

    async def _check_visibility(
        self,
        snapshot: PolicySnapshot | None,
        ctx: CallContext,
        exposed_name: str,
        *,
        destructive_hint: bool | None = None,
        idempotent_hint: bool | None = None,
    ) -> tuple[bool, PolicyEvaluation]:
        """Layer 1: static, per-client, argument-free visibility check."""
        if snapshot is not None:
            is_visible = visible(snapshot, ctx, exposed_name)
            ev = snapshot.evaluate(
                exposed_name,
                trust_ceiling=ctx.trust_ceiling,
                destructive_hint=destructive_hint,
                idempotent_hint=idempotent_hint,
            )
            return is_visible, ev
        ev = await self._eval_policy(
            ctx,
            exposed_name,
            destructive_hint=destructive_hint,
            idempotent_hint=idempotent_hint,
        )
        return ev.decision != "deny", ev

    async def _evaluate_allowance(
        self,
        snapshot: PolicySnapshot | None,
        ctx: CallContext,
        exposed_name: str,
        *,
        tier: Tier,
        args: dict[str, Any],
        destructive_hint: bool | None = None,
        idempotent_hint: bool | None = None,
        fallback_eval: PolicyEvaluation,
    ) -> tuple[str, str, str | None]:
        """Layer 2: per-call allowance evaluation via `EngineChain`."""
        if snapshot is None:
            return fallback_eval.decision, "table", None
        policy_input = PolicyInput(
            principal_id=ctx.principal_id,
            client=PolicyClient(
                id=ctx.client_id,
                name=ctx.client_name,
                trust_ceiling=ctx.trust_ceiling,
            ),
            tool=PolicyTool(
                exposed_name=exposed_name,
                tier=tier,
                destructive_hint=destructive_hint,
                idempotent_hint=idempotent_hint,
            ),
            args=args,
        )
        chain_decision = await self._policy_chain.evaluate(
            policy_input,
            table_engine=TableEngine(snapshot),
        )
        return chain_decision.decision, chain_decision.engine, chain_decision.error

    @staticmethod
    def _audit_payload(
        *,
        tier: str,
        engine: str,
        engine_error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"tier": tier, "engine": engine}
        if engine_error is not None:
            payload["engine_error"] = engine_error
        if extra:
            payload.update(extra)
        return payload

    async def _audit(
        self,
        ctx: CallContext,
        provider_id: UUID | None,
        tool: str,
        decision: str,
        outcome: str,
        args: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._record_audit is not None:
            await self._record_audit(ctx, provider_id, tool, decision, outcome, args, payload)

    async def _build_tool_map(
        self,
    ) -> dict[str, tuple[ToolProvider, ProviderTool]]:
        mapping: dict[str, tuple[ToolProvider, ProviderTool]] = {}
        for provider in self.registry.online_providers():
            try:
                provider_tools = await provider.list_tools()
            except Exception:
                logger.exception("Failed to list tools from provider %s", provider.namespace)
                continue
            for ptool in provider_tools:
                exposed = format_tool_name(provider.namespace, ptool.name)
                if exposed in mapping:
                    logger.warning(
                        "Rejecting colliding tool %r from provider %s (already registered by %s)",
                        exposed,
                        provider.namespace,
                        mapping[exposed][0].namespace,
                    )
                    continue
                mapping[exposed] = (provider, ptool)
        return mapping

    async def list_tools(self, ctx: CallContext) -> list[types.Tool]:
        tools: list[types.Tool] = []
        snapshot = await self._load_snapshot(ctx)

        # Layer 1 (Visibility): always-on zos_status unless hidden by policy
        status_visible, status_eval = await self._check_visibility(
            snapshot, ctx, ZOS_STATUS_TOOL_NAME, destructive_hint=False
        )
        if status_visible:
            tools.append(zos_status_tool(annotations=status_eval.annotations))

        tool_map = await self._build_tool_map()
        for exposed_name in sorted(tool_map):
            _provider, ptool = tool_map[exposed_name]
            is_visible, ev = await self._check_visibility(
                snapshot,
                ctx,
                exposed_name,
                destructive_hint=ptool.destructive_hint,
                idempotent_hint=ptool.idempotent_hint,
            )
            if not is_visible:
                continue
            tools.append(
                types.Tool(
                    name=exposed_name,
                    description=ptool.description,
                    input_schema=ptool.input_schema,
                    annotations=ev.annotations,
                )
            )
        return tools

    async def _enforce_confirmation(
        self,
        *,
        ctx: CallContext,
        provider_id: UUID | None,
        exposed_name: str,
        decision: str,
        tier: str,
        engine_name: str,
        engine_error: str | None,
        call_args: dict[str, Any],
        server_session: ServerSession | None,
        request_id: Any,
    ) -> types.CallToolResult | None:
        """Run human confirmation flow for `require_approval`; return error result if not approved."""
        if self._confirm_call is None:
            msg = (
                f"Approval required for '{exposed_name}', but confirmation is unavailable. "
                f"To override, run: zos policy set --client {ctx.client_name} {exposed_name} allow"
            )
            await self._audit(
                ctx,
                provider_id,
                exposed_name,
                decision,
                "denied",
                call_args,
                self._audit_payload(
                    tier=tier,
                    engine=engine_name,
                    engine_error=engine_error,
                    extra={"reason": "no_confirmation_handler"},
                ),
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=msg)],
                is_error=True,
            )

        approved, reason = await self._confirm_call(
            server_session, request_id, ctx, exposed_name, call_args
        )
        if not approved:
            await self._audit(
                ctx,
                provider_id,
                exposed_name,
                decision,
                "denied",
                call_args,
                self._audit_payload(
                    tier=tier,
                    engine=engine_name,
                    engine_error=engine_error,
                    extra={"reason": reason or "user_declined"},
                ),
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=reason or f"Call to '{exposed_name}' was not approved.")],
                is_error=True,
            )
        return None

    async def call_tool(
        self,
        exposed_name: str,
        args: dict[str, Any] | None,
        ctx: CallContext,
        *,
        server_session: ServerSession | None = None,
        request_id: Any = None,
    ) -> types.CallToolResult:
        call_args = dict(args) if args else {}

        # Router-owned zos_status
        if exposed_name == ZOS_STATUS_TOOL_NAME:
            snapshot = await self._load_snapshot(ctx)
            is_visible, vis_eval = await self._check_visibility(
                snapshot, ctx, ZOS_STATUS_TOOL_NAME, destructive_hint=False
            )
            if not is_visible:
                await self._audit(
                    ctx,
                    None,
                    exposed_name,
                    "deny",
                    "denied",
                    call_args,
                    self._audit_payload(
                        tier=vis_eval.tier,
                        engine="visibility",
                        extra={"reason": "policy_deny"},
                    ),
                )
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Permission denied for tool {exposed_name}.")],
                    is_error=True,
                )
            decision, engine_name, engine_error = await self._evaluate_allowance(
                snapshot,
                ctx,
                ZOS_STATUS_TOOL_NAME,
                tier=vis_eval.tier,
                args=call_args,
                destructive_hint=False,
                fallback_eval=vis_eval,
            )
            if decision == "deny":
                await self._audit(
                    ctx,
                    None,
                    exposed_name,
                    decision,
                    "denied",
                    call_args,
                    self._audit_payload(
                        tier=vis_eval.tier,
                        engine=engine_name,
                        engine_error=engine_error,
                        extra={"reason": "policy_deny"},
                    ),
                )
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Permission denied for tool {exposed_name}.")],
                    is_error=True,
                )
            if decision == "require_approval":
                denied = await self._enforce_confirmation(
                    ctx=ctx,
                    provider_id=None,
                    exposed_name=exposed_name,
                    decision=decision,
                    tier=vis_eval.tier,
                    engine_name=engine_name,
                    engine_error=engine_error,
                    call_args=call_args,
                    server_session=server_session,
                    request_id=request_id,
                )
                if denied is not None:
                    return denied
            result = render_status_result(self.registry)
            await self._audit(
                ctx,
                None,
                exposed_name,
                decision,
                "ok",
                call_args,
                self._audit_payload(
                    tier=vis_eval.tier,
                    engine=engine_name,
                    engine_error=engine_error,
                ),
            )
            return result

        # Check offline provider before declaring unknown
        tool_map = await self._build_tool_map()
        if exposed_name not in tool_map:
            split = split_prefix(exposed_name)
            if split is not None:
                prefix, _ = split
                provider = self.registry.get(prefix)
                if provider is not None and not provider.is_online():
                    if getattr(provider, "needs_reconnect", False):
                        hint = (
                            provider._reconnect_hint()
                            if hasattr(provider, "_reconnect_hint")
                            else f"Reconnect provider '{prefix}'."
                        )
                        code = "UPSTREAM_NEEDS_RECONNECT"
                        msg = f"{code}: provider '{prefix}' requires reconnection. {hint}"
                    else:
                        code = "ACTUATOR_OFFLINE" if provider.kind == "actuator" else "PROVIDER_OFFLINE"
                        msg = f"{code}: provider '{prefix}' is currently offline."
                    await self._audit(
                        ctx,
                        provider.provider_id,
                        exposed_name,
                        "allow",
                        "error",
                        call_args,
                        {"error_code": code},
                    )
                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=msg)],
                        is_error=True,
                    )
            await self._audit(ctx, None, exposed_name, "deny", "error", call_args, {"error_code": "UNKNOWN_TOOL"})
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown tool: {exposed_name}")],
                is_error=True,
            )

        provider, ptool = tool_map[exposed_name]
        snapshot = await self._load_snapshot(ctx)

        # Layer 1 (Visibility): reject hidden tools before any allowance engine runs
        is_visible, vis_eval = await self._check_visibility(
            snapshot,
            ctx,
            exposed_name,
            destructive_hint=ptool.destructive_hint,
            idempotent_hint=ptool.idempotent_hint,
        )
        if not is_visible:
            await self._audit(
                ctx,
                provider.provider_id,
                exposed_name,
                "deny",
                "denied",
                call_args,
                self._audit_payload(
                    tier=vis_eval.tier,
                    engine="visibility",
                    extra={"reason": "policy_deny"},
                ),
            )
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=(
                            f"Permission denied for tool '{exposed_name}'. "
                            f"To allow, run: zos policy set --client {ctx.client_name} {exposed_name} allow"
                        ),
                    )
                ],
                is_error=True,
            )

        # Layer 2 (Allowance): evaluate configured policy engines + TableEngine
        decision, engine_name, engine_error = await self._evaluate_allowance(
            snapshot,
            ctx,
            exposed_name,
            tier=vis_eval.tier,
            args=call_args,
            destructive_hint=ptool.destructive_hint,
            idempotent_hint=ptool.idempotent_hint,
            fallback_eval=vis_eval,
        )

        if decision == "deny":
            await self._audit(
                ctx,
                provider.provider_id,
                exposed_name,
                decision,
                "denied",
                call_args,
                self._audit_payload(
                    tier=vis_eval.tier,
                    engine=engine_name,
                    engine_error=engine_error,
                    extra={"reason": "policy_deny"},
                ),
            )
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=(
                            f"Permission denied for tool '{exposed_name}'. "
                            f"To allow, run: zos policy set --client {ctx.client_name} {exposed_name} allow"
                        ),
                    )
                ],
                is_error=True,
            )

        if decision == "require_approval":
            denied = await self._enforce_confirmation(
                ctx=ctx,
                provider_id=provider.provider_id,
                exposed_name=exposed_name,
                decision=decision,
                tier=vis_eval.tier,
                engine_name=engine_name,
                engine_error=engine_error,
                call_args=call_args,
                server_session=server_session,
                request_id=request_id,
            )
            if denied is not None:
                return denied

        # Execute tool on provider (exposing downstream MCP session via ContextVar for elicitation passthrough)
        downstream_token = current_downstream_mcp.set(
            DownstreamMCPContext(
                server_session=server_session,
                request_id=request_id,
                call_ctx=ctx,
                exposed_tool=exposed_name,
            )
        )
        try:
            result = await provider.call_tool(ptool.name, call_args, ctx)
        except Exception as exc:
            logger.exception("Provider %s tool %s raised", provider.namespace, ptool.name)
            await self._audit(
                ctx,
                provider.provider_id,
                exposed_name,
                decision,
                "error",
                call_args,
                self._audit_payload(
                    tier=vis_eval.tier,
                    engine=engine_name,
                    engine_error=engine_error,
                    extra={"exception": type(exc).__name__},
                ),
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Tool execution failed: {exc}")],
                is_error=True,
            )
        finally:
            current_downstream_mcp.reset(downstream_token)

        outcome = "error" if result.is_error else "ok"
        await self._audit(
            ctx,
            provider.provider_id,
            exposed_name,
            decision,
            outcome,
            call_args,
            self._audit_payload(
                tier=vis_eval.tier,
                engine=engine_name,
                engine_error=engine_error,
            ),
        )
        return result
