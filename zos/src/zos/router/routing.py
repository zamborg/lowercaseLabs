"""Tool router: merges online provider tools, applies policy/confirmation, routes calls, audits."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import mcp_types as types
from mcp.server.session import ServerSession

from zos.auth.context import CallContext
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


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    tier: str
    decision: str
    annotations: types.ToolAnnotations


PolicyEvaluator = Callable[
    [CallContext, str, bool | None, bool | None],
    Awaitable[PolicyEvaluation],
]
PolicySnapshotLoader = Callable[[CallContext], Awaitable[Any]]
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
        confirm_call: ConfirmationHandler | None = None,
        record_audit: AuditRecorder | None = None,
    ) -> None:
        self.registry = registry
        self.sessions = sessions or SessionTracker()
        self._evaluate_policy = evaluate_policy
        self._load_policy_snapshot = load_policy_snapshot
        self._confirm_call = confirm_call
        self._record_audit = record_audit

    async def _eval_policy(
        self,
        ctx: CallContext,
        exposed_name: str,
        *,
        destructive_hint: bool | None = None,
        idempotent_hint: bool | None = None,
        snapshot: Any = None,
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
        snapshot = (
            await self._load_policy_snapshot(ctx) if self._load_policy_snapshot is not None else None
        )

        # Always-on zos_status (unless explicitly denied by policy)
        status_eval = await self._eval_policy(
            ctx, ZOS_STATUS_TOOL_NAME, destructive_hint=False, snapshot=snapshot
        )
        if status_eval.decision != "deny":
            tools.append(zos_status_tool(annotations=status_eval.annotations))

        tool_map = await self._build_tool_map()
        for exposed_name in sorted(tool_map):
            _provider, ptool = tool_map[exposed_name]
            ev = await self._eval_policy(
                ctx,
                exposed_name,
                destructive_hint=ptool.destructive_hint,
                idempotent_hint=ptool.idempotent_hint,
                snapshot=snapshot,
            )
            if ev.decision == "deny":
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
            ev = await self._eval_policy(ctx, ZOS_STATUS_TOOL_NAME, destructive_hint=False)
            if ev.decision == "deny":
                await self._audit(ctx, None, exposed_name, ev.decision, "denied", call_args, {"reason": "policy_deny"})
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Permission denied for tool {exposed_name}.")],
                    is_error=True,
                )
            result = render_status_result(self.registry)
            await self._audit(ctx, None, exposed_name, ev.decision, "ok", call_args, None)
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
        ev = await self._eval_policy(
            ctx,
            exposed_name,
            destructive_hint=ptool.destructive_hint,
            idempotent_hint=ptool.idempotent_hint,
        )

        if ev.decision == "deny":
            await self._audit(
                ctx,
                provider.provider_id,
                exposed_name,
                ev.decision,
                "denied",
                call_args,
                {"reason": "policy_deny", "tier": ev.tier},
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

        if ev.decision == "require_approval":
            if self._confirm_call is None:
                msg = (
                    f"Approval required for '{exposed_name}', but confirmation is unavailable. "
                    f"To override, run: zos policy set --client {ctx.client_name} {exposed_name} allow"
                )
                await self._audit(
                    ctx,
                    provider.provider_id,
                    exposed_name,
                    ev.decision,
                    "denied",
                    call_args,
                    {"reason": "no_confirmation_handler", "tier": ev.tier},
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
                    provider.provider_id,
                    exposed_name,
                    ev.decision,
                    "denied",
                    call_args,
                    {"reason": reason or "user_declined", "tier": ev.tier},
                )
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=reason or f"Call to '{exposed_name}' was not approved.")],
                    is_error=True,
                )

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
                ev.decision,
                "error",
                call_args,
                {"exception": type(exc).__name__},
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
            ev.decision,
            outcome,
            call_args,
            {"tier": ev.tier},
        )
        return result
