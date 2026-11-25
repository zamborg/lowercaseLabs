"""Two-layer policy model: static visibility + per-call allowance engine chain.

Layer 1 — Visibility (`visible`):
    Static, per-client, argument-free check backed by the policy table snapshot.
    Determines which tools appear in `tools/list`. Hidden tools are rejected at
    call time before any allowance engine runs.

Layer 2 — Allowance (`EngineChain`):
    Per-call evaluation that receives tool metadata and call arguments via
    `PolicyInput`. Configured `PolicyEngine` instances run in order; the first
    non-`pass` verdict wins. `TableEngine` (the existing tier + rule lookup) is
    always appended last and never returns `pass`.

Core invariants enforced by `EngineChain`:
    - Any engine exception or timeout (`ZOS_POLICY_ENGINE_TIMEOUT_S`) becomes `escalate`.
    - After the chain, `trust_ceiling == "untrusted"` clamps `escalate` to `deny`.
    - Every decision records the deciding engine's name in the audit payload.

Plugging in a future LLM judge:
    An LLM judge plugs in as a `PolicyEngine` configured via `ZOS_POLICY_ENGINES`
    (e.g. `my_pkg.judge:llm_judge`). Inside `decide(policy_input)`, the judge can
    invoke a ZOS module tool (or remote MCP endpoint) with the summarized call
    context and return `"allow"`, `"escalate"`, `"deny"`, or `"pass"`. Because
    `EngineChain` enforces `ZOS_POLICY_ENGINE_TIMEOUT_S` and converts any error
    or timeout into `"escalate"`, a slow or unavailable judge fails safe to human
    approval (or `"deny"` for untrusted clients).
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from zos.auth.context import CallContext, TrustCeiling
from zos.policy.decisions import Decision, PolicySnapshot, resolve_decision
from zos.policy.tiers import Tier, resolve_tier

logger = logging.getLogger(__name__)

Verdict = Literal["allow", "escalate", "deny", "pass"]
FinalVerdict = Literal["allow", "escalate", "deny"]

_FINAL_VERDICTS = frozenset({"allow", "escalate", "deny"})


@dataclass(frozen=True, slots=True)
class PolicyClient:
    """Client identity and trust ceiling passed into allowance engines."""

    id: UUID
    name: str
    trust_ceiling: TrustCeiling


@dataclass(frozen=True, slots=True)
class PolicyTool:
    """Tool metadata passed into allowance engines."""

    exposed_name: str
    tier: Tier
    destructive_hint: bool | None = None
    idempotent_hint: bool | None = None


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """Immutable per-call input passed to each `PolicyEngine`."""

    principal_id: UUID
    client: PolicyClient
    tool: PolicyTool
    args: Mapping[str, Any]


@runtime_checkable
class PolicyEngine(Protocol):
    """Pluggable per-call allowance engine."""

    name: str

    async def decide(self, policy_input: PolicyInput) -> Verdict: ...


@dataclass(frozen=True, slots=True)
class ChainDecision:
    """Resolved allowance outcome from `EngineChain`."""

    verdict: FinalVerdict
    decision: Decision
    engine: str
    error: str | None = None


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return copy.deepcopy(value)


def freeze_policy_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a read-only deep copy of tool arguments so policy engines cannot mutate call args."""
    if not args:
        return MappingProxyType({})
    return MappingProxyType({str(k): _freeze_value(v) for k, v in args.items()})


def verdict_to_decision(verdict: FinalVerdict) -> Decision:
    """Map an allowance verdict onto ZOS's existing enforcement/audit decision values."""
    if verdict == "escalate":
        return "require_approval"
    return verdict


def visible(
    snapshot: PolicySnapshot,
    client: CallContext | PolicyClient,
    tool: str,
) -> bool:
    """Layer 1 visibility check: static, per-client, argument-free.

    A tool is visible iff its policy-table decision (with no arguments) is not `deny`.
    Because `resolve_decision` downgrades `require_approval` -> `deny` for untrusted
    clients, untrusted clients continue to hide external tools automatically.
    """
    tier = resolve_tier(tool, snapshot.tiers)
    decision = resolve_decision(
        tool,
        tier=tier,
        trust_ceiling=client.trust_ceiling,
        client_rules=snapshot.client_rules,
        global_rules=snapshot.global_rules,
    )
    return decision != "deny"


class TableEngine:
    """Fallback allowance engine backed by `PolicySnapshot` (never returns `pass`)."""

    name: str = "table"

    def __init__(self, snapshot: PolicySnapshot) -> None:
        self.snapshot = snapshot

    async def decide(self, policy_input: PolicyInput) -> Verdict:
        decision = resolve_decision(
            policy_input.tool.exposed_name,
            tier=policy_input.tool.tier,
            trust_ceiling=policy_input.client.trust_ceiling,
            client_rules=self.snapshot.client_rules,
            global_rules=self.snapshot.global_rules,
        )
        if decision == "require_approval":
            return "escalate"
        return decision


class FunctionPolicyEngine:
    """Adapter turning a sync or async callable `(PolicyInput) -> Verdict` into a `PolicyEngine`."""

    def __init__(
        self,
        name: str,
        fn: Callable[[PolicyInput], Verdict | Awaitable[Verdict]],
    ) -> None:
        self.name = name
        self._fn = fn

    async def decide(self, policy_input: PolicyInput) -> Verdict:
        result = self._fn(policy_input)
        if inspect.isawaitable(result):
            return await result
        return result

    async def __call__(self, policy_input: PolicyInput) -> Verdict:
        return await self.decide(policy_input)


class EngineChain:
    """Runs configured `PolicyEngine`s in order, ending with `TableEngine`."""

    def __init__(
        self,
        engines: Sequence[PolicyEngine] = (),
        *,
        timeout_s: float = 10.0,
    ) -> None:
        self.engines: tuple[PolicyEngine, ...] = tuple(engines)
        self.timeout_s = timeout_s

    async def evaluate(
        self,
        policy_input: PolicyInput,
        *,
        table_engine: TableEngine,
    ) -> ChainDecision:
        verdict: FinalVerdict | None = None
        deciding_engine = table_engine.name
        error: str | None = None

        for engine in (*self.engines, table_engine):
            engine_name = getattr(engine, "name", type(engine).__name__)
            try:
                raw_verdict = await asyncio.wait_for(
                    engine.decide(policy_input),
                    timeout=self.timeout_s,
                )
            except TimeoutError:
                logger.warning(
                    "Policy engine %r timed out after %.2fs for tool %s; escalating",
                    engine_name,
                    self.timeout_s,
                    policy_input.tool.exposed_name,
                )
                verdict = "escalate"
                deciding_engine = engine_name
                error = "timeout"
                break
            except Exception as exc:
                logger.exception(
                    "Policy engine %r raised %s for tool %s; escalating",
                    engine_name,
                    type(exc).__name__,
                    policy_input.tool.exposed_name,
                )
                verdict = "escalate"
                deciding_engine = engine_name
                error = type(exc).__name__
                break

            if raw_verdict == "pass":
                continue

            if raw_verdict not in _FINAL_VERDICTS:
                logger.warning(
                    "Policy engine %r returned invalid verdict %r for tool %s; escalating",
                    engine_name,
                    raw_verdict,
                    policy_input.tool.exposed_name,
                )
                verdict = "escalate"
                deciding_engine = engine_name
                error = "invalid_verdict"
                break

            verdict = raw_verdict
            deciding_engine = engine_name
            break

        if verdict is None:
            verdict = "escalate"
            deciding_engine = table_engine.name
            error = "no_verdict"

        # Core invariant: untrusted clients never escalate (escalate -> deny).
        if policy_input.client.trust_ceiling == "untrusted" and verdict == "escalate":
            verdict = "deny"

        return ChainDecision(
            verdict=verdict,
            decision=verdict_to_decision(verdict),
            engine=deciding_engine,
            error=error,
        )


def load_policy_engine(spec: str) -> PolicyEngine:
    """Import a `PolicyEngine` or callable from a `module:attr` specification."""
    raw = spec.strip()
    if ":" not in raw:
        raise ValueError(f"Invalid policy engine spec {spec!r}; expected 'module:attr'")
    module_name, attr_path = raw.split(":", 1)
    if not module_name or not attr_path:
        raise ValueError(f"Invalid policy engine spec {spec!r}; expected 'module:attr'")

    mod = importlib.import_module(module_name)
    obj: Any = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)

    if isinstance(obj, type):
        obj = obj()

    if isinstance(obj, PolicyEngine):
        return obj

    if callable(obj):
        engine_name = getattr(obj, "name", None) or getattr(obj, "__name__", None) or raw
        return FunctionPolicyEngine(name=str(engine_name), fn=obj)

    raise TypeError(
        f"Policy engine {spec!r} must implement PolicyEngine or be callable, got {type(obj).__name__}"
    )


def load_policy_engines(specs: Sequence[str]) -> list[PolicyEngine]:
    """Load an ordered list of policy engines from `module:attr` specs."""
    return [load_policy_engine(spec) for spec in specs if spec.strip()]
