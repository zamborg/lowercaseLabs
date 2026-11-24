"""Immutable request authentication contexts."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

TrustCeiling = Literal["owner", "untrusted"]
ClientKind = Literal["oauth", "bearer"]


@dataclass(frozen=True, slots=True)
class CallContext:
    """Immutable identity & trust context attached to every authenticated MCP call."""

    principal_id: UUID
    principal_slug: str
    space_id: UUID
    space_slug: str
    client_id: UUID
    client_name: str
    client_kind: ClientKind
    trust_ceiling: TrustCeiling
    token_id: UUID
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActuatorAuthContext:
    """Immutable auth context for a connected dial-out actuator."""

    principal_id: UUID
    space_id: UUID
    provider_id: UUID
    provider_slug: str
    token_id: UUID


current_call_context: ContextVar[CallContext | None] = ContextVar(
    "zos_current_call_context", default=None
)
