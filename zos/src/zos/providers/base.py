"""Core ToolProvider protocol — the single seam behind the ZOS router."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

import mcp_types as types

from zos.auth.context import CallContext

ProviderKind = Literal["hosted", "upstream", "actuator"]


@dataclass(frozen=True, slots=True)
class ProviderTool:
    """A tool advertised by a ToolProvider before ZOS namespacing and tier annotations."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    destructive_hint: bool = False
    idempotent_hint: bool | None = None


@runtime_checkable
class ToolProvider(Protocol):
    """Single provider seam behind the router: hosted, upstream, or actuator."""

    @property
    def namespace(self) -> str:
        """Provider slug and tool prefix (no underscores, e.g. 'memory', 'gcal', 'gmail', 'mac')."""
        ...

    @property
    def kind(self) -> ProviderKind:
        """Provider category: hosted, upstream, or actuator."""
        ...

    @property
    def provider_id(self) -> UUID | None:
        """Database provider UUID if persisted, else None."""
        ...

    def is_online(self) -> bool:
        """Whether this provider is currently reachable and its tools should be listed."""
        ...

    async def list_tools(self) -> list[ProviderTool]:
        """Return unprefixed tool definitions offered by this provider."""
        ...

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        ctx: CallContext,
    ) -> types.CallToolResult:
        """Invoke a tool by its unprefixed local name."""
        ...
