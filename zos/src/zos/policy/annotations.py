"""MCP ToolAnnotations mapping derived from tool tier and provider hints."""

from __future__ import annotations

import mcp_types as types

from zos.policy.tiers import Tier


def build_tool_annotations(
    tier: Tier,
    *,
    destructive_hint: bool = False,
    idempotent_hint: bool | None = None,
) -> types.ToolAnnotations:
    """Map global tool tier + provider destructive/idempotent hints to MCP ToolAnnotations.

    - read -> readOnlyHint=True, openWorldHint=False
    - write -> readOnlyHint=False, openWorldHint=False
    - external -> readOnlyHint=False, openWorldHint=True
    """
    return types.ToolAnnotations(
        read_only_hint=(tier == "read"),
        destructive_hint=destructive_hint,
        idempotent_hint=idempotent_hint,
        open_world_hint=(tier == "external"),
    )
