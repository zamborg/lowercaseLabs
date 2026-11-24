"""MCP ToolAnnotations mapping derived from tool tier and provider hints."""

from __future__ import annotations

import mcp_types as types

from zos.policy.tiers import Tier


def build_tool_annotations(
    tier: Tier,
    *,
    destructive_hint: bool | None = None,
    idempotent_hint: bool | None = None,
) -> types.ToolAnnotations:
    """Map global tool tier + provider destructive/idempotent hints to MCP ToolAnnotations.

    - read -> readOnlyHint=True, destructiveHint=False, openWorldHint=False
    - write -> readOnlyHint=False, openWorldHint=False, destructiveHint=True unless explicitly False
    - external -> readOnlyHint=False, openWorldHint=True, destructiveHint=True unless explicitly False
    """
    if tier == "read":
        resolved_destructive = False
    elif destructive_hint is None:
        resolved_destructive = True
    else:
        resolved_destructive = bool(destructive_hint)

    return types.ToolAnnotations(
        read_only_hint=(tier == "read"),
        destructive_hint=resolved_destructive,
        idempotent_hint=idempotent_hint,
        open_world_hint=(tier == "external"),
    )
