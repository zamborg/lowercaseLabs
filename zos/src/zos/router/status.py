"""Always-on `zos_status` router-owned tool."""

from __future__ import annotations

import json

import mcp_types as types

from zos.router.registry import ProviderRegistry

ZOS_STATUS_TOOL_NAME = "zos_status"


def zos_status_tool(annotations: types.ToolAnnotations | None = None) -> types.Tool:
    return types.Tool(
        name=ZOS_STATUS_TOOL_NAME,
        description="List ZOS providers and their current online status.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        annotations=annotations,
    )


def render_status_result(registry: ProviderRegistry) -> types.CallToolResult:
    providers = [
        {
            "namespace": p.namespace,
            "kind": p.kind,
            "online": p.is_online(),
        }
        for p in registry.all_providers()
    ]
    payload = {"providers": providers}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2))],
    )
