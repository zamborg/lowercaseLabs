"""Server-side confirmation via MCP elicitation (`elicitation/create`)."""

from __future__ import annotations

import logging
from typing import Any

import mcp_types as types
from mcp.server.session import ServerSession

from zos.audit import canonical_json
from zos.auth.context import CallContext

logger = logging.getLogger(__name__)

APPROVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approve": {
            "type": "boolean",
            "title": "Approve tool call",
            "description": "Set to true to execute this exact tool call.",
        }
    },
    "required": ["approve"],
}


def format_approval_message(tool_name: str, args: dict[str, Any]) -> str:
    """Render the exact tool name and full untruncated canonical JSON arguments."""
    return f"Approve ZOS tool call?\nTool: {tool_name}\nArguments: {canonical_json(args)}"


def format_no_elicitation_message(client_name: str, tool_name: str) -> str:
    """Error message naming the exact CLI policy override when client lacks elicitation."""
    return (
        f"Approval required for '{tool_name}', but this client does not support MCP elicitation. "
        f"To override, run: zos policy set --client {client_name} {tool_name} allow"
    )


def session_supports_elicitation(session: ServerSession | None) -> bool:
    if session is None:
        return False
    if not session.can_send_request:
        return False
    return session.check_client_capability(
        types.ClientCapabilities(elicitation=types.ElicitationCapability())
    )


async def request_tool_confirmation(
    session: ServerSession | None,
    request_id: Any,
    ctx: CallContext,
    tool_name: str,
    args: dict[str, Any],
) -> tuple[bool, str | None]:
    """Enforce `require_approval` via MCP elicitation.

    Returns (approved, denial_reason).
    Executes ONLY when action == 'accept' AND content['approve'] is True.
    """
    if not session_supports_elicitation(session):
        return False, format_no_elicitation_message(ctx.client_name, tool_name)

    assert session is not None
    message = format_approval_message(tool_name, args)
    try:
        result = await session.elicit_form(
            message=message,
            requested_schema=APPROVAL_SCHEMA,
            related_request_id=request_id,
        )
    except Exception as exc:
        logger.warning("Elicitation failed for tool %s: %s", tool_name, exc)
        return False, f"Approval elicitation failed for '{tool_name}': {exc}"

    if (
        result.action == "accept"
        and isinstance(result.content, dict)
        and result.content.get("approve") is True
    ):
        return True, None

    if result.action == "accept":
        return False, f"Call to '{tool_name}' was denied (approve was not true)."
    verb = {"decline": "declined", "cancel": "cancelled"}.get(result.action, result.action)
    return False, f"Call to '{tool_name}' was {verb} by user."
