"""MCP server `instructions` returned in InitializeResult (spec-native, client-agnostic)."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MCP_INSTRUCTIONS = (
    "ZOS is Zubin's personal context and tool routing layer.\n\n"
    "- Context first: before starting work that may depend on Zubin's preferences, "
    "background, or ongoing projects, search memory with `memory_search` or browse keys "
    "with `memory_list`.\n"
    "- Saving memories: persist durable facts with `memory_put` using descriptive "
    "kebab-case keys and relevant tags (e.g. `type:preference`, `project:<name>`). "
    "Never store secrets, tokens, or passwords.\n"
    "- Providers: tool prefixes map to providers (`memory_*`, `gcal_*`, `gmail_*`, "
    "`mac_*`); call `zos_status` to check which providers are currently online.\n"
    "- Approvals: `external`-tier tools (sending email, changing calendar events, "
    "running shell commands) require explicit owner approval before executing."
)


def load_mcp_instructions(instructions_file: str = "") -> str:
    """Return default MCP server instructions, or load an override from `ZOS_INSTRUCTIONS_FILE`."""
    path_str = instructions_file.strip()
    if not path_str:
        return DEFAULT_MCP_INSTRUCTIONS
    path = Path(path_str).expanduser()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"ZOS_INSTRUCTIONS_FILE is empty: {path}")
    return text
