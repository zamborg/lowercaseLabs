"""Exhaustive unit matrix for M1.5 tool tiers, policy decisions, trust_ceiling, and annotations."""

from __future__ import annotations

import pytest

from zos.policy import (
    Decision,
    Tier,
    build_tool_annotations,
    resolve_decision,
    resolve_tier,
)


@pytest.mark.parametrize(
    ("tool", "tiers", "expected"),
    [
        ("memory_get", {"memory_get": "read", "memory_*": "write"}, "read"),
        ("memory_put", {"memory_get": "read", "memory_*": "write"}, "write"),
        ("memory_put", {"*": "read", "memory_*": "write"}, "write"),
        ("unknown_tool", {"memory_get": "read"}, "external"),
        ("unknown_tool", {}, "external"),
        ("gcal_list_events", {"gcal_*": "external", "gcal_list_events": "read"}, "read"),
        ("gcal_delete_event", {"gcal_*": "external", "gcal_list_*": "read"}, "external"),
    ],
)
def test_tier_resolution(tool: str, tiers: dict[str, Tier], expected: Tier) -> None:
    assert resolve_tier(tool, tiers) == expected


@pytest.mark.parametrize(
    ("tier", "trust_ceiling", "client_rules", "global_rules", "expected"),
    [
        # Tier defaults for owner
        ("read", "owner", None, None, "allow"),
        ("write", "owner", None, None, "allow"),
        ("external", "owner", None, None, "require_approval"),
        # Tier defaults for untrusted (require_approval -> deny)
        ("read", "untrusted", None, None, "allow"),
        ("write", "untrusted", None, None, "allow"),
        ("external", "untrusted", None, None, "deny"),
        # Global rule overrides tier default
        ("external", "owner", None, {"gmail_send": "allow"}, "allow"),
        ("read", "owner", None, {"gmail_*": "deny"}, "deny"),
        ("write", "owner", None, {"memory_put": "require_approval"}, "require_approval"),
        ("write", "untrusted", None, {"memory_put": "require_approval"}, "deny"),
        # Client rule overrides global rule
        ("external", "owner", {"gmail_send": "allow"}, {"gmail_send": "deny"}, "allow"),
        ("read", "owner", {"memory_get": "deny"}, {"memory_*": "allow"}, "deny"),
        ("external", "untrusted", {"gmail_send": "allow"}, {"gmail_send": "deny"}, "allow"),
        ("read", "untrusted", {"memory_get": "require_approval"}, {"memory_get": "allow"}, "deny"),
        # Longest literal match within client rules
        (
            "external",
            "owner",
            {"gmail_*": "deny", "gmail_send": "allow"},
            None,
            "allow",
        ),
        (
            "external",
            "owner",
            {"*": "allow", "gmail_*": "deny"},
            None,
            "deny",
        ),
    ],
)
def test_decision_matrix(
    tier: Tier,
    trust_ceiling: str,
    client_rules: dict[str, Decision] | None,
    global_rules: dict[str, Decision] | None,
    expected: Decision,
) -> None:
    tool_name = (
        next(iter(client_rules)) if client_rules and "*" not in next(iter(client_rules)) else None
    )
    if tool_name is None:
        tool_name = (
            next(iter(global_rules)) if global_rules and "*" not in next(iter(global_rules)) else None
        )
    if tool_name is None:
        tool_name = "gmail_send" if tier == "external" else ("memory_put" if tier == "write" else "memory_get")
    # Ensure wildcard test cases test exact matching target
    if client_rules and ("gmail_send" in client_rules or "gmail_*" in client_rules):
        tool_name = "gmail_send"
    elif global_rules and "gmail_*" in global_rules:
        tool_name = "gmail_read"

    actual = resolve_decision(
        tool_name,
        tier=tier,
        trust_ceiling=trust_ceiling,  # type: ignore[arg-type]
        client_rules=client_rules,
        global_rules=global_rules,
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("tier", "destructive", "expected_ro", "expected_dest", "expected_ow"),
    [
        ("read", False, True, False, False),
        ("write", False, False, False, False),
        ("write", True, False, True, False),
        ("external", False, False, False, True),
        ("external", True, False, True, True),
    ],
)
def test_annotations_mapping(
    tier: Tier,
    destructive: bool,
    expected_ro: bool,
    expected_dest: bool,
    expected_ow: bool,
) -> None:
    ann = build_tool_annotations(tier, destructive_hint=destructive)
    assert ann.read_only_hint is expected_ro
    assert ann.destructive_hint is expected_dest
    assert ann.open_world_hint is expected_ow
