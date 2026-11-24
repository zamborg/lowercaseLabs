"""Exhaustive unit matrix for M1.5 tool tiers, policy decisions, trust_ceiling, and annotations."""

from __future__ import annotations

import pytest

from zos.policy import (
    Decision,
    PolicySnapshot,
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
        ("read", None, True, False, False),
        ("read", True, True, False, False),
        ("write", False, False, False, False),
        ("write", True, False, True, False),
        ("write", None, False, True, False),
        ("external", False, False, False, True),
        ("external", True, False, True, True),
        ("external", None, False, True, True),
    ],
)
def test_annotations_mapping(
    tier: Tier,
    destructive: bool | None,
    expected_ro: bool,
    expected_dest: bool,
    expected_ow: bool,
) -> None:
    ann = build_tool_annotations(tier, destructive_hint=destructive)
    assert ann.read_only_hint is expected_ro
    assert ann.destructive_hint is expected_dest
    assert ann.open_world_hint is expected_ow


def test_policy_snapshot_edge_cases_exact_tiebreak_and_untrusted_downgrade() -> None:
    snap = PolicySnapshot(
        tiers={
            "memory_*": "write",
            "memory_get": "read",
            "gmail_send*": "write",
            "gmail_send": "external",
        },
        client_rules={
            "memory_put": "require_approval",
            "mac_shell_run": "allow",
        },
        global_rules={
            "gcal_*": "require_approval",
            "gcal_list_events": "allow",
        },
    )

    # Exact match beats wildcard with identical literal character count ('gmail_send' vs 'gmail_send*')
    ev_owner = snap.evaluate("gmail_send", trust_ceiling="owner")
    assert ev_owner.tier == "external"
    assert ev_owner.decision == "require_approval"
    assert ev_owner.annotations.open_world_hint is True

    # Untrusted turns tier-default, client-rule, AND global-rule require_approval into deny
    assert snap.evaluate("gmail_send", trust_ceiling="untrusted").decision == "deny"
    assert snap.evaluate("memory_put", trust_ceiling="untrusted").decision == "deny"
    assert snap.evaluate("gcal_create_event", trust_ceiling="untrusted").decision == "deny"

    # Untrusted preserves explicit allow (client or global) and read/write defaults
    assert snap.evaluate("mac_shell_run", trust_ceiling="untrusted").decision == "allow"
    assert snap.evaluate("gcal_list_events", trust_ceiling="untrusted").decision == "allow"
    assert snap.evaluate("memory_get", trust_ceiling="untrusted").decision == "allow"

    # Unknown tool defaults to external -> owner: require_approval, untrusted: deny
    unk_owner = snap.evaluate("brand_new_tool", trust_ceiling="owner")
    unk_untrusted = snap.evaluate("brand_new_tool", trust_ceiling="untrusted")
    assert unk_owner.tier == "external" and unk_owner.decision == "require_approval"
    assert unk_owner.annotations.open_world_hint is True
    assert unk_owner.annotations.read_only_hint is False
    assert unk_untrusted.tier == "external" and unk_untrusted.decision == "deny"
