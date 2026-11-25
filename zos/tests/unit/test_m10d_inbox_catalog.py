"""Catalog validation test for the checked-in `catalog/modules/inbox.toml` manifest."""

from __future__ import annotations

from pathlib import Path

from zos.catalog import get_module_catalog_entry, load_module_catalog


def test_checked_in_inbox_module_manifest_is_valid() -> None:
    entries = load_module_catalog()
    assert any(e.slug == "inbox" for e in entries)

    inbox = get_module_catalog_entry("inbox")
    assert inbox is not None
    assert inbox.manifest_version == 2
    assert inbox.runtime.kind == "container"
    assert inbox.runtime.build == "modules/inbox"
    assert inbox.runtime.mcp_port == 8081
    assert inbox.runtime.mcp_path == "/mcp"
    assert inbox.runtime.ui_port == 8082
    assert inbox.storage.postgres is True
    assert inbox.network.egress is False
    assert inbox.ui is not None and inbox.ui.title == "Inbox"
    assert inbox.tiers.patterns["inbox_list"] == "read"
    assert inbox.tiers.patterns["inbox_read"] == "read"
    assert inbox.tiers.patterns["inbox_directory"] == "read"
    assert inbox.tiers.patterns["inbox_send"] == "write"
    assert inbox.tiers.patterns["inbox_archive"] == "write"

    # Build directory and Dockerfile exist on disk
    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "modules" / "inbox" / "Dockerfile").is_file()
    assert (repo_root / "modules" / "inbox" / "pyproject.toml").is_file()
