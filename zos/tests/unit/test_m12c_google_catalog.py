"""Catalog and control-plane boundary tests for M12c Google modules (`gmail`, `gcal`)."""

from __future__ import annotations

from pathlib import Path

from zos.catalog import get_module_catalog_entry, load_module_catalog


def test_checked_in_gmail_and_gcal_manifests_and_core_has_no_google_provider() -> None:
    entries = {e.slug: e for e in load_module_catalog()}
    assert {"inbox", "memory", "gmail", "gcal"} <= set(entries)

    gmail = get_module_catalog_entry("gmail")
    assert gmail is not None
    assert gmail.manifest_version == 2
    assert gmail.runtime.build == "modules/google"
    assert gmail.storage.postgres is True
    assert gmail.network.egress is True
    assert gmail.ui is not None and gmail.ui.title == "Gmail"
    assert {f.name: f.secret for f in gmail.config} == {
        "GOOGLE_CLIENT_ID": False,
        "GOOGLE_CLIENT_SECRET": True,
    }
    assert gmail.tiers.patterns == {
        "gmail_search": "read",
        "gmail_read": "read",
        "gmail_send": "external",
    }

    gcal = get_module_catalog_entry("gcal")
    assert gcal is not None
    assert gcal.manifest_version == 2
    assert gcal.runtime.build == "modules/google"
    assert gcal.storage.postgres is True
    assert gcal.network.egress is True
    assert gcal.ui is not None and gcal.ui.title == "Google Calendar"
    assert {"GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GCAL_BACKEND", "GCAL_MCP_URL"} == {
        f.name for f in gcal.config
    }
    assert gcal.tiers.patterns["gcal_list_events"] == "read"
    assert gcal.tiers.patterns["gcal_freebusy"] == "read"
    assert gcal.tiers.patterns["gcal_create_event"] == "external"

    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "modules" / "google" / "Dockerfile").is_file()
    assert (repo_root / "modules" / "google" / "pyproject.toml").is_file()
    assert not (repo_root / "src" / "zos" / "providers" / "google").exists()
    assert not (repo_root / "src" / "zos" / "providers" / "memory.py").exists()
