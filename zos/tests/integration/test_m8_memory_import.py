"""M8.1 tests for host memory importers (claude-code, codex) and JSONL CLI export."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from zos.cli import app as cli_app
from zos.importers import ClaudeCodeMemoryImporter, CodexMemoryImporter, dumps_jsonl, loads_jsonl


def test_claude_code_importer_frontmatter_skip_index_and_collisions(tmp_path: Path) -> None:
    proj_a = tmp_path / "-Users-zubin-proj-alpha" / "memory"
    proj_b = tmp_path / "-Users-zubin-proj-beta" / "memory"
    proj_a.mkdir(parents=True)
    proj_b.mkdir(parents=True)

    # Index file must be skipped
    (proj_a / "MEMORY.md").write_text("# Index\n- [foo](foo.md)\n", encoding="utf-8")

    # Nested metadata.type frontmatter
    (proj_a / "prefs.md").write_text(
        "---\n"
        "name: Editor Preferences\n"
        "description: Prefers minimal tooling\n"
        "metadata:\n"
        "  type: user\n"
        "  modified: 2026-11-20\n"
        "---\n\n"
        "Uses Neovim and tmux.\n",
        encoding="utf-8",
    )

    # Colliding name across two projects + top-level type frontmatter
    (proj_a / "deploy.md").write_text(
        "---\nname: deploy-notes\ndescription: Alpha deploy\ntype: project\n---\nAlpha body.\n",
        encoding="utf-8",
    )
    (proj_b / "deploy.md").write_text(
        "---\nname: deploy-notes\ndescription: Beta deploy\nmetadata:\n  type: reference\n---\nBeta body.\n",
        encoding="utf-8",
    )

    importer = ClaudeCodeMemoryImporter()
    items = importer.discover(tmp_path)
    by_key = {item.key: item for item in items}

    assert len(items) == 3
    assert "memory" not in by_key

    assert "editor-preferences" in by_key
    ep = by_key["editor-preferences"]
    assert ep.content == "Prefers minimal tooling\n\nUses Neovim and tmux."
    assert set(ep.tags) == {"source:claude-code", "type:user", "project:users-zubin-proj-alpha"}

    assert "users-zubin-proj-alpha/deploy-notes" in by_key
    assert "users-zubin-proj-beta/deploy-notes" in by_key
    assert by_key["users-zubin-proj-alpha/deploy-notes"].content == "Alpha deploy\n\nAlpha body."
    assert "type:project" in by_key["users-zubin-proj-alpha/deploy-notes"].tags
    assert "type:reference" in by_key["users-zubin-proj-beta/deploy-notes"].tags


def test_codex_importer_reads_markdown_and_ignores_sqlite(tmp_path: Path) -> None:
    (tmp_path / "state.sqlite").write_bytes(b"SQLite format 3\x00not-markdown")
    (tmp_path / "MEMORY.md").write_text("# Index\n", encoding="utf-8")
    (tmp_path / "style.md").write_text(
        "---\nname: code-style\ndescription: Python style\ntype: feedback\n---\nKeep functions small.\n",
        encoding="utf-8",
    )

    items = CodexMemoryImporter().discover(tmp_path)
    assert len(items) == 1
    assert items[0].key == "code-style"
    assert items[0].content == "Python style\n\nKeep functions small."
    assert set(items[0].tags) == {"source:codex", "type:feedback"}

    # JSONL roundtrip
    roundtripped = loads_jsonl(dumps_jsonl(items))
    assert roundtripped == items


def test_cli_memory_export_emits_jsonl_without_db(tmp_path: Path) -> None:
    proj = tmp_path / "-Users-zubin-work" / "memory"
    proj.mkdir(parents=True)
    (proj / "MEMORY.md").write_text("# Index\n", encoding="utf-8")
    (proj / "m1.md").write_text(
        "---\nname: work-context\ndescription: Work notes\ntype: project\n---\nDurable fact.\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    r_exp = runner.invoke(cli_app, ["memory", "export", "claude-code", "--root", str(tmp_path)])
    assert r_exp.exit_code == 0, r_exp.output

    items = loads_jsonl(r_exp.output)
    assert len(items) == 1
    assert items[0].key == "work-context"
    assert items[0].content == "Work notes\n\nDurable fact."
    assert set(items[0].tags) == {
        "source:claude-code",
        "type:project",
        "project:users-zubin-work",
    }
