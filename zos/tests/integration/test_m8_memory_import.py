"""M8.1 tests for memory importers (claude-code, codex), JSONL stdin transport, idempotency, and audit."""

from __future__ import annotations

import asyncio
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from typer.testing import CliRunner

from tests.conftest import MIGRATIONS_DIR, get_test_database_url
from zos.cli import app as cli_app
from zos.db import open_pool, run_migrations
from zos.importers import ClaudeCodeMemoryImporter, CodexMemoryImporter, dumps_jsonl, loads_jsonl


def _reset_db(url: str) -> None:
    async def _prep() -> None:
        async with open_pool(url) as pool:
            await run_migrations(pool, MIGRATIONS_DIR)
            async with pool.connection() as conn:
                await conn.execute("TRUNCATE principals RESTART IDENTITY CASCADE")
                await conn.commit()

    asyncio.run(_prep())


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


def test_cli_memory_import_idempotent_stdin_and_audit(tmp_path: Path, monkeypatch) -> None:
    url = get_test_database_url()
    _reset_db(url)
    monkeypatch.setenv("DATABASE_URL", url)

    proj = tmp_path / "-Users-zubin-work" / "memory"
    proj.mkdir(parents=True)
    secret_body = "PRIVATE_MEMORY_CONTENT_NOT_IN_AUDIT_PAYLOAD"
    f1 = proj / "m1.md"
    f1.write_text(
        f"---\nname: work-context\ndescription: Work notes\ntype: project\n---\n{secret_body}\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    assert runner.invoke(cli_app, ["init", "--principal", "zubin"]).exit_code == 0

    # Export on host -> pipe JSONL into `zos memory import -`
    r_exp = runner.invoke(cli_app, ["memory", "export", "claude-code", "--root", str(tmp_path)])
    assert r_exp.exit_code == 0, r_exp.output

    r_imp1 = runner.invoke(
        cli_app,
        ["memory", "import", "-", "--source-label", "claude-code"],
        input=r_exp.output,
    )
    assert r_imp1.exit_code == 0, r_imp1.output
    assert r_imp1.output.strip() == "discovered=1 created=1 updated=0 unchanged=0"

    # Re-running unchanged is a no-op
    r_imp2 = runner.invoke(
        cli_app,
        ["memory", "import", "claude-code", "--root", str(tmp_path)],
    )
    assert r_imp2.exit_code == 0, r_imp2.output
    assert r_imp2.output.strip() == "discovered=1 created=0 updated=0 unchanged=1"

    # Editing content triggers an update
    f1.write_text(
        "---\nname: work-context\ndescription: Work notes\ntype: project\n---\nUpdated body.\n",
        encoding="utf-8",
    )
    r_imp3 = runner.invoke(
        cli_app,
        ["memory", "import", "claude-code", "--root", str(tmp_path)],
    )
    assert r_imp3.exit_code == 0, r_imp3.output
    assert r_imp3.output.strip() == "discovered=1 created=0 updated=1 unchanged=0"

    # Verify attribution to `memory-import` client and audit events (zero raw content in payload!)
    with psycopg.connect(url, row_factory=dict_row) as conn:
        mem = conn.execute(
            """
            SELECT m.key, m.content, c.name AS author
            FROM memories m
            JOIN clients c ON c.id = m.created_by_client_id
            WHERE m.key = 'work-context'
            """
        ).fetchone()
        assert mem is not None
        assert mem["author"] == "memory-import"
        assert "Updated body." in mem["content"]

        evs = conn.execute(
            "SELECT type, tool, outcome, args_digest, payload::text AS ptxt FROM events WHERE type = 'memory.imported' ORDER BY id ASC"
        ).fetchall()
    assert len(evs) == 2  # 1 created + 1 updated (unchanged was a no-op!)
    assert evs[0]["tool"] == "memory_put"
    assert evs[0]["outcome"] == "ok"
    assert evs[0]["args_digest"] is not None
    assert secret_body not in evs[0]["ptxt"]
    assert '"action": "created"' in evs[0]["ptxt"]
    assert '"action": "updated"' in evs[1]["ptxt"]
