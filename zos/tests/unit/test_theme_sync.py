"""Module UIs embed a generated copy of the shared theme (they must not import zos)."""

from pathlib import Path

from zos.web.theme import module_theme_block, module_ui_files


def test_every_module_ui_embeds_current_theme() -> None:
    files = module_ui_files()
    assert files, "expected module UIs under modules/*/src/*/ui.py"
    block = module_theme_block()
    stale = [f for f in files if block not in Path(f).read_text()]
    assert not stale, f"theme drifted; run `uv run python -m zos.web.theme`: {stale}"
