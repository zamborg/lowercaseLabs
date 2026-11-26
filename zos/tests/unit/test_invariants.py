"""Static boundary invariant checks: core never imports module code; modules never import zos."""

from __future__ import annotations

import ast
from pathlib import Path

ZOS_ROOT = Path(__file__).resolve().parents[2]


def _imported_roots(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_core_never_imports_module_code() -> None:
    """`src/zos/` must never import any package from `modules/` (e.g. `zos_inbox`)."""
    forbidden_roots = {"modules", "zos_inbox"}
    for mod_dir in (ZOS_ROOT / "modules").iterdir():
        if mod_dir.is_dir():
            src_dir = mod_dir / "src"
            if src_dir.is_dir():
                for pkg in src_dir.iterdir():
                    if pkg.is_dir():
                        forbidden_roots.add(pkg.name)

    for py_file in (ZOS_ROOT / "src" / "zos").rglob("*.py"):
        imports = _imported_roots(py_file)
        overlap = imports & forbidden_roots
        assert not overlap, f"Core file {py_file} imports module code: {overlap}"


def test_modules_never_import_zos() -> None:
    """Every module under `modules/` and `zos_module_sdk/` must never import `zos`."""
    for py_file in (ZOS_ROOT / "zos_module_sdk").rglob("*.py"):
        assert "zos" not in _imported_roots(py_file), f"{py_file} must not import zos"

    for mod_dir in (ZOS_ROOT / "modules").iterdir():
        src_dir = mod_dir / "src"
        if not src_dir.is_dir():
            continue
        for py_file in src_dir.rglob("*.py"):
            assert "zos" not in _imported_roots(py_file), f"Module file {py_file} imports zos"
