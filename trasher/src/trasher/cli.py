from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.filesize import decimal
from rich.prompt import Confirm
from rich.table import Table
import typer

from . import __version__
from .trash import (
    expand_path,
    get_trash_context,
    list_trash_items,
    move_from_trash,
    read_trash_map,
    trash_locations,
    trash_paths,
    write_trash_map,
)

out_console = Console()
err_console = Console(stderr=True)

rm_app = typer.Typer(
    add_completion=False,
    subcommand_metavar="",
    rich_markup_mode="rich",
    help=(
        "Move files and folders to the macOS Trash instead of deleting them.\n\n"
        "Examples:\n"
        "  trash file.txt\n"
        "  trash -r folder/\n"
        "  trash -rf build/ dist/\n"
        "\n"
        "Trash inspection:\n"
        "  trash ls\n"
        "  trash where\n"
        "\n"
        "Restore / empty:\n"
        "  trash restore <name-or-path>\n"
        "  trash empty\n"
    ),
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit(code=0)


@rm_app.callback(invoke_without_command=True)
def trash_command(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(None, help="Files and folders to move to Trash."),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
    recursive: bool = typer.Option(
        False,
        "-r",
        "-R",
        "--recursive",
        help="Allow removing directories and their contents.",
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Ignore nonexistent files and never prompt.",
    ),
    interactive: bool = typer.Option(
        False,
        "-i",
        "--interactive",
        help="Prompt before every removal.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be moved, without changing anything.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Print each moved path.",
    ),
) -> None:
    _ = version
    if not paths:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)

    trash_paths(
        paths,
        recursive=recursive,
        force=force,
        interactive=interactive,
        dry_run=dry_run,
        verbose=verbose,
    )


manage_app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
)


@manage_app.command("ls", help="List items currently in the Trash.")
def list_command(
    long: bool = typer.Option(False, "-l", "--long", help="Show more detail."),
    include_dotfiles: bool = typer.Option(
        False, "-a", "--all", help="Include entries starting with '.'."
    ),
    include_fallback: bool = typer.Option(
        False, "--include-fallback", help="Also scan ~/.trash (fallback)."
    ),
    home_only: bool = typer.Option(
        False, "--home-only", help="Only show items from ~/.Trash."
    ),
    paths_only: bool = typer.Option(
        False, "--paths", help="Print full paths, one per line."
    ),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse sort order."),
    sort: str = typer.Option(
        "name",
        "--sort",
        help="Sort by: name, modified, size.",
        show_default=True,
    ),
) -> None:
    sort_key = sort.strip().lower()
    if sort_key not in {"name", "modified", "size"}:
        raise typer.BadParameter("sort must be one of: name, modified, size")

    ctx = get_trash_context()
    items, warnings = list_trash_items(
        ctx,
        include_volumes=not home_only,
        include_fallback=include_fallback,
        include_dotfiles=include_dotfiles,
    )

    if sort_key == "name":
        items.sort(key=lambda item: item.path.name.casefold(), reverse=reverse)
    elif sort_key == "modified":
        items.sort(key=lambda item: item.modified_local, reverse=not reverse)
    else:
        items.sort(key=lambda item: item.size_bytes or -1, reverse=reverse)

    if warnings and not paths_only:
        for path, exc in warnings:
            err_console.print(f"[yellow]warn:[/yellow] {path}: {exc}")

    if paths_only:
        for item in items:
            typer.echo(str(item.path))
        return

    table = Table(title="Trash", show_lines=False)
    table.add_column("Name", overflow="fold")
    table.add_column("Type", no_wrap=True)
    table.add_column("Trash Dir", overflow="fold")
    if long:
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("Modified", no_wrap=True)

    for item in items:
        trash_dir_display = str(item.trash_dir)
        try:
            trash_dir_display = str(Path("~") / item.trash_dir.relative_to(ctx.home))
        except ValueError:
            pass
        if long:
            size = "-" if item.size_bytes is None else decimal(item.size_bytes)
            table.add_row(
                item.path.name,
                item.kind,
                trash_dir_display,
                size,
                item.modified_local.strftime("%Y-%m-%d %H:%M"),
            )
        else:
            table.add_row(item.path.name, item.kind, trash_dir_display)

    out_console.print(table)


@manage_app.command("list", help="Alias for ls.")
def list_alias(
    long: bool = typer.Option(False, "-l", "--long", help="Show more detail."),
    include_dotfiles: bool = typer.Option(
        False, "-a", "--all", help="Include entries starting with '.'."
    ),
    include_fallback: bool = typer.Option(
        False, "--include-fallback", help="Also scan ~/.trash (fallback)."
    ),
    home_only: bool = typer.Option(
        False, "--home-only", help="Only show items from ~/.Trash."
    ),
    paths_only: bool = typer.Option(
        False, "--paths", help="Print full paths, one per line."
    ),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse sort order."),
    sort: str = typer.Option(
        "name",
        "--sort",
        help="Sort by: name, modified, size.",
        show_default=True,
    ),
) -> None:
    list_command(
        long=long,
        include_dotfiles=include_dotfiles,
        include_fallback=include_fallback,
        home_only=home_only,
        paths_only=paths_only,
        reverse=reverse,
        sort=sort,
    )


@manage_app.command("where", help="Show which Trash directories are used/scanned.")
def where_command(
    home_only: bool = typer.Option(
        False, "--home-only", help="Only show ~/.Trash and ~/.trash."
    ),
) -> None:
    ctx = get_trash_context()
    for location in trash_locations(
        ctx, include_volumes=not home_only, include_fallback=True
    ):
        typer.echo(str(location))


@manage_app.command("restore", help="Restore item(s) trashed by this tool.")
def restore_command(
    targets: list[str] = typer.Argument(
        ..., help="A trashed name, original path, or a full Trash path."
    ),
    parents: bool = typer.Option(
        False,
        "-p",
        "--parents",
        help="Create missing parent directories in the restore destination.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be restored, without changing anything.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Print each restored path.",
    ),
) -> None:
    ctx = get_trash_context()
    items = read_trash_map(ctx)

    if not items:
        err_console.print(
            "[yellow]No ~/.trasher map entries found; restore only works for items trashed by this tool.[/yellow]"
        )
        raise typer.Exit(code=1)

    def exists_in_trash(map_key: str) -> bool:
        try:
            return Path(map_key).exists()
        except OSError:
            return False

    def resolve_target(target: str) -> list[str]:
        expanded_path = expand_path(target, ctx)
        expanded_text = str(expanded_path)
        looks_like_path = (
            ("/" in target)
            or target.startswith(("~", ".", "/"))
            or ("/" in expanded_text)
            or expanded_path.is_absolute()
        )

        if looks_like_path:
            candidate = expanded_path
            candidate_abs = (
                candidate if candidate.is_absolute() else (Path.cwd() / candidate).absolute()
            )
            key = str(candidate_abs)
            if key in items and exists_in_trash(key):
                return [key]

            original_matches = [
                k
                for k, v in items.items()
                if v.get("original") == key and exists_in_trash(k)
            ]
            return original_matches

        needle = expanded_text
        name_matches = [
            k for k in items.keys() if Path(k).name == needle and exists_in_trash(k)
        ]
        if name_matches:
            return name_matches

        original_name_matches: list[str] = [
            k
            for k, v in items.items()
            if Path(v.get("original", "")).name == needle and exists_in_trash(k)
        ]
        return original_name_matches

    had_error = False
    changed = False

    for target in targets:
        matches = resolve_target(target)
        if not matches:
            err_console.print(
                f"[red]restore:[/red] {target!r}: no match (try `trash ls` or `trash ls --paths`)"
            )
            had_error = True
            continue
        if len(matches) > 1:
            err_console.print(
                f"[red]restore:[/red] {target!r}: ambiguous (multiple matches); use `trash ls --paths` and pass the full path."
            )
            had_error = True
            continue

        match_key = matches[0]

        meta = items.get(match_key, {})
        original_str = meta.get("original")
        if not original_str:
            err_console.print(
                f"[red]restore:[/red] {target!r}: missing original path in ~/.trasher map"
            )
            had_error = True
            continue

        trashed_path = Path(match_key)
        original_path = Path(original_str)

        if not trashed_path.exists():
            err_console.print(
                f"[red]restore:[/red] {target!r}: item is no longer in Trash: {trashed_path}"
            )
            had_error = True
            continue

        if original_path.exists():
            err_console.print(
                f"[red]restore:[/red] {target!r}: destination already exists: {original_path}"
            )
            had_error = True
            continue

        if parents:
            try:
                original_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                err_console.print(
                    f"[red]restore:[/red] {target!r}: could not create parent directories: {e}"
                )
                had_error = True
                continue

        if not original_path.parent.exists():
            err_console.print(
                f"[red]restore:[/red] {target!r}: destination parent does not exist: {original_path.parent} (use --parents)"
            )
            had_error = True
            continue

        if dry_run:
            out_console.print(f"{trashed_path} -> {original_path}")
            continue

        try:
            move_from_trash(trashed_path, original_path)
        except Exception as e:  # noqa: BLE001
            err_console.print(f"[red]restore:[/red] {target!r}: {e}")
            had_error = True
            continue

        if verbose:
            out_console.print(f"{trashed_path} -> {original_path}")

        items.pop(match_key, None)
        changed = True

    if changed and not dry_run:
        try:
            write_trash_map(ctx, items)
        except Exception as e:  # noqa: BLE001
            err_console.print(f"[yellow]warn:[/yellow] could not update ~/.trasher map: {e}")

    if had_error:
        raise typer.Exit(code=1)


@manage_app.command("empty", help="Permanently delete items from the Trash.")
def empty_command(
    yes: bool = typer.Option(
        False, "-y", "--yes", help="Do not prompt before permanently deleting."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be permanently deleted, without changing anything.",
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Print each permanently deleted path."
    ),
    include_fallback: bool = typer.Option(
        False, "--include-fallback", help="Also empty ~/.trash (fallback)."
    ),
    home_only: bool = typer.Option(
        False, "--home-only", help="Only empty ~/.Trash."
    ),
) -> None:
    ctx = get_trash_context()
    trash_items, warnings = list_trash_items(
        ctx,
        include_volumes=not home_only,
        include_fallback=include_fallback,
        include_dotfiles=True,
    )

    if warnings:
        for path, exc in warnings:
            err_console.print(f"[yellow]warn:[/yellow] {path}: {exc}")

    deletable = [
        item
        for item in trash_items
        if item.path.name not in {".DS_Store", ".localized"}
    ]

    if not deletable:
        out_console.print("Trash is already empty.")
        return

    if dry_run:
        for item in deletable:
            out_console.print(str(item.path))
        return

    if not yes and not Confirm.ask(
        f"Permanently delete {len(deletable)} item(s) from Trash?",
        default=False,
        console=err_console,
    ):
        raise typer.Exit(code=0)

    had_error = False
    removed_paths: set[str] = set()
    for item in deletable:
        try:
            if item.kind == "dir":
                shutil.rmtree(item.path)
            else:
                item.path.unlink()
            removed_paths.add(str(item.path))
            if verbose:
                out_console.print(str(item.path))
        except Exception as e:  # noqa: BLE001
            err_console.print(f"[red]empty:[/red] {item.path}: {e}")
            had_error = True

    try:
        map_items = read_trash_map(ctx)
        changed = False
        for key in list(map_items.keys()):
            if key in removed_paths:
                del map_items[key]
                changed = True
                continue
            try:
                if not Path(key).exists():
                    del map_items[key]
                    changed = True
            except OSError:
                continue
        if changed:
            write_trash_map(ctx, map_items)
    except Exception as e:  # noqa: BLE001
        err_console.print(f"[yellow]warn:[/yellow] could not update ~/.trasher map: {e}")

    if had_error:
        raise typer.Exit(code=1)


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in {"ls", "list", "where", "restore", "empty"}:
        manage_app()
    else:
        rm_app()
