from __future__ import annotations

import errno
import json
import os
import pwd
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import typer
from rich.console import Console
from rich.prompt import Confirm


out_console = Console()
err_console = Console(stderr=True)


@dataclass(frozen=True)
class TrashContext:
    uid: int
    home: Path
    home_dev: int


@dataclass(frozen=True)
class TrashItem:
    path: Path
    trash_dir: Path
    kind: str
    size_bytes: int | None
    modified_local: datetime


def _invoking_uid() -> int:
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid:
        try:
            return int(sudo_uid)
        except ValueError:
            pass
    return os.getuid()


def _home_for_uid(uid: int) -> Path:
    return Path(pwd.getpwuid(uid).pw_dir)


def _path_lstat(path: Path) -> os.stat_result:
    return os.lstat(path)


def _is_dir_no_follow(path: Path, st: os.stat_result | None = None) -> bool:
    st = st or _path_lstat(path)
    return stat.S_ISDIR(st.st_mode)


def _find_mount_point(path: Path) -> Path:
    absolute_path = path.absolute()
    try:
        st = _path_lstat(absolute_path)
    except FileNotFoundError:
        raise

    if _is_dir_no_follow(absolute_path, st=st):
        current = absolute_path
        dev = st.st_dev
    else:
        current = absolute_path.parent
        dev = _path_lstat(current).st_dev

    while True:
        parent = current.parent
        if parent == current:
            return current
        if _path_lstat(parent).st_dev != dev:
            return current
        current = parent


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_trash_context() -> TrashContext:
    uid = _invoking_uid()
    home = _home_for_uid(uid)
    return TrashContext(uid=uid, home=home, home_dev=_path_lstat(home).st_dev)


def expand_path(raw: str, ctx: TrashContext) -> Path:
    expanded = os.path.expandvars(raw)
    if expanded == "~":
        return ctx.home
    if expanded.startswith("~/"):
        return ctx.home / expanded[2:]
    return Path(os.path.expanduser(expanded))


def _maybe_chown(path: Path, *, uid: int) -> None:
    if os.geteuid() != 0 or uid == 0:
        return
    try:
        gid = pwd.getpwuid(uid).pw_gid
    except KeyError:
        return
    try:
        os.chown(path, uid, gid)
    except PermissionError:
        return
    except OSError:
        return


def _state_dir(ctx: TrashContext) -> Path:
    return ctx.home / ".trasher"


def _state_map_path(ctx: TrashContext) -> Path:
    return _state_dir(ctx) / "map.json"


def _ensure_state_dir(ctx: TrashContext) -> Path:
    path = _state_dir(ctx)
    _ensure_dir(path)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    _maybe_chown(path, uid=ctx.uid)
    return path


def _load_state_map(ctx: TrashContext) -> dict:
    map_path = _state_map_path(ctx)
    try:
        raw = map_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"version": 1, "items": {}}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        backup = map_path.with_name(
            f"{map_path.name}.broken.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        try:
            os.replace(map_path, backup)
        except OSError:
            pass
        return {"version": 1, "items": {}}

    if not isinstance(data, dict):
        return {"version": 1, "items": {}}

    items = data.get("items")
    if not isinstance(items, dict):
        data["items"] = {}
    if data.get("version") != 1:
        data["version"] = 1
    return data


def _save_state_map(ctx: TrashContext, data: dict) -> None:
    _ensure_state_dir(ctx)
    map_path = _state_map_path(ctx)
    tmp_path = map_path.with_name(f"{map_path.name}.tmp.{os.getpid()}")
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"

    tmp_path.write_text(payload, encoding="utf-8")
    try:
        tmp_path.chmod(0o600)
    except OSError:
        pass
    _maybe_chown(tmp_path, uid=ctx.uid)
    os.replace(tmp_path, map_path)
    try:
        map_path.chmod(0o600)
    except OSError:
        pass
    _maybe_chown(map_path, uid=ctx.uid)


def read_trash_map(ctx: TrashContext) -> dict[str, dict]:
    data = _load_state_map(ctx)
    items = data.get("items", {})
    if not isinstance(items, dict):
        return {}
    return dict(items)


def write_trash_map(ctx: TrashContext, items: dict[str, dict]) -> None:
    _save_state_map(ctx, {"version": 1, "items": items})


def record_trash_moves(ctx: TrashContext, moves: Iterable[tuple[Path, Path]]) -> None:
    items = read_trash_map(ctx)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    for trashed_path, original_path in moves:
        items[str(trashed_path)] = {
            "original": str(original_path),
            "deleted_at": now,
            "uid": ctx.uid,
        }
    write_trash_map(ctx, items)


def _unique_destination(trash_dir: Path, name: str) -> Path:
    candidate = trash_dir / name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for i in range(1, 10_000):
        numbered = trash_dir / f"{stem} ({i}){suffix}"
        if not numbered.exists():
            return numbered

    raise RuntimeError(f"Unable to find unique destination for {name!r} in {trash_dir}")


def _trash_dir_for_path(path: Path, ctx: TrashContext) -> Path:
    is_macos = os.uname().sysname == "Darwin"
    if not is_macos:
        return ctx.home / ".trash"

    try:
        source_dev = _path_lstat(path).st_dev
    except FileNotFoundError:
        source_dev = None

    if source_dev is not None and source_dev == ctx.home_dev:
        return ctx.home / ".Trash"

    mount_point = _find_mount_point(path)
    return mount_point / ".Trashes" / str(ctx.uid)


def _move_into_trash(src: Path, ctx: TrashContext) -> Path:
    trash_dir = _trash_dir_for_path(src, ctx)
    try:
        _ensure_dir(trash_dir)
    except PermissionError:
        fallback = ctx.home / ".Trash"
        try:
            _ensure_dir(fallback)
            trash_dir = fallback
        except PermissionError:
            trash_dir = ctx.home / ".trash"
            _ensure_dir(trash_dir)

    destination = _unique_destination(trash_dir, src.name)

    try:
        os.rename(src, destination)
        return destination
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EACCES, errno.EPERM):
            raise

    shutil.move(str(src), str(destination))
    return destination


def move_from_trash(src: Path, dest: Path) -> None:
    try:
        os.rename(src, dest)
        return
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EACCES, errno.EPERM):
            raise
    shutil.move(str(src), str(dest))


def trash_locations(
    ctx: TrashContext, *, include_volumes: bool, include_fallback: bool
) -> list[Path]:
    is_macos = os.uname().sysname == "Darwin"
    if not is_macos:
        return [ctx.home / ".trash"]

    candidates: list[Path] = [ctx.home / ".Trash"]
    if include_fallback:
        candidates.append(ctx.home / ".trash")

    if include_volumes:
        candidates.append(Path("/") / ".Trashes" / str(ctx.uid))
        volumes = Path("/Volumes")
        if volumes.is_dir():
            for volume in volumes.iterdir():
                if volume.is_dir():
                    candidates.append(volume / ".Trashes" / str(ctx.uid))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def list_trash_items(
    ctx: TrashContext,
    *,
    include_volumes: bool,
    include_fallback: bool,
    include_dotfiles: bool,
) -> tuple[list[TrashItem], list[tuple[Path, Exception]]]:
    items: list[TrashItem] = []
    warnings: list[tuple[Path, Exception]] = []

    for trash_dir in trash_locations(
        ctx, include_volumes=include_volumes, include_fallback=include_fallback
    ):
        try:
            entries = list(trash_dir.iterdir())
        except FileNotFoundError:
            continue
        except PermissionError as e:
            warnings.append((trash_dir, e))
            continue

        for entry in entries:
            if not include_dotfiles and entry.name.startswith("."):
                continue
            try:
                st = _path_lstat(entry)
            except (FileNotFoundError, PermissionError, OSError) as e:
                warnings.append((entry, e))
                continue

            if stat.S_ISDIR(st.st_mode):
                kind = "dir"
                size_bytes = None
            elif stat.S_ISLNK(st.st_mode):
                kind = "link"
                size_bytes = None
            elif stat.S_ISREG(st.st_mode):
                kind = "file"
                size_bytes = st.st_size
            else:
                kind = "other"
                size_bytes = None

            items.append(
                TrashItem(
                    path=entry,
                    trash_dir=trash_dir,
                    kind=kind,
                    size_bytes=size_bytes,
                    modified_local=datetime.fromtimestamp(st.st_mtime).astimezone(),
                )
            )

    return items, warnings


def trash_paths(
    paths: Iterable[str],
    *,
    recursive: bool,
    force: bool,
    interactive: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    ctx = get_trash_context()

    if force:
        interactive = False

    had_error = False
    moved: list[tuple[Path, Path]] = []
    for raw in paths:
        path = expand_path(raw, ctx)

        raw_clean = raw.strip()
        if raw_clean == "" or Path(raw_clean) in {Path("."), Path("..")}:
            err_console.print(f"[red]trash:[/red] refusing to trash {raw!r}")
            had_error = True
            continue

        if os.path.normpath(str(path.absolute())) == "/":
            err_console.print("[red]trash:[/red] refusing to trash '/'")
            had_error = True
            continue

        try:
            st = _path_lstat(path)
        except FileNotFoundError:
            if force:
                continue
            err_console.print(f"[red]trash:[/red] {raw}: No such file or directory")
            had_error = True
            continue
        except PermissionError:
            err_console.print(f"[red]trash:[/red] {raw}: Permission denied")
            had_error = True
            continue

        original_abs = path.absolute()

        if _is_dir_no_follow(path, st=st) and not recursive:
            err_console.print(
                f"[red]trash:[/red] {raw}: is a directory (use [bold]-r[/bold])"
            )
            had_error = True
            continue

        if interactive and not Confirm.ask(
            f"Trash {raw!r}?", default=False, console=err_console
        ):
            continue

        trash_dir = _trash_dir_for_path(path, ctx)
        destination = _unique_destination(trash_dir, path.name)

        if dry_run:
            if verbose:
                out_console.print(f"{path} -> {destination}")
            continue

        try:
            destination = _move_into_trash(path, ctx)
        except FileNotFoundError:
            if force:
                continue
            err_console.print(f"[red]trash:[/red] {raw}: No such file or directory")
            had_error = True
            continue
        except Exception as e:  # noqa: BLE001
            err_console.print(f"[red]trash:[/red] {raw}: {e}")
            had_error = True
            continue

        moved.append((destination, original_abs))

        if verbose:
            out_console.print(f"{path} -> {destination}")

    if moved:
        try:
            record_trash_moves(ctx, moved)
        except Exception as e:  # noqa: BLE001
            err_console.print(f"[yellow]warn:[/yellow] could not update ~/.trasher map: {e}")

    if had_error:
        raise typer.Exit(code=1)
