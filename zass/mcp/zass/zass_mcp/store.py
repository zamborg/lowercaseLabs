"""File-backed store for zass state.

The filesystem is the database: every record is a markdown file with YAML
frontmatter under the state/ directory. This module is the only place that
knows the on-disk layout; the MCP server exposes it as tools.

State dir resolution order:
  1. ZASS_STATE_DIR environment variable
  2. <repo root>/state, where repo root is three levels above this file
     (zass/mcp/zass/zass_mcp/store.py -> zass/state)
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

# ---------------------------------------------------------------- paths


def state_dir() -> Path:
    env = os.environ.get("ZASS_STATE_DIR")
    base = Path(env) if env else Path(__file__).resolve().parents[3] / "state"
    return base


def _subdir(name: str) -> Path:
    d = state_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def todos_dir() -> Path:
    return _subdir("todos")


def notes_dir() -> Path:
    return _subdir("notes")


def people_dir() -> Path:
    return _subdir("people")


def memory_dir() -> Path:
    return _subdir("memory")


def journal_dir() -> Path:
    return _subdir("journal")


def inbox_file() -> Path:
    state_dir().mkdir(parents=True, exist_ok=True)
    return state_dir() / "inbox.md"


# ---------------------------------------------------------------- doc I/O

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def read_doc(path: Path) -> tuple[dict, str]:
    """Return (frontmatter dict, body) for a markdown file."""
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    body = text[m.end():]
    return meta, body


def write_doc(path: Path, meta: dict, body: str) -> None:
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    body = body.rstrip("\n")
    text = f"---\n{front}\n---\n"
    if body:
        text += f"\n{body}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "item"


def unique_path(directory: Path, slug: str) -> Path:
    path = directory / f"{slug}.md"
    n = 2
    while path.exists():
        path = directory / f"{slug}-{n}.md"
        n += 1
    return path


def find_by_id(directory: Path, item_id: str) -> Path | None:
    """Find a doc by its filename stem (the canonical id)."""
    exact = directory / f"{item_id}.md"
    if exact.exists():
        return exact
    # forgiving match: slugified query against stems
    q = slugify(item_id)
    candidates = [p for p in directory.glob("*.md") if q and q in p.stem]
    if len(candidates) == 1:
        return candidates[0]
    return None


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def today() -> str:
    return date.today().isoformat()


# ---------------------------------------------------------------- todos

TODO_STATUSES = ("open", "waiting", "scheduled", "done", "dropped")
PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}


def todo_add(
    title: str,
    due: str | None = None,
    priority: str = "normal",
    tags: list[str] | None = None,
    notes: str = "",
    status: str = "open",
) -> dict:
    path = unique_path(todos_dir(), slugify(title))
    meta = {
        "id": path.stem,
        "title": title,
        "status": status,
        "due": due,
        "priority": priority,
        "tags": tags or [],
        "created": now_iso(),
        "completed": None,
    }
    write_doc(path, meta, notes)
    return meta


def todo_all() -> list[tuple[Path, dict, str]]:
    out = []
    for p in sorted(todos_dir().glob("*.md")):
        meta, body = read_doc(p)
        out.append((p, meta, body))
    return out


def todo_list(
    status: str | None = None,
    tag: str | None = None,
    due_within_days: int | None = None,
    include_done: bool = False,
) -> list[dict]:
    rows = []
    horizon = (
        (date.today() + timedelta(days=due_within_days)).isoformat()
        if due_within_days is not None
        else None
    )
    for _, meta, body in todo_all():
        st = meta.get("status", "open")
        if status is not None:
            if st != status:
                continue
        elif not include_done and st in ("done", "dropped"):
            continue
        if tag and tag not in (meta.get("tags") or []):
            continue
        if horizon is not None:
            due = meta.get("due")
            if not due or str(due) > horizon:
                continue
        meta = dict(meta)
        meta["notes"] = body.strip()
        rows.append(meta)
    rows.sort(
        key=lambda m: (
            str(m.get("due") or "9999-12-31"),
            PRIORITY_ORDER.get(m.get("priority", "normal"), 1),
            str(m.get("created", "")),
        )
    )
    return rows


def todo_update(item_id: str, **changes) -> dict | None:
    path = find_by_id(todos_dir(), item_id)
    if path is None:
        return None
    meta, body = read_doc(path)
    append_note = changes.pop("append_note", None)
    for k, v in changes.items():
        if v is not None:
            meta[k] = v
    if changes.get("status") == "done" and not meta.get("completed"):
        meta["completed"] = now_iso()
    if append_note:
        body = (body.rstrip("\n") + f"\n\n- {today()}: {append_note}").lstrip("\n")
    write_doc(path, meta, body)
    return meta


# ---------------------------------------------------------------- notes


def note_add(title: str, body: str, tags: list[str] | None = None) -> dict:
    path = unique_path(notes_dir(), slugify(title))
    meta = {
        "id": path.stem,
        "title": title,
        "tags": tags or [],
        "created": now_iso(),
        "updated": now_iso(),
    }
    write_doc(path, meta, body)
    return meta


def note_get(item_id: str) -> tuple[dict, str] | None:
    path = find_by_id(notes_dir(), item_id)
    if path is None:
        return None
    return read_doc(path)


def note_update(
    item_id: str,
    body: str | None = None,
    append: str | None = None,
    title: str | None = None,
    tags: list[str] | None = None,
) -> dict | None:
    path = find_by_id(notes_dir(), item_id)
    if path is None:
        return None
    meta, old_body = read_doc(path)
    if title:
        meta["title"] = title
    if tags is not None:
        meta["tags"] = tags
    new_body = old_body
    if body is not None:
        new_body = body
    if append:
        new_body = (new_body.rstrip("\n") + f"\n\n{append}").lstrip("\n")
    meta["updated"] = now_iso()
    write_doc(path, meta, new_body)
    return meta


def note_list(tag: str | None = None) -> list[dict]:
    rows = []
    for p in sorted(notes_dir().glob("*.md")):
        meta, _ = read_doc(p)
        if tag and tag not in (meta.get("tags") or []):
            continue
        rows.append(meta)
    rows.sort(key=lambda m: str(m.get("updated", "")), reverse=True)
    return rows


# ---------------------------------------------------------------- people


def person_upsert(
    name: str,
    relation: str | None = None,
    emails: list[str] | None = None,
    phones: list[str] | None = None,
    birthday: str | None = None,
    tags: list[str] | None = None,
    note: str | None = None,
) -> dict:
    slug = slugify(name)
    path = people_dir() / f"{slug}.md"
    if path.exists():
        meta, body = read_doc(path)
    else:
        meta, body = {
            "id": slug,
            "name": name,
            "relation": None,
            "emails": [],
            "phones": [],
            "birthday": None,
            "tags": [],
            "created": now_iso(),
        }, ""
    if relation:
        meta["relation"] = relation
    for field, values in (("emails", emails), ("phones", phones), ("tags", tags)):
        if values:
            merged = list(meta.get(field) or [])
            merged.extend(v for v in values if v not in merged)
            meta[field] = merged
    if birthday:
        meta["birthday"] = birthday
    meta["updated"] = now_iso()
    if note:
        body = (body.rstrip("\n") + f"\n\n- {today()}: {note}").lstrip("\n")
    write_doc(path, meta, body)
    return meta


def person_get(item_id: str) -> tuple[dict, str] | None:
    path = find_by_id(people_dir(), item_id)
    if path is None:
        return None
    return read_doc(path)


def person_log(item_id: str, entry: str) -> dict | None:
    path = find_by_id(people_dir(), item_id)
    if path is None:
        return None
    meta, body = read_doc(path)
    body = (body.rstrip("\n") + f"\n\n- {today()}: {entry}").lstrip("\n")
    meta["updated"] = now_iso()
    write_doc(path, meta, body)
    return meta


def person_list(query: str | None = None) -> list[dict]:
    rows = []
    q = (query or "").lower()
    for p in sorted(people_dir().glob("*.md")):
        meta, body = read_doc(p)
        if q:
            hay = " ".join(
                [str(meta.get("name", "")), str(meta.get("relation", "")), body]
            ).lower()
            if q not in hay:
                continue
        rows.append(meta)
    return rows


# ---------------------------------------------------------------- memory


def memory_save(
    content: str,
    slug: str | None = None,
    kind: str = "fact",
    subject: str | None = None,
) -> dict:
    slug = slugify(slug or content[:50])
    path = memory_dir() / f"{slug}.md"
    if path.exists():
        meta, _ = read_doc(path)
        meta["updated"] = now_iso()
        meta["type"] = kind
        if subject:
            meta["subject"] = subject
    else:
        meta = {
            "id": slug,
            "type": kind,
            "subject": subject or "user",
            "created": now_iso(),
            "updated": now_iso(),
        }
    write_doc(path, meta, content)
    return meta


def memory_list(query: str | None = None) -> list[tuple[dict, str]]:
    rows = []
    q = (query or "").lower()
    for p in sorted(memory_dir().glob("*.md")):
        meta, body = read_doc(p)
        if q and q not in (p.stem + " " + body).lower():
            continue
        rows.append((meta, body.strip()))
    return rows


def memory_forget(item_id: str) -> bool:
    path = find_by_id(memory_dir(), item_id)
    if path is None:
        return False
    path.unlink()
    return True


# ---------------------------------------------------------------- journal


def journal_log(entry: str) -> str:
    path = journal_dir() / f"{today()}.md"
    stamp = datetime.now().strftime("%H:%M")
    if not path.exists():
        path.write_text(f"# {today()}\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {stamp} {entry.strip()}\n")
    return str(path)


def journal_read(days: int = 3) -> str:
    files = sorted(journal_dir().glob("*.md"), reverse=True)[:days]
    parts = [f.read_text(encoding="utf-8").strip() for f in sorted(files)]
    return "\n\n".join(parts) if parts else "(journal is empty)"


# ---------------------------------------------------------------- inbox


def capture(text: str) -> str:
    path = inbox_file()
    if not path.exists():
        path.write_text("# Inbox\n\nQuick-capture queue. Process with /triage.\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- [ ] {now_iso()} — {text.strip()}\n")
    return str(path)


def inbox_read() -> str:
    path = inbox_file()
    if not path.exists():
        return "(inbox is empty)"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- search


def search(query: str, kind: str | None = None, max_results: int = 30) -> list[dict]:
    """Case-insensitive substring search across state files.

    kind: one of todos|notes|people|memory|journal, or None for all.
    Returns [{file, line_no, line}].
    """
    roots: list[Path]
    if kind:
        roots = [state_dir() / kind]
    else:
        roots = [state_dir()]
    q = query.lower()
    hits: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.md")):
            try:
                lines = p.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                if q in line.lower():
                    hits.append(
                        {
                            "file": str(p.relative_to(state_dir())),
                            "line_no": i,
                            "line": line.strip(),
                        }
                    )
                    if len(hits) >= max_results:
                        return hits
    return hits
