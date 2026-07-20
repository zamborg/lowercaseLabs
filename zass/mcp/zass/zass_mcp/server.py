"""zass MCP server — personal assistant primitives over the state/ directory.

Run: `uv run --project mcp/zass zass-mcp` (stdio transport).
All tools return markdown strings for direct agent consumption.
"""

from __future__ import annotations

from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

from . import apple, store

mcp = FastMCP(
    "zass",
    instructions=(
        "zass is the user's personal assistant state layer. State lives as "
        "markdown files under state/ (todos, notes, people, memory, journal, "
        "inbox). Call briefing() at the start of a session. Log meaningful "
        "actions with journal_log so future sessions have continuity. Save "
        "durable facts about the user and their world with memory_save."
    ),
)


def _fmt_todo(m: dict) -> str:
    bits = [f"[{m.get('status', 'open')}]", f"**{m.get('title')}**", f"(id: {m.get('id')})"]
    if m.get("due"):
        bits.append(f"due {m['due']}")
    if m.get("priority") and m["priority"] != "normal":
        bits.append(f"priority {m['priority']}")
    if m.get("tags"):
        bits.append("#" + " #".join(m["tags"]))
    line = "- " + " ".join(str(b) for b in bits)
    if m.get("notes"):
        line += f"\n  {m['notes'].splitlines()[0]}"
    return line


def _fmt_todos(rows: list[dict], empty: str = "(none)") -> str:
    return "\n".join(_fmt_todo(m) for m in rows) if rows else empty


# ---------------------------------------------------------------- session


@mcp.tool()
def briefing() -> str:
    """Session bootstrap: overdue/due-soon todos, inbox, recent journal, note activity.

    Call this once at the start of every assistant session before doing
    anything else — it is the fastest way to load current context.
    """
    today = date.today().isoformat()
    week = (date.today() + timedelta(days=7)).isoformat()
    open_todos = store.todo_list()
    overdue = [m for m in open_todos if m.get("due") and str(m["due"]) < today]
    due_today = [m for m in open_todos if str(m.get("due") or "") == today]
    due_week = [m for m in open_todos if m.get("due") and today < str(m["due"]) <= week]
    waiting = [m for m in open_todos if m.get("status") == "waiting"]
    notes = store.note_list()[:5]
    note_lines = "\n".join(f"- {m.get('title')} (id: {m.get('id')})" for m in notes) or "(none)"
    return f"""# zass briefing — {today}

## Overdue
{_fmt_todos(overdue)}

## Due today
{_fmt_todos(due_today)}

## Due this week
{_fmt_todos(due_week)}

## Waiting on others
{_fmt_todos(waiting)}

## Open todos: {len(open_todos)} total

## Inbox
{store.inbox_read()}

## Journal (last 2 days)
{store.journal_read(days=2)}

## Recently updated notes
{note_lines}
"""


@mcp.tool()
def journal_log(entry: str) -> str:
    """Append a timestamped entry to today's journal.

    Use for continuity across sessions: decisions made, actions taken on the
    user's behalf (messages sent, emails handled), and important events. Write
    entries so a future session with no other context understands them.
    """
    path = store.journal_log(entry)
    return f"Logged to {path}"


@mcp.tool()
def journal_read(days: int = 3) -> str:
    """Read the last N days of journal entries (default 3)."""
    return store.journal_read(days=days)


# ---------------------------------------------------------------- capture


@mcp.tool()
def capture(text: str) -> str:
    """Quick-capture a thought, task, or link into the inbox for later triage.

    Use when the user mentions something in passing that shouldn't be lost but
    doesn't warrant a full todo/note yet. Process the inbox with the /triage skill.
    """
    path = store.capture(text)
    return f"Captured to {path}"


@mcp.tool()
def inbox_read() -> str:
    """Read the quick-capture inbox queue."""
    return store.inbox_read()


# ---------------------------------------------------------------- todos


@mcp.tool()
def todo_add(
    title: str,
    due: str | None = None,
    priority: str = "normal",
    tags: list[str] | None = None,
    notes: str = "",
    status: str = "open",
) -> str:
    """Create a todo.

    due: ISO date (YYYY-MM-DD). priority: high|normal|low.
    status: open|waiting|scheduled (use 'waiting' when blocked on someone else).
    """
    if status not in store.TODO_STATUSES:
        return f"Invalid status '{status}'. Use one of: {', '.join(store.TODO_STATUSES)}"
    meta = store.todo_add(title, due=due, priority=priority, tags=tags, notes=notes, status=status)
    return f"Created todo `{meta['id']}`: {meta['title']}" + (f" (due {due})" if due else "")


@mcp.tool()
def todo_list(
    status: str | None = None,
    tag: str | None = None,
    due_within_days: int | None = None,
    include_done: bool = False,
) -> str:
    """List todos, sorted by due date then priority.

    status filters to one of open|waiting|scheduled|done|dropped; by default
    done/dropped are hidden. due_within_days limits to items due in that window.
    """
    rows = store.todo_list(
        status=status, tag=tag, due_within_days=due_within_days, include_done=include_done
    )
    return _fmt_todos(rows, empty="(no matching todos)")


@mcp.tool()
def todo_update(
    id: str,
    title: str | None = None,
    status: str | None = None,
    due: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
    append_note: str | None = None,
) -> str:
    """Update fields on a todo by id, and/or append a dated progress note."""
    if status is not None and status not in store.TODO_STATUSES:
        return f"Invalid status '{status}'. Use one of: {', '.join(store.TODO_STATUSES)}"
    meta = store.todo_update(
        id, title=title, status=status, due=due, priority=priority, tags=tags,
        append_note=append_note,
    )
    if meta is None:
        return f"No todo found matching id '{id}'. Use todo_list to see ids."
    return f"Updated todo `{meta['id']}` — status: {meta.get('status')}"


@mcp.tool()
def todo_done(id: str) -> str:
    """Mark a todo as done."""
    meta = store.todo_update(id, status="done")
    if meta is None:
        return f"No todo found matching id '{id}'. Use todo_list to see ids."
    return f"Done: {meta.get('title')} ✓"


# ---------------------------------------------------------------- notes


@mcp.tool()
def note_add(title: str, body: str, tags: list[str] | None = None) -> str:
    """Create a note (markdown body). Use notes for reference material,
    meeting notes, research, plans — anything longer-lived than a todo."""
    meta = store.note_add(title, body, tags=tags)
    return f"Created note `{meta['id']}`: {meta['title']}"


@mcp.tool()
def note_get(id: str) -> str:
    """Read a note by id (filename stem, forgiving partial match)."""
    doc = store.note_get(id)
    if doc is None:
        return f"No note found matching id '{id}'. Use note_list to see ids."
    meta, body = doc
    tags = " #".join(meta.get("tags") or [])
    header = f"# {meta.get('title')} (id: {meta.get('id')})"
    if tags:
        header += f"\n#{tags}"
    return f"{header}\n\n{body.strip()}"


@mcp.tool()
def note_update(
    id: str,
    body: str | None = None,
    append: str | None = None,
    title: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update a note: replace body, or append to it, or retitle/retag."""
    meta = store.note_update(id, body=body, append=append, title=title, tags=tags)
    if meta is None:
        return f"No note found matching id '{id}'. Use note_list to see ids."
    return f"Updated note `{meta['id']}`"


@mcp.tool()
def note_list(tag: str | None = None) -> str:
    """List notes (most recently updated first), optionally filtered by tag."""
    rows = store.note_list(tag=tag)
    if not rows:
        return "(no notes)"
    return "\n".join(
        f"- **{m.get('title')}** (id: {m.get('id')})"
        + (" #" + " #".join(m["tags"]) if m.get("tags") else "")
        for m in rows
    )


# ---------------------------------------------------------------- people


@mcp.tool()
def person_upsert(
    name: str,
    relation: str | None = None,
    emails: list[str] | None = None,
    phones: list[str] | None = None,
    birthday: str | None = None,
    tags: list[str] | None = None,
    note: str | None = None,
) -> str:
    """Create or enrich a person profile. Lists merge (never overwrite);
    `note` appends a dated entry to their log.

    Use whenever you learn something durable about a person in the user's
    life: who they are, contact info, preferences, life events.
    """
    meta = store.person_upsert(
        name, relation=relation, emails=emails, phones=phones,
        birthday=birthday, tags=tags, note=note,
    )
    return f"Saved person `{meta['id']}`: {meta['name']}"


@mcp.tool()
def person_get(id: str) -> str:
    """Read a person's full profile and log by id or name fragment."""
    doc = store.person_get(id)
    if doc is None:
        return f"No person found matching '{id}'. Use person_list to see everyone."
    meta, body = doc
    lines = [f"# {meta.get('name')} (id: {meta.get('id')})"]
    for field in ("relation", "emails", "phones", "birthday", "tags"):
        if meta.get(field):
            lines.append(f"- {field}: {meta[field]}")
    if body.strip():
        lines.append("\n" + body.strip())
    return "\n".join(lines)


@mcp.tool()
def person_log(id: str, entry: str) -> str:
    """Append a dated entry to a person's log (interactions, facts learned, gift ideas)."""
    meta = store.person_log(id, entry)
    if meta is None:
        return f"No person found matching '{id}'. Use person_upsert to create them."
    return f"Logged on {meta.get('name')}"


@mcp.tool()
def person_list(query: str | None = None) -> str:
    """List people, optionally filtered by a search string over name/relation/notes."""
    rows = store.person_list(query=query)
    if not rows:
        return "(no people saved yet)"
    return "\n".join(
        f"- **{m.get('name')}** (id: {m.get('id')})"
        + (f" — {m['relation']}" if m.get("relation") else "")
        for m in rows
    )


# ---------------------------------------------------------------- memory


@mcp.tool()
def memory_save(content: str, slug: str | None = None, kind: str = "fact", subject: str | None = None) -> str:
    """Save a durable memory: facts, preferences, routines, context about the
    user or their world. Overwrites if the slug already exists (use that to
    correct stale facts).

    kind: fact|preference|routine|context. subject: who/what it's about (default 'user').
    Save proactively — anything a great human assistant would remember.
    """
    meta = store.memory_save(content, slug=slug, kind=kind, subject=subject)
    return f"Saved memory `{meta['id']}` ({meta['type']})"


@mcp.tool()
def memory_list(query: str | None = None) -> str:
    """List memories with their content, optionally filtered by a search string."""
    rows = store.memory_list(query=query)
    if not rows:
        return "(no memories saved yet)"
    return "\n\n".join(
        f"**{m.get('id')}** ({m.get('type')}, subject: {m.get('subject')})\n{body}"
        for m, body in rows
    )


@mcp.tool()
def memory_forget(id: str) -> str:
    """Delete a memory that is wrong or no longer relevant."""
    if store.memory_forget(id):
        return f"Forgot `{id}`"
    return f"No memory found matching id '{id}'."


# ---------------------------------------------------------------- search


@mcp.tool()
def search(query: str, kind: str | None = None) -> str:
    """Search all assistant state (todos, notes, people, memory, journal, inbox)
    for a substring. kind optionally limits to one of: todos|notes|people|memory|journal.
    """
    hits = store.search(query, kind=kind)
    if not hits:
        return f"No matches for '{query}'."
    return "\n".join(f"- `{h['file']}:{h['line_no']}` {h['line']}" for h in hits)


# ---------------------------------------------------------------- macOS connectors


@mcp.tool()
def imessage_send(to: str, message: str) -> str:
    """Send an iMessage via the local Messages app (macOS only).

    `to` is a phone number (+15551234567) or iMessage email. IMPORTANT: always
    show the user the exact recipient and message text and get their explicit
    confirmation before calling this — it sends immediately and is irreversible.
    Log every sent message with journal_log.
    """
    try:
        apple.send_imessage(to, message)
    except Exception as e:  # surface AppleScript/permission errors to the agent
        return f"Failed to send: {e}"
    return f"Sent iMessage to {to}."


@mcp.tool()
def contacts_search(query: str) -> str:
    """Search the local macOS Contacts app by name. Returns names, emails, phones.

    Use to resolve 'text Mom' into a real phone number. Consider saving
    frequently used contacts as zass people via person_upsert.
    """
    try:
        return apple.contacts_search(query)
    except Exception as e:
        return f"Contacts search failed: {e}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
