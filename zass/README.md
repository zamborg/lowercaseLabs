# zass

A portable personal-assistant harness. Drop this directory into any coding
agent — Claude Code, Codex, anything that can read files — and it becomes a
persistent personal assistant with memory, todos, notes, people, and
messaging, running on your agent subscription instead of API tokens.

## Design

**The filesystem is the database.** All state is markdown + YAML frontmatter
under `state/`, git-versioned. The `zass` MCP server is a structured,
convenient interface over those files — but any agent can also just read and
grep them. That makes the whole assistant portable: migration is `git clone`,
backup is `git push`, and no runtime is load-bearing.

```
zass/
├── CLAUDE.md            # operating manual (Claude Code entrypoint)
├── AGENTS.md            # same, for Codex and others
├── .mcp.json            # auto-registers the zass MCP server
├── mcp/zass/            # internal MCP server (Python, uv, official MCP SDK)
├── state/               # THE DATA
│   ├── inbox.md         # quick-capture queue
│   ├── todos/ notes/ people/ memory/ journal/
├── .claude/skills/      # playbooks: brief, triage, capture, weekly-review
├── connectors/          # recipes for external MCPs (Gmail, Calendar, ...)
└── scripts/setup.sh
```

## Quickstart

```bash
./scripts/setup.sh   # uv sync + smoke test
claude               # run claude-code here; .mcp.json wires everything up
```

Then try: `/brief`, "capture: book flights for August", `/triage`.

## Primitives (23 MCP tools)

| Domain | Tools |
|---|---|
| Session | `briefing`, `journal_log`, `journal_read` |
| Capture | `capture`, `inbox_read` |
| Todos | `todo_add`, `todo_list`, `todo_update`, `todo_done` |
| Notes | `note_add`, `note_get`, `note_update`, `note_list` |
| People | `person_upsert`, `person_get`, `person_log`, `person_list` |
| Memory | `memory_save`, `memory_list`, `memory_forget` |
| Search | `search` (greps all state) |
| macOS | `imessage_send`, `contacts_search` (local, no OAuth) |

External connectors (Gmail, Google Calendar, iMessage history) are wired via
`.mcp.json` — see [connectors/README.md](connectors/README.md).

## Extending

- New workflow over existing tools → add a skill in `.claude/skills/`.
- New primitive → add a tool in `mcp/zass/zass_mcp/server.py` (+ store fn + test).
- New external capability → add an MCP server block to `.mcp.json`.

## Development

```bash
uv run --project mcp/zass pytest mcp/zass/tests -q   # store tests
uv run --project mcp/zass zass-mcp                   # run server (stdio)
```
