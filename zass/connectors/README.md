# External connectors

zass does not reimplement email/calendar clients — it wires existing MCP
servers into `.mcp.json`. The internal `zass` server already covers state
plus two local macOS connectors (`imessage_send`, `contacts_search`) that
need no auth beyond one-time macOS Automation permission prompts.

Add a connector by merging its block into `.mcp.json` at the repo root
(claude-code auto-loads it), or `claude mcp add <name> -- <command>`.
For Codex, add the equivalent to `~/.codex/config.toml`.

## Gmail

[`@gongrzhe/server-gmail-autoauth-mcp`](https://github.com/GongRzhe/Gmail-MCP-Server)
— send/read/search/label. One-time OAuth: create a Google Cloud OAuth client
(Desktop app), save it as `~/.gmail-mcp/gcp-oauth.keys.json`, then run
`npx @gongrzhe/server-gmail-autoauth-mcp auth` once. Then:

```json
"gmail": {
  "command": "npx",
  "args": ["@gongrzhe/server-gmail-autoauth-mcp"]
}
```

## Google Calendar

[`@cocal/google-calendar-mcp`](https://github.com/nspady/google-calendar-mcp)
— list/create/update events, availability. Same OAuth pattern:

```json
"gcal": {
  "command": "npx",
  "args": ["@cocal/google-calendar-mcp"],
  "env": { "GOOGLE_OAUTH_CREDENTIALS": "/Users/zubinaysola/.gcal-mcp/gcp-oauth.keys.json" }
}
```

Tip: one Google Cloud project can hold both OAuth clients (enable the Gmail
API and the Calendar API on it).

## Reading iMessage history (sending is already built in)

[`mac_messages_mcp`](https://github.com/carterlasalle/mac_messages_mcp) reads
the local Messages database. Requires granting Full Disk Access to the
terminal/agent process:

```json
"imessage-history": {
  "command": "uvx",
  "args": ["mac-messages-mcp"]
}
```

## Apple Notes / Reminders (import sources)

zass keeps its own notes/todos in `state/` on purpose (portable, greppable,
git-versioned). If existing data lives in Apple Notes/Reminders, use an MCP
like `apple-mcp` to read it once and migrate into zass, rather than running
two systems of record.

## Anything else

Search the MCP registry (`claude mcp` docs, or the mcp-registry tools if
available) for Slack, WhatsApp, Notion, Spotify, etc. Prefer connectors that
run locally over ones that proxy your data through third-party servers.

## Conventions for new connectors

1. Add the server block to `.mcp.json` and document setup here.
2. Note in `CLAUDE.md` only if it changes safety rules (anything that can
   send/delete gets the confirm-before-acting rule by default).
3. If a connector needs secrets, keep them OUT of this repo (use `env` blocks
   pointing at files in `$HOME`, as the examples above do).
