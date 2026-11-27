# zos-mac

Standalone macOS stdio MCP server for ZOS: Contacts, full iMessage/SMS, and a persistent
tmux shell. It works directly in any stdio MCP client (e.g. Claude Desktop) or behind
[`zos-device`](../../device/), which relays it to ZOS. ZOS prefixes every tool with the
device slug, e.g. `mac_messages_send`. Install and setup: [DEVICES.md](../../DEVICES.md).

## Tools

| Tool | Annotations | ZOS default tier |
|---|---|---|
| `contacts_search(query, limit=10)` | `readOnlyHint: true` | `read` |
| `contacts_get(id)` | `readOnlyHint: true` | `read` |
| `messages_threads(limit=20, since?, query?)` | `readOnlyHint: true` | `read` |
| `messages_read(chat, limit=50, before?)` | `readOnlyHint: true` | `read` |
| `messages_search(query, handle?, since?, limit=20)` | `readOnlyHint: true` | `read` |
| `messages_send(to, text)` | no `readOnlyHint` | `external` (approval required) |
| `shell_run(command, timeout_s, reset)` | `readOnlyHint: false` | `external` (approval required) |

**Texting someone by name:** `contacts_search` → (optionally `messages_threads` /
`messages_read` for context) → `messages_send`.

- **Contact ids** are the AddressBook `ZUNIQUEID` (or `<source>:<pk>` when absent). Phone
  and email labels are cleaned (`_$!<Mobile>!$_` → `mobile`; custom labels kept).
- **Chat ids** are `chat.guid` values (`iMessage;-;+14155550101`, `iMessage;+;chat123…`).
  `messages_threads`, `messages_read`, and `messages_search` all return them.
- **`messages_read(chat)`** accepts a chat id, a group `chat_identifier`, a phone/email, or a
  contact id. For a person, all their 1:1 chats (e.g. iMessage + SMS) are merged. Pass the
  returned `next_before` as `before` for older pages; tapbacks are skipped.
- **`messages_send(to, text)`**:
  - `to` is a phone, an email, a chat id (needed for group chats), or a contact id. Names are
    rejected; resolve them with `contacts_search` first.
  - A contact with one handle uses it. With several, the handle from the most recent 1:1
    conversation wins. If none (or a tie), the tool returns an error listing the options and
    sends nothing.
  - Sends through Messages.app with `osascript`: iMessage first, then SMS if the iMessage
    attempt fails and an SMS account is available. Group chats send to the chat id directly.
  - The recipient and text are passed only as AppleScript `on run argv` items, never
    interpolated into script source.
  - Text is capped at 4000 characters; empty `to`/`text` are rejected.
  - After sending, polls `chat.db` (~5 s) for the new outgoing row: `status: "sent"` with
    `chat_id` and `message_id`, or `"unconfirmed"` with a note (check Messages before
    retrying to avoid duplicates).

## macOS permissions (TCC)

Both are granted to the dedicated interpreter `~/.zos-actuator/python/bin/python3.12`:

1. **Full Disk Access** — reading `~/Library/Messages/chat.db` and the AddressBook SQLite
   stores (`~/Library/Application Support/AddressBook`). Without it, read tools return an
   error naming the exact interpreter path.
2. **Automation → Messages** — required only by `messages_send`. macOS prompts the first time
   a send runs; if denied, sends fail with a hint pointing to **System Settings → Privacy &
   Security → Automation**.

zos-mac never tries to bypass TCC.

## Configuration

`--config` TOML (or CLI flags) keys: `chat_db_path`, `index_db_path`, `addressbook_root`,
`tmux_socket`, `tmux_session`.

## Development

```bash
uv run pytest
uv run ruff check
```

Tests use only synthetic fixture databases (`tests/conftest.py`) and a mocked `osascript`
runner. An autouse guard redirects the default AddressBook path, so tests never read real
Contacts, Messages, or send anything. The only real `osascript` call in tests runs an
echo-only script (no app access) to check argv round-tripping.
