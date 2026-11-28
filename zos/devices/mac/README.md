# zos-mac

Standalone macOS stdio MCP server for ZOS (`v0.3.1`): Contacts, full iMessage/SMS, and a
persistent tmux shell. It works directly in any stdio MCP client (e.g. Claude Desktop) or
behind [`zos-device`](../../device/), which relays it to ZOS. ZOS prefixes every tool with
the device slug, e.g. `mac_messages_send`. Install and setup: [DEVICES.md](../../DEVICES.md).

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
  - Text is capped at 4000 characters; empty `to`/`text` are rejected.
  - After sending, polls `chat.db` (~5 s) for the new outgoing row: `status: "sent"` with
    `chat_id`, `message_id`, and `backend`, or `"unconfirmed"` with a note (check Messages
    before retrying to avoid duplicates).

## Send backends (`send_backend = "auto" | "shortcuts" | "applescript"`)

Default is `send_backend = "auto"` (set in `~/.zos-actuator/config.toml` or via
`--send-backend`).

1. **Shortcuts (recommended for 1:1 handles):**
   - Uses `/usr/bin/shortcuts run "ZOS Send Message" --input-path <tmp.json>` with a private
     `0600` temp file containing `{"to": "<handle>", "text": "<message>"}` (deleted right
     afterwards; never logged).
   - Runs inside Apple's own Shortcuts app, avoiding macOS's Automation → Messages TCC bug on
     bare launchd Python binaries.
   - Under `auto`, `zos-mac` automatically uses Shortcuts whenever `shortcuts list` contains
     `ZOS Send Message` (cached briefly), and falls back to AppleScript otherwise.
   - On the first send, macOS shows Shortcuts' own privacy prompt — click **Always Allow**.
2. **AppleScript (`osascript` → Messages.app):**
   - Required for **group chats** (and any explicit `chat_id` target), because Shortcuts'
     Send Message action cannot address a chat by id.
   - Passes recipient and text strictly via `on run argv` (never interpolated into script
     source); tries iMessage first, then SMS.
   - Requires the macOS **Automation → Messages** grant for `~/.zos-actuator/python/bin/python3.12`.

### Installing the `ZOS Send Message` Shortcut

**Option A — Import the signed `.shortcut` file (fastest):**

1. Double-click [`shortcuts/ZOS Send Message.shortcut`](shortcuts/ZOS%20Send%20Message.shortcut)
   (signed with `shortcuts sign --mode anyone` from the committed XML source
   [`shortcuts/ZOS Send Message.plist`](shortcuts/ZOS%20Send%20Message.plist); contains zero
   personal data).
2. Click **Add Shortcut** in Shortcuts.app, keeping the exact name **`ZOS Send Message`**.
3. On your first `mac_messages_send` call, Shortcuts prompts to allow sending messages —
   choose **Always Allow**.

**Option B — Build it manually in Shortcuts.app (fallback):**

1. Open **Shortcuts.app** and create a new shortcut named exactly **`ZOS Send Message`**.
2. Add action **Get Dictionary from Input** — set input to **Shortcut Input**.
3. Add action **Get Dictionary Value** — set *Get* **Value** for key **`to`** in **Dictionary**
   (output of step 2).
4. Add action **Get Dictionary Value** — set *Get* **Value** for key **`text`** in **Dictionary**
   (output of step 2).
5. Add action **Send Message**:
   - Set **Message** to the Dictionary Value from step 4 (`text`).
   - Set **Recipients** to the Dictionary Value from step 3 (`to`) (right-click Recipients →
     *Select Magic Variable* → Dictionary Value).
   - Expand the disclosure arrow and turn **Show When Run** **OFF**.

## macOS permissions (TCC)

1. **Full Disk Access** (required for reading Messages & Contacts) — grant to the dedicated
   interpreter `~/.zos-actuator/python/bin/python3.12` in **System Settings → Privacy &
   Security → Full Disk Access**.
2. **Shortcuts privacy prompt** (recommended for 1:1 sending) — triggered automatically on the
   first `ZOS Send Message` send; click **Always Allow**.
3. **Automation → Messages** (only needed for group chats or the AppleScript backend) —
   **System Settings → Privacy & Security → Automation → python3.12 → Messages**.

zos-mac never tries to bypass TCC.

## Configuration

`--config` TOML (or CLI flags) keys: `chat_db_path`, `index_db_path`, `addressbook_root`,
`tmux_socket`, `tmux_session`, `send_backend` (`auto` | `shortcuts` | `applescript`).

## Development

```bash
uv run pytest
uv run ruff check
```

Tests use only synthetic fixture databases (`tests/conftest.py`) and mocked subprocess runners
(with autouse guards preventing real AddressBook reads, `shortcuts`, or `osascript` sends).
The only real `osascript` call in tests runs an echo-only script (no app access) to check argv
round-tripping.
