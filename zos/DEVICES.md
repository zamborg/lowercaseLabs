# ZOS Devices

**Turn any local stdio MCP server into a ZOS device.**

A ZOS **device** is a normal stdio MCP server running on your own hardware (a Mac, a Linux workstation, a Raspberry Pi) that dials out to your ZOS server over a persistent WebSocket (`/actuator`). While the bridge is running, its tools appear to every agent connected to your ZOS as `<slug>_<tool>`. When you stop it or close your laptop, its tools disappear cleanly.

ZOS never needs inbound access to your machine — only an outbound WebSocket connection to `wss://<your-zos-host>/actuator`.

---

## 1. Add a Device in `/admin`

1. Open **`https://<your-zos-host>/admin/modules`** and scroll to **Runs on your machine**.
2. Enter a slug (lowercase letters, digits, or hyphens — e.g. `mac`, `studio-mac`, `home-lab`) and click **Add device**.
   - Slugs are unique across all modules, connectors, and devices (`zos` is reserved).
3. Copy the one-time token (shown once — never paste it into AI chat), along with the ready-to-run `zos-device` command and `config.toml` snippet.

*(You can also manage devices from the server CLI: `zos device add <slug>`, `list`, `rotate`, `disable`, `enable`, `remove` — or the backwards-compatible `zos actuator add <slug>`.)*

---

## 2. Bridge Any Local Stdio MCP Server

Install `zos-device` from the `zos-v3.2.8` release:

```bash
git clone --depth 1 --branch zos-v3.2.8 https://github.com/zamborg/lowercaseLabs.git
cd lowercaseLabs/zos/device
uv tool install .
```

Then run **any** standard stdio MCP server (Python, Node, Go, Rust, etc.) behind `zos-device`:

```bash
zos-device \
  --url wss://my-zos.exe.xyz/actuator \
  --token zos_a_... \
  -- your-mcp-server --flag value
```

Or keep the token out of shell history and process lists using a `0600` TOML config file:

```toml
url = "wss://my-zos.exe.xyz/actuator"
token = "zos_a_..."
command = ["your-mcp-server", "--flag", "value"]
```

```bash
chmod 600 device.toml
zos-device --config device.toml
```

*(You can also pass `-- <command...>` alongside `--config` to override only the command, or set `ZOS_DEVICE_URL` / `ZOS_DEVICE_TOKEN` in the environment.)*

### What `zos-device` does

- Connects outbound to `/actuator` with `Authorization: Bearer <token>`.
- Spawns your stdio MCP server as a child process and relays JSON-RPC frames bidirectionally (zero tool logic in the bridge).
- Reconnects automatically with jittered exponential backoff if the network drops or ZOS restarts, spawning a fresh stdio session per connection so MCP `initialize` is always clean.
- When `zos-device` stops, ZOS immediately removes the device's tools and notifies connected MCP clients (`notifications/tools/list_changed`).

---

## 3. Tool Tiers & Approvals

Device tools require **zero** server-side manifest or hardcoded seed rows. ZOS assigns each discovered device tool a default tier from its standard MCP `ToolAnnotations`:

| MCP annotation on the device tool | Default ZOS tier | Default behavior |
|---|---|---|
| `readOnlyHint: true` | `read` | Allowed automatically (visible to all clients) |
| Anything else (`readOnlyHint: false` or omitted) | `external` | Requires explicit owner approval (hidden from `untrusted` clients) |

Explicit owner tier overrides in **`/admin/policy`** (or `zos policy set`) **always win** over defaults.

---

## 4. Reference Example: `zos-mac` (`devices/mac/`)

`zos-mac` (`zos/devices/mac/`, v0.3.1) is a standalone macOS stdio MCP server for Contacts, full iMessage/SMS, and a shell. ZOS prefixes each tool with the device slug (`mac_`):

| Tool | Hint → ZOS tier | What it does |
|---|---|---|
| `contacts_search(query, limit)` | `readOnlyHint: true` → `read` | Fuzzy match (typo-tolerant) over name, nickname, organization, or an exact phone/email. Returns contact ids with labeled phones/emails. |
| `contacts_get(id)` | `readOnlyHint: true` → `read` | One contact with all labeled phones and emails. |
| `messages_threads(limit, since?, query?)` | `readOnlyHint: true` → `read` | Recent conversations, newest first: chat id, display/group name, participants, last message time + snippet, `is_group`, unread count. |
| `messages_read(chat, limit, before?)` | `readOnlyHint: true` → `read` | One conversation in chronological order (`chat` = chat id, phone/email, or contact id), paginated via `next_before`. |
| `messages_search(query, handle?, since?, limit)` | `readOnlyHint: true` → `read` | Incremental FTS5 search over `chat.db`; results include `chat_id` for `messages_read`. |
| `messages_send(to, text)` | none → `external` (approval required) | Sends via macOS Shortcuts (`ZOS Send Message`, recommended for 1:1 handles) or Messages.app AppleScript (`osascript`, required for group `chat_id`s). `to` = phone, email, chat id, or contact id; ambiguous contacts return options instead of guessing. Confirms by polling `chat.db` and reports `sent` / `unconfirmed`, `chat_id`, and `backend`. |
| `shell_run` | `readOnlyHint: false` → `external` (approval required) | Runs commands in a persistent isolated `tmux -L zos` shell pane. |

"Text Jared" workflow for any connected agent: `mac_contacts_search` → (optionally `mac_messages_threads` / `mac_messages_read`) → `mac_messages_send` (approved by you in ZOS). See [`devices/mac/README.md`](devices/mac/README.md).

Because `zos-mac` is just a standard stdio MCP server, you can also plug it directly into **Claude Desktop** (without ZOS):

```json
{
  "mcpServers": {
    "mac": {
      "command": "/Users/<you>/.zos-actuator/venv/bin/zos-mac"
    }
  }
}
```

---

## 5. Installing `zos-mac` on Your Mac (Against a Hosted ZOS)

Anyone running their own hosted ZOS (for example `https://my-zos.exe.xyz` on exe.dev) can attach their Mac in a few minutes:

1. **Prerequisites on the Mac:** `brew install uv tmux`.
2. **Add the device in ZOS:** Open `https://my-zos.exe.xyz/admin/modules` → **Runs on your machine** → add slug `mac` → copy the one-time token (`zos_a_...`).
3. **Install `zos-device`, `zos-mac`, and the LaunchAgents:**
   ```bash
   git clone --depth 1 --branch zos-v3.2.8 https://github.com/zamborg/lowercaseLabs.git
   cd lowercaseLabs/zos
   ./deploy/launchd/install.sh
   ```
   This provisions a dedicated Python 3.12 interpreter at `~/.zos-actuator/python/bin/python3.12` (so macOS Full Disk Access is scoped strictly to this binary, not shared with other Python scripts), installs both `zos-device` and `zos-mac` into `~/.zos-actuator/venv`, and installs the `com.zos.tmux` and `com.zos.actuator` LaunchAgents.
4. **Configure `~/.zos-actuator/config.toml` (`chmod 600`):**
   ```toml
   url = "wss://my-zos.exe.xyz/actuator"
   token = "zos_a_YOUR_TOKEN_HERE"
   ```
   *(The `com.zos.actuator` LaunchAgent launches `zos-device --config ~/.zos-actuator/config.toml -- ~/.zos-actuator/venv/bin/python -m zos_mac --config ~/.zos-actuator/config.toml`, keeping `zos-mac` running under the dedicated interpreter so macOS TCC attributes Full Disk Access to it.)*
5. **Grant macOS permissions (TCC) and set up sending:**
   - **Full Disk Access** (reading Messages & Contacts): **System Settings → Privacy & Security → Full Disk Access** → add `~/.zos-actuator/python/bin/python3.12` (exact resolved path printed by `install.sh`).
   - **Shortcuts backend (recommended for `mac_messages_send` 1:1 sends):** Import [`devices/mac/shortcuts/ZOS Send Message.shortcut`](devices/mac/shortcuts/ZOS%20Send%20Message.shortcut) (or build it manually — see [`devices/mac/README.md`](devices/mac/README.md)) keeping the exact name `ZOS Send Message`. Under the default `send_backend = "auto"`, `zos-mac` uses it automatically for 1:1 handles; on the first send, click **Always Allow** on Shortcuts' own privacy prompt.
   - **AppleScript backend (needed for group chats / `chat_id` sends):** Requires **Automation → Messages** (**System Settings → Privacy & Security → Automation → python3.12 → Messages**). Set `send_backend = "shortcuts"` in `~/.zos-actuator/config.toml` to disable AppleScript fallback, or `"applescript"` to force it. Without it, 1:1 Shortcut sends and all reads keep working. ZOS never bypasses TCC.
6. **Start the LaunchAgents:**
   ```bash
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.tmux.plist || true
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.actuator.plist || true
   ```
7. **Verify:** Refresh `https://my-zos.exe.xyz/admin/modules` — under **Runs on your machine**, `mac` should show **`online`** with **`7`** tools (`mac_contacts_search`, `mac_contacts_get`, `mac_messages_threads`, `mac_messages_read`, `mac_messages_search` at `read` tier; `mac_messages_send` and `mac_shell_run` at `external` tier).

---

## 6. Web Browser Device (`web` — Playwright MCP, `devices/web/`)

Give your agents a real Chromium browser running on your Mac in a **dedicated, isolated profile**, powered by Microsoft's [`@playwright/mcp`](https://github.com/microsoft/playwright-mcp) (Apache-2.0, pinned to `0.0.86` in [`devices/web/VERSION`](devices/web/VERSION)).

### Isolation guarantees

- **Completely separate from personal browsers:** Runs Playwright's bundled Chromium (`--browser chromium`, Chrome for Testing) installed under `~/.zos-web/browsers` with a dedicated profile at `~/.zos-web/profile` (`chmod 700`). Never launches system Chrome, never uses `--extension`, and never attaches to your personal Chrome or Safari cookies/passwords.
- **Separate device & LaunchAgent:** Registered as slug `web` (tools appear as `web_browser_*`) with its own `zos-device` venv, token, and `com.zos.web` LaunchAgent under `~/.zos-web/`. Never touches `~/.zos-actuator` or `com.zos.actuator`.
- **Hardened flags:** Passes `--no-webmcp` (prevents untrusted websites from injecting WebMCP tools into ZOS) and runs with `"allowUnrestrictedFileAccess": false`, `--ignore-roots`, `--output-dir ~/.zos-web/output`, and `WorkingDirectory = ~/.zos-web/output` (blocks `file://` URLs and restricts file uploads/output strictly to `~/.zos-web/output`).

### Install & login flow

1. **Prerequisites:** `uv` (`brew install uv`) and Node.js `>= 20` (Homebrew or `nvm` — the installer resolves and embeds an absolute `node` path so `launchd` works without `nvm` in `PATH`).
2. **Register the `web` device on your ZOS host:** Open `https://my-zos.exe.xyz/admin/modules` → **Runs on your machine** → add slug **`web`** (or run `zos device add web` on the server) and copy the one-time token.
3. **Install on your Mac:**
   ```bash
   git clone --depth 1 --branch zos-v3.2.8 https://github.com/zamborg/lowercaseLabs.git
   cd lowercaseLabs/zos
   ./deploy/launchd/install-web.sh
   ```
4. **Configure `~/.zos-web/config.toml` (`chmod 600`):** Set `url = "wss://my-zos.exe.xyz/actuator"` and `token = "zos_a_..."`.
5. **Start the LaunchAgent:**
   ```bash
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.web.plist || true
   ```
6. **Manual logins & headless mode:** Headed by default (`"headless": false` in `~/.zos-web/playwright-mcp.json`), so the Chromium window appears on your logged-in desktop when an agent uses it. Log into whichever sites you choose inside that dedicated window — sessions persist in `~/.zos-web/profile`. To run headless, set `"headless": true` in `~/.zos-web/playwright-mcp.json` and run `launchctl kickstart -k gui/$(id -u)/com.zos.web`.

### Tool tiers & prompt-injection note (`@playwright/mcp@0.0.86`)

ZOS applies its standard device defaults from upstream `readOnlyHint` (zero allow overrides):

- **`read` (7 tools):** `web_browser_console_messages`, `web_browser_find`, `web_browser_network_request`, `web_browser_network_requests`, `web_browser_snapshot`, `web_browser_take_screenshot`, `web_browser_wait_for`.
- **`external` (18 tools — approval required):** `web_browser_click`, `web_browser_close`, `web_browser_drag`, `web_browser_drop`, `web_browser_emulate_media`, `web_browser_evaluate`, `web_browser_file_upload`, `web_browser_fill_form`, `web_browser_handle_dialog`, `web_browser_hover`, `web_browser_navigate`, `web_browser_navigate_back`, `web_browser_press_key`, `web_browser_resize`, `web_browser_run_code_unsafe`, `web_browser_select_option`, `web_browser_tabs`, `web_browser_type`.

> **Prompt-injection risk:** Webpages an agent reads via snapshots or screenshots are untrusted input and may contain hidden instructions trying to steer the agent into clicking, submitting forms, or exfiltrating session data. Keep all navigation and interaction tools at `external` (approval required) — do not add blanket `allow` overrides.

### Wiping the profile

To clear all logins, cookies, and browser state:

```bash
launchctl bootout gui/$(id -u)/com.zos.web || true
rm -rf ~/.zos-web/profile && mkdir -m 700 ~/.zos-web/profile
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.web.plist
```

See [`devices/web/README.md`](devices/web/README.md) for the full reference.

---

## Managing Devices in `/admin/modules`

Every action under **Runs on your machine** is audited in `/admin/activity`:

- **Rotate token:** Revokes old tokens, disconnects any live WebSocket immediately, and shows a fresh token once.
- **Disable / Enable:** Closes any active connection and hides the device's tools while keeping the device row and tokens intact until re-enabled.
- **Remove:** Two-step confirm; disconnects any active connection, revokes all tokens, and deletes the device row while preserving audit history.
