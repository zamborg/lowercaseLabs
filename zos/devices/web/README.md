# ZOS Web Device (`web` — Playwright MCP)

Gives agents connected to ZOS a real Chromium browser running on your Mac inside a **completely isolated browser profile**, backed by Microsoft's [`@playwright/mcp`](https://github.com/microsoft/playwright-mcp) (Apache-2.0, pinned in [`VERSION`](VERSION) to `0.0.86`).

Agents **never** touch your personal Chrome or Safari profiles, cookies, bookmarks, or saved passwords.

---

## Isolation & security model

- **Separate browser binary:** Launches Playwright's own bundled Chromium (`--browser chromium`, Chrome for Testing) installed under `~/.zos-web/browsers` via `PLAYWRIGHT_BROWSERS_PATH`. Never launches your system Chrome channel, never uses `--extension`, and never attaches via CDP to your personal browser.
- **Separate persistent profile:** Stored at `~/.zos-web/profile` (`chmod 700`) via `--user-data-dir`. You sign into whichever sites you want agents to use inside this dedicated window; those logins persist across restarts without exposing your personal browser.
- **Separate device & LaunchAgent:** Runs under slug `web` (tools appear as `web_browser_*`) via `com.zos.web` with its own `zos-device` venv and device token in `~/.zos-web/`. Completely separate from `com.zos.actuator` and `~/.zos-actuator`.
- **No page-injected MCP tools:** Passes `--no-webmcp` so untrusted web pages cannot register WebMCP tools into ZOS.
- **Restricted local file access:** Runs with `"allowUnrestrictedFileAccess": false`, `--ignore-roots`, `--output-dir ~/.zos-web/output`, and `WorkingDirectory = ~/.zos-web/output`. This blocks `file://` URL navigation and restricts `browser_file_upload` / output files strictly to `~/.zos-web/output`. In addition, `web_browser_file_upload` is `external`-tier (requires your explicit approval).
- **Prompt-injection warning:** Any webpage an agent reads (via `web_browser_snapshot`, `web_browser_take_screenshot`, `web_browser_console_messages`, etc.) is **untrusted input** that can contain prompt injections attempting to hijack the agent into clicking, submitting forms, or navigating to attacker-controlled URLs. That is why all state-changing and navigation tools remain `external`-tier (approval-gated) by default — **do not** add blanket `allow` overrides on `web_*` tools.

---

## Install (macOS)

Prerequisites: `uv` (`brew install uv`) and Node.js `>= 20` (Homebrew or `nvm`).

1. **Register the `web` device on your ZOS host:**
   - Open `https://<your-zos-host>/admin/modules` → **Runs on your machine** → add slug **`web`** → copy the one-time token (`zos_a_...`).
   - *(Or on the ZOS host: `zos device add web`.)*
2. **Run the installer on your Mac:**
   ```bash
   git clone --depth 1 --branch zos-v3.2.8 https://github.com/zamborg/lowercaseLabs.git
   cd lowercaseLabs/zos
   ./deploy/launchd/install-web.sh
   ```
   *(Pass `--node /path/to/node` if you want a specific Node.js binary, or `--headless` to start in headless mode.)*
3. **Configure `~/.zos-web/config.toml` (`chmod 600`):**
   ```toml
   url = "wss://<your-zos-host>/actuator"
   token = "zos_a_YOUR_WEB_DEVICE_TOKEN"
   ```
4. **Load the LaunchAgent:**
   ```bash
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.web.plist || true
   ```
5. **Verify:** Refresh `https://<your-zos-host>/admin/modules` — `web` should appear **`online`** with **`25`** tools (`7` `read`, `18` `external`).

---

## Headed mode & manual logins

By default (`"headless": false` in `~/.zos-web/playwright-mcp.json`), the Chromium window launched by the `com.zos.web` LaunchAgent **appears on your logged-in macOS desktop** (in your GUI session) whenever an agent opens the browser.

- Log into sites manually inside that dedicated window (passwords, 2FA, passkeys) — sessions persist in `~/.zos-web/profile`.
- To switch to headless mode later, edit `~/.zos-web/playwright-mcp.json` and set `"headless": true`, then restart the agent:
  ```bash
  launchctl kickstart -k gui/$(id -u)/com.zos.web
  ```

---

## Resetting / wiping the browser profile

To log out of every site and erase all cookies, localStorage, and cache for the ZOS browser:

```bash
launchctl bootout gui/$(id -u)/com.zos.web || true
rm -rf ~/.zos-web/profile
mkdir -m 700 ~/.zos-web/profile
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.web.plist
```

---

## Tool tiers (ZOS defaults for `@playwright/mcp@0.0.86`)

ZOS derives device tool tiers directly from upstream MCP `ToolAnnotations` (`readOnlyHint: true` → `read`; everything else → `external`, requiring owner approval). No overrides are seeded.

### `read` tier (7 tools — allowed by default)

- `web_browser_console_messages`
- `web_browser_find`
- `web_browser_network_request`
- `web_browser_network_requests`
- `web_browser_snapshot`
- `web_browser_take_screenshot`
- `web_browser_wait_for`

### `external` tier (18 tools — approval required by default)

- `web_browser_click`
- `web_browser_close`
- `web_browser_drag`
- `web_browser_drop`
- `web_browser_emulate_media`
- `web_browser_evaluate`
- `web_browser_file_upload` *(also jailed to `~/.zos-web/output`)*
- `web_browser_fill_form`
- `web_browser_handle_dialog`
- `web_browser_hover`
- `web_browser_navigate`
- `web_browser_navigate_back`
- `web_browser_press_key`
- `web_browser_resize`
- `web_browser_run_code_unsafe`
- `web_browser_select_option`
- `web_browser_tabs`
- `web_browser_type`

---

## Running tests

```bash
cd zos/devices/web
uv sync --frozen
uv run ruff check
uv run pytest
```
