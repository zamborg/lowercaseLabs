# ZOS Mac Actuator

A lightweight standalone macOS daemon (`zos-actuator`) that dials out to your local ZOS server over WebSocket (`ws://127.0.0.1:8787/actuator`) and exposes local macOS capabilities as MCP tools.

Actuator tools appear in your MCP clients only while the daemon is connected. If your Mac sleeps or the daemon stops, ZOS hides `mac_*` tools automatically and broadcasts `notifications/tools/list_changed` when it reconnects.

## Tools exposed

| Tool | Tier | Description |
|---|---|---|
| `mac_messages_search` | `read` | Read-only full-text search (SQLite FTS5 mirror at `~/.zos-actuator/messages.db`) over `~/Library/Messages/chat.db`, decoding Apple `typedstream` rich message bodies and resolving contact names from local macOS AddressBook databases. |
| `mac_shell_run` | `external` (requires approval) | Runs shell commands in a persistent, isolated local `tmux` session (`tmux -L zos`) with configurable timeout and optional Ctrl-C reset. |

Because `mac_shell_run` runs in a separate `tmux` LaunchAgent (`com.zos.tmux`), commands executed via `mac_shell_run` do **not** inherit the actuator's Full Disk Access grant.

## Install (dedicated Python interpreter)

`deploy/launchd/install.sh` installs the actuator with its own isolated copy of Python 3.12 instead of linking against `uv`'s shared interpreter:

- `~/.zos-actuator/python/` — a full copy of `uv`'s CPython 3.12 installation directory (copied intact because `bin/python3.12` links `@executable_path/../lib/libpython3.12.dylib`).
- `~/.zos-actuator/venv/` — virtual environment created with `uv venv --python ~/.zos-actuator/python/bin/python3.12`.
- `~/.zos-actuator/config.toml` — WebSocket URL and actuator bearer token (`chmod 600`).
- `~/Library/LaunchAgents/com.zos.tmux.plist` and `com.zos.actuator.plist`.

### Steps

1. **Mint an actuator token on ZOS:**
   ```bash
   docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
     uv run --no-sync zos actuator add mac
   ```

2. **Run the installer** (from `zos/`):
   ```bash
   ./deploy/launchd/install.sh
   ```
   Re-running `./deploy/launchd/install.sh` later upgrades `zos-actuator` code without touching `~/.zos-actuator/python/` (so your macOS Full Disk Access grant stays intact). Pass `--upgrade-python` only if you want to refresh the copied Python binary itself.

3. **Paste your token into `~/.zos-actuator/config.toml`:**
   ```toml
   url = "ws://127.0.0.1:8787/actuator"
   token = "<token-from-step-1>"
   ```
   Keep the URL on `127.0.0.1` — `/actuator` is intentionally blocked on public Cloudflare Tunnels.

4. **Grant Full Disk Access (for `mac_messages_search` and contact names):**
   1. Open **System Settings → Privacy & Security → Full Disk Access**.
   2. Click **`+`**, press **`Cmd+Shift+G`**, and add:
      ```text
      ~/.zos-actuator/python/bin/python3.12
      ```
   Why the dedicated copy? macOS grants Full Disk Access per resolved executable binary. Granting it to `uv`'s shared Python binary would give Full Disk Access to every unrelated Python project on your machine; isolating it to `~/.zos-actuator/python/bin/python3.12` scopes the permission strictly to `zos-actuator`.

5. **Start the LaunchAgents:**
   ```bash
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.tmux.plist || true
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.actuator.plist || true
   ```

6. **Verify connection:**
   ```bash
   curl -s http://127.0.0.1:8787/healthz | python3 -m json.tool
   ```
   Expect `{"namespace": "mac", "kind": "actuator", "online": true}`. Logs live in `~/Library/Logs/zos-actuator.{out,err}.log`.

## Running actuator tests

From `zos/actuator/`:

```bash
uv run ruff check
uv run pytest
```
