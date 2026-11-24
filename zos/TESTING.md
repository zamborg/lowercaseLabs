# ZOS V0 — Testing & Morning Setup Guide

ZOS V0 is **running live in OrbStack** right now at `http://localhost:8787` with the macOS actuator connected.

## What's running right now

- **ZOS + Postgres 17 (OrbStack Docker Compose):** `deploy/docker-compose.yml`
  - Bound to `127.0.0.1:8787` only (`restart: unless-stopped`, persistent volume `deploy_zos_pgdata`).
  - Principal `zubin`, space `main`, hosted `memory` provider, and all 14 V0 `tool_tiers` are initialized.
- **Mac actuator (LaunchAgents):**
  - `com.zos.tmux` (`tmux -L zos` isolated from actuator permissions)
  - `com.zos.actuator` (`~/.zos-actuator/venv`, dialed into `ws://127.0.0.1:8787/actuator`)
- **Quick health check:**
  ```bash
  curl -s http://127.0.0.1:8787/healthz | python3 -m json.tool
  ```
  Expect `memory` (`online: true`), `mac` (`online: true`), and `gcal`/`gmail` (`online: false` until you connect Google below).

## Local secrets (gitignored, mode `0600`)

- `zos/.env` — Postgres password, `ZOS_SECRET_KEY`, and single-quoted `ZOS_OWNER_PASSWORD_HASH`.
- `zos/.owner-passphrase` — generated owner passphrase for `/oauth/authorize` and `/connect/google`. Read with `cat zos/.owner-passphrase` (you can delete it once stored in your password manager, or rotate by putting a new single-quoted hash from `uv run zos hash-password` into `zos/.env`).
- `~/.zos-actuator/config.toml` — actuator WebSocket URL and token.

---

## 1. Connect Claude Code and Codex

### Option A — OAuth 2.1 (browser login)

Point both clients at **`http://localhost:8787/mcp`**:

- **Claude Code:**
  ```bash
  claude mcp add --transport http zos http://localhost:8787/mcp
  ```
  When prompted to sign in, your browser opens `http://localhost:8787/oauth/authorize` — enter the passphrase from `zos/.owner-passphrase` and click **Approve**.
- **Codex:**
  ```bash
  codex mcp add zos --url http://localhost:8787/mcp
  ```
  (Or configure `[mcp_servers.zos]` with `url = "http://localhost:8787/mcp"` in `~/.codex/config.toml` and complete the OAuth login.)

### Option B — Bearer token fallback

If either client prefers a static bearer header over browser OAuth:

```bash
cd /Users/zubinaysola/Documents/personal/lowercaseLabs/zos
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos client add claude-code --bearer
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos client add codex --bearer
```

Pass `Authorization: Bearer <token>` to `http://localhost:8787/mcp`.

### Smoke tests to try

1. **Shared memory (marquee problem):** Ask Claude Code to store a memory (`memory_put`), then ask Codex to retrieve or search it (`memory_get` / `memory_search`).
2. **Status & Mac shell:** Call `zos_status`, then `mac_shell_run` (external tier — clients with MCP elicitation prompt for approval; clients without elicitation receive the exact `zos policy set` override command).
3. **Granting an override (if needed):**
   ```bash
   docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
     uv run --no-sync zos policy set --client <client-name-or-id> mac_shell_run allow
   ```
4. **Actuator live visibility:** Stop and restart the actuator while connected — `mac_*` tools vanish and reappear automatically via `notifications/tools/list_changed`:
   ```bash
   launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.zos.actuator.plist
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.actuator.plist
   ```

---

## 2. Pending morning steps (require GUI / browser)

### A. Grant Full Disk Access for `mac_messages_search`

1. Open **System Settings → Privacy & Security → Full Disk Access**.
2. Click **`+`**, press **`Cmd+Shift+G`**, and select the exact resolved Python binary:
   ```text
   /Users/zubinaysola/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/bin/python3.12
   ```
3. Restart the actuator so it picks up the permission:
   ```bash
   launchctl kickstart -k gui/$(id -u)/com.zos.actuator
   ```
4. Test `mac_messages_search` from any connected MCP client. *(Re-grant Full Disk Access if you upgrade Python.)*

### B. Connect Google Calendar + Gmail

1. In [Google Cloud Console](https://console.cloud.google.com/), create (or pick) a project (e.g. `zos-personal`) and enable **Google Calendar API** and **Gmail API**.
2. Under **Google Auth Platform → Audience**: set User Type **External**, then click **Publish App** (**In production**; do *not* submit for verification).
   - *Fallback if restricted Gmail scopes are blocked on an unverified personal account:* switch to **Testing** mode, add your personal Gmail as a Test User, and note the 7-day refresh token expiry.
3. Under **Clients → Create Client → Web application**, add Authorized Redirect URI:
   ```text
   http://localhost:8787/connect/google/callback
   ```
   *(Optional: also add `http://127.0.0.1:8765/callback` if you want to run `uv run python spikes/google/oauth_spike.py`.)*
4. Add `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` to `zos/.env`, then recreate the container:
   ```bash
   cd /Users/zubinaysola/Documents/personal/lowercaseLabs/zos
   docker compose --env-file .env -f deploy/docker-compose.yml up -d --force-recreate zos
   ```
5. Open **`http://localhost:8787/connect/google`** in your browser, enter the passphrase from `zos/.owner-passphrase`, and complete Google consent. Both `gcal_*` and `gmail_*` tools appear immediately via `notifications/tools/list_changed`.

### C. Later / optional

- **Cloud deploy (Fly + Neon):** `deploy/fly.toml` and `deploy/Dockerfile` are ready whenever you want to deploy off localhost.

---

## Running the automated test suites

From `zos/`:

```bash
uv run pytest
uv run ruff check
(cd actuator && uv run pytest && uv run ruff check)
```

*(Integration/E2E tests target the local `zos-pg-test` Postgres 17 container on `127.0.0.1:55432`, completely isolated from the live `deploy_zos_pgdata` volume.)*
