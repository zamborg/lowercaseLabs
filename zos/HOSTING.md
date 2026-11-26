# Hosting ZOS

Self-contained deployment guide for running ZOS on a single Linux server (Ubuntu 24.04, ≥2 GB RAM, ≥20 GB disk). Written so a coding agent with SSH access can run the setup end-to-end while delegating only the minimal **👤 Human** steps below.

---

## Architecture at a Glance

- **Core (`zos`):** Control plane only — MCP endpoint (`/mcp`), OAuth 2.1 + DCR, policy engine, async approvals, audit log, AEAD vault, module lifecycle, and `/admin` UI + reverse proxy.
- **Database (`postgres:17`):** Runs inside Docker Compose under a restricted non-superuser app role (`zos_app`, `CREATEROLE`, `NOSUPERUSER`). Each stateful module receives its own isolated `mod_<slug>` role and schema.
- **Default modules:** `memory` (Postgres FTS + `pg_trgm` shared memory store) and `inbox` (async agent-to-agent and owner mailboxes), both running on the internal `zos_modules` network with zero internet egress.
- **Edge (`--edge`):** Every public edge enforces the same single-source path allowlist (`deploy/caddy/Caddyfile.allowlist`: `/mcp`, `/mcp/*`, `/.well-known/*`, `/oauth/*`, `/admin`, `/admin/*`, `/connect/*`, and `/actuator` WebSocket) and returns `404` for everything else (including `/healthz`):
  - `exe` *(recommended no-domain path)*: plain-HTTP `edge-filter` (`caddy:2.11.9-alpine` on `:8000`) behind [exe.dev](https://exe.dev/)'s managed HTTPS proxy (`https://<vm>.exe.xyz`). No domain registration or TLS setup required.
  - `caddy` *(bring-your-own-domain path)*: automatic Let's Encrypt HTTPS for `--domain` on ports `80`/`443`.
  - `cloudflared`: outbound Cloudflare Tunnel (no open inbound ports) routed through `edge-filter` on an isolated `zos_edge` Docker network with no published host ports.
  - `none`: binds `127.0.0.1:8787` only (for local machines, SSH tunnels, or rehearsals).

---

## Step 1 — Provision a Host

Choose **Route A** (fastest, no domain needed) or **Route B** (bring your own domain).

### Route A — exe.dev *(Recommended — No Domain Required)*

1. **👤 Sign up at [exe.dev](https://exe.dev/) (Personal plan) and add your SSH public key.**
2. **Create your VM** (run from your local machine — an agent with SSH access to `exe.dev` can also run this):
   ```bash
   ssh exe.dev new --name=my-zos
   ```
   > **Note:** **Your VM name becomes your public URL** (`--name=my-zos` → `https://my-zos.exe.xyz`). exe.dev VM names must be **5–52 characters**: lowercase letters and digits with single hyphens, starting with a letter (so `zos` alone is too short). Pick something like `my-zos` or `zos-home`. You can also attach a custom domain later with `ssh exe.dev domain add`.
3. exe.dev VMs boot Ubuntu 24.04 with Docker, Compose, systemd, and passwordless `sudo` already installed (`ssh my-zos.exe.xyz` connects as `exedev`). Hand `ssh my-zos.exe.xyz` to your agent (or run Steps 2–4 yourself).

### Route B — Bring Your Own Domain (Hetzner / Any Ubuntu 24.04 VPS + Caddy)

> **👤 Human step:** An agent cannot open cloud accounts or modify your registrar.

1. **Rent an Ubuntu 24.04 VPS (≥2 GB RAM, 2 vCPU recommended).**
   - **Worked example (Hetzner Cloud):** Create a project → **Add Server** → Location of choice → Image **Ubuntu 24.04** → Type **CX22** (x86_64, 4 GB RAM) or **CAX11** (ARM64, 4 GB RAM) → add your SSH public key → **Create & Buy Now**.
   - Any other Ubuntu 24.04 provider (DigitalOcean, Vultr, Linode, AWS Lightsail) works identically.
   - Ensure inbound TCP ports **22** (SSH), **80**, and **443** are allowed in the provider firewall (only port 22 is needed if using `--edge cloudflared`).
2. **Add one DNS `A` record** at your DNS provider:
   - Type: `A`
   - Name: `zos` (for `zos.<your-domain>`)
   - Value: `<VPS_IPV4_ADDRESS>`
   - Wait until `dig +short zos.<your-domain>` returns `<VPS_IPV4_ADDRESS>`.
3. Hand the SSH connection (`root@<VPS_IPV4_ADDRESS>`) and domain name (`zos.<your-domain>`) to your agent (or run Steps 2–4 yourself).

---

## Step 2 — Run the Installer (Agent or Human)

SSH into the server and run `deploy/install.sh` pinned to the release tag (`zos-v3.1.2`). Choose the command matching your edge mode (all variants install into `/opt/zos` by default):

- **`--edge exe` *(recommended no-domain path — run on the exe.dev VM from any directory)*:**
  ```bash
  curl -fsSL https://raw.githubusercontent.com/zamborg/lowercaseLabs/zos-v3.1.2/zos/deploy/install.sh -o /tmp/zos-install.sh
  sudo bash /tmp/zos-install.sh \
    --edge exe \
    --ref zos-v3.1.2
  ```
  *(Sets `ZOS_BASE_URL=https://$(hostname).exe.xyz` automatically. If you configured a custom domain via `ssh exe.dev domain add`, pass `--domain zos.<your-domain>`.)*
- **`--edge caddy` *(bring-your-own-domain VPS — automatic Let's Encrypt HTTPS; run from any directory)*:**
  ```bash
  curl -fsSL https://raw.githubusercontent.com/zamborg/lowercaseLabs/zos-v3.1.2/zos/deploy/install.sh -o /tmp/zos-install.sh
  sudo bash /tmp/zos-install.sh \
    --domain zos.<your-domain> \
    --edge caddy \
    --ref zos-v3.1.2
  ```
- **`--edge cloudflared` *(outbound Cloudflare Tunnel; run from any directory)*:**
  ```bash
  curl -fsSL https://raw.githubusercontent.com/zamborg/lowercaseLabs/zos-v3.1.2/zos/deploy/install.sh -o /tmp/zos-install.sh
  sudo bash /tmp/zos-install.sh \
    --domain zos.<your-domain> \
    --edge cloudflared \
    --ref zos-v3.1.2
  ```
- **`--edge none` *(local `127.0.0.1:8787` only — rehearsals or SSH-tunneled access)*:**
  From GitHub (run from any directory):
  ```bash
  curl -fsSL https://raw.githubusercontent.com/zamborg/lowercaseLabs/zos-v3.1.2/zos/deploy/install.sh -o /tmp/zos-install.sh
  sudo bash /tmp/zos-install.sh \
    --edge none \
    --ref zos-v3.1.2
  ```
  Or from a local checkout on the host (run from the `zos/` directory of the repo):
  ```bash
  cd /path/to/lowercaseLabs/zos
  sudo bash deploy/install.sh --edge none --local-source .
  ```

*(To track the latest development branch instead of a release tag, use `main` in the URL and `--ref main`. First-run install choices — `ZOS_INSTALL_REF`, `ZOS_INSTALL_SOURCE`, `ZOS_INSTALL_EDGE`, and `ZOS_INSTALL_DOMAIN` — are saved in `/opt/zos/zos/.env` and reused automatically on re-runs unless overridden.)*

### What `install.sh` does (idempotent)

1. Ensures Docker Engine and the Docker Compose plugin are installed on Ubuntu 24.04 (reuses preinstalled Docker on exe.dev).
2. Clones `https://github.com/zamborg/lowercaseLabs.git` into `/opt/zos` at `--ref` (or copies from `--local-source <path>` for offline/rehearsal installs).
   - **Prebuilt digest-pinned images (optional):** Set `ZOS_IMAGE_CORE` (e.g. `ghcr.io/zamborg/zos@sha256:<64-hex>`) and optionally `ZOS_IMAGE_MEMORY`, `ZOS_IMAGE_INBOX`, `ZOS_IMAGE_GMAIL`, `ZOS_IMAGE_GCAL` (`ghcr.io/zamborg/zos-<slug>@sha256:<64-hex>`) before running `install.sh` to pull published GHCR images by digest instead of building from source.
3. Generates strong random secrets (`POSTGRES_PASSWORD`, `ZOS_DB_PASSWORD`, `ZOS_SECRET_KEY`, `ZOS_BACKUP_KEY`) into `/opt/zos/zos/.env` (`chmod 600`) without ever printing them.
4. Starts Postgres 17, provisions the restricted non-superuser `zos_app` role (`CREATEROLE`, `NOSUPERUSER`) and shared `extensions` schema (`pg_trgm`), starts ZOS + your chosen edge (`edge-filter`, Caddy, or `cloudflared`), runs `zos init`, and installs the `memory` and `inbox` modules.
5. Enables the nightly encrypted backup systemd timer (`zos-backup.timer` → `/var/backups/zos`).

> **Agent note:** When running `install.sh` on behalf of a human, pass `--skip-passphrase` during the automated install and then hand Step 3 directly to the human so the passphrase never passes through agent chat or logs.

---

## Step 3 — 👤 Set the Owner Passphrase (and Publish on exe.dev)

> **Human step:** Never paste your owner passphrase into an AI chat.

1. **Set the owner passphrase:**
   - If you ran `install.sh` interactively, it prompts for the owner passphrase directly (**asked twice, input hidden**, hashed with Argon2id into `/opt/zos/zos/.env`).
   - If an agent ran `install.sh --skip-passphrase`, the installer prints `Owner passphrase NOT set — run Step 3`. SSH into the server yourself and run:
     ```bash
     sudo bash /opt/zos/zos/deploy/install.sh
     ```
     Because `install.sh` is idempotent and reuses your persisted edge, domain, and ref/source from `/opt/zos/zos/.env`, it skips already-completed steps, prompts only for the missing owner passphrase (**asked twice, input hidden**), hashes it into `.env`, and recreates `zos`.
2. **exe.dev only — make the HTTPS proxy public** (run from your **local machine**, not on the VM; replace `my-zos` if you chose a different VM name):
   ```bash
   ssh exe.dev share set-public my-zos
   ```
   *(exe.dev VMs are private by default; MCP clients and OAuth callbacks need the public proxy enabled, while `edge-filter` blocks `/healthz` and unknown paths with `404`.)*

---

## Step 4 — Verify the Deployment (Agent or Human)

Run from the server:

1. **Local health & default modules** (`/healthz` is intentionally local-only; public `/healthz` returns `404`):
   ```bash
   curl -sf http://127.0.0.1:8787/healthz | python3 -m json.tool
   ```
   Confirm `"ok": true` and both `memory` (5 tools) and `inbox` (5 tools) report `"online": true`.
   - **When using `--edge exe`:** exe.dev VMs cannot hairpin to `https://<vm>.exe.xyz`, so check the path filter on `127.0.0.1:8000` from the VM (and check `https://my-zos.exe.xyz/healthz` from your local machine after `share set-public`):
     ```bash
     curl -s -o /dev/null -w "healthz=%{http_code}\n" http://127.0.0.1:8000/healthz
     curl -s -o /dev/null -w "nope=%{http_code}\n" http://127.0.0.1:8000/nope
     curl -s -o /dev/null -w "well_known=%{http_code}\n" http://127.0.0.1:8000/.well-known/oauth-protected-resource
     curl -s -o /dev/null -w "mcp=%{http_code}\n" http://127.0.0.1:8000/mcp
     ```
     Confirm `healthz=404`, `nope=404`, `well_known=200`, and `mcp=401`.
   - **When using `--edge caddy`:** **wait a few seconds for Caddy to finish issuing the Let's Encrypt TLS certificate** (you can check `cd /opt/zos/zos && docker compose --env-file .env -f deploy/docker-compose.yml -f deploy/docker-compose.caddy.yml logs --tail 20 caddy`), then confirm public `/healthz` is blocked (`404`):
     ```bash
     curl -s -o /dev/null -w "%{http_code}\n" https://zos.<your-domain>/healthz
     ```
2. **Non-superuser Postgres check:**
   ```bash
   cd /opt/zos/zos
   docker compose --env-file .env -f deploy/docker-compose.yml exec -T zos \
     uv run --no-sync python -c "import os, psycopg; print(psycopg.connect(os.environ['DATABASE_URL']).execute('SELECT current_user, rolsuper FROM pg_roles WHERE rolname = current_user').fetchone())"
   ```
   Confirm `('zos_app', False)`.
3. **Admin UI:**
   - With `--edge exe`, open `https://my-zos.exe.xyz/admin` (replace `my-zos` with your VM name).
   - With `--edge caddy` or `--edge cloudflared`, open `https://zos.<your-domain>/admin`.
   - With `--edge none`, open an SSH tunnel from your local machine first:
     ```bash
     ssh -L 8787:127.0.0.1:8787 root@<VPS_IPV4_ADDRESS>
     ```
     Then open `http://localhost:8787/admin`.
   - Sign in with your owner passphrase and verify all admin pages load cleanly:
     - `/admin` (Overview)
     - `/admin/approvals` (Approvals)
     - `/admin/modules` (Modules)
     - `/admin/upstreams` (MCP Servers)
     - `/admin/clients` (Clients & OAuth Redirect Allowlist)
     - `/admin/policy` (Tools & Policy)
     - `/admin/audit` (Activity)
     - `/admin/m/memory/` (Memory module UI)
     - `/admin/m/inbox/` (Inbox module UI)
4. **Nightly backup timer:**
   ```bash
   systemctl status zos-backup.timer
   sudo /opt/zos/zos/deploy/backup/backup.sh
   ```
   Confirm an encrypted `.dump.enc` file is created in `/var/backups/zos`.

---

## Step 5 — 👤 Connect Your Agents

> **Human step:** Point your MCP clients at `https://my-zos.exe.xyz/mcp` (or `https://zos.<your-domain>/mcp` if using a custom domain).

- **Claude Code:**
  ```bash
  claude mcp add --transport http zos https://my-zos.exe.xyz/mcp
  ```
- **Codex:**
  ```bash
  codex mcp add zos --url https://my-zos.exe.xyz/mcp
  ```
- **ChatGPT / Claude web & mobile:** Add a custom MCP server pointing at `https://my-zos.exe.xyz/mcp` and complete OAuth sign-in in your browser.
  - `chatgpt.com`, `claude.ai`, and `agent.meta.ai` are seeded on the OAuth redirect allowlist by default (loopback hosts are always allowed).
  - If a new external client is rejected during OAuth registration, open **`https://my-zos.exe.xyz/admin/clients`** — you will see `"<client> tried to connect from <host>"` with **Allow** and **Dismiss** buttons (both audited). Click **Allow** and retry connecting the client.
- **Headless / bearer scripts:** Create a bearer client on `/admin/clients` or from `/opt/zos/zos` on the server (token is printed once — **don't paste the token into chat**):
  ```bash
  cd /opt/zos/zos
  docker compose --env-file .env -f deploy/docker-compose.yml exec -T zos \
    uv run --no-sync zos client add <name> --bearer
  ```
  Pass the returned token as `Authorization: Bearer <token>`.

---

## Off-Box Nightly Backups (`rclone` Example)

`zos-backup.timer` runs `/opt/zos/zos/deploy/backup/backup.sh` nightly at `03:15 UTC` (±15m jitter), creating AES-256-CBC (PBKDF2) encrypted `pg_dump` archives in `/var/backups/zos` (retaining 14 backups locally).

To copy every encrypted backup off-box (S3, Cloudflare R2, Backblaze B2, Google Drive, etc.) without adding cloud SDKs to ZOS, configure `rclone` once and set `ZOS_BACKUP_POST_HOOK`:

1. Install and configure `rclone` as root:
   ```bash
   apt-get install -y rclone
   rclone config   # create a remote named e.g. 'offbox'
   ```
2. Add the hook to `/opt/zos/zos/.env`:
   ```env
   ZOS_BACKUP_POST_HOOK='rclone copy "$ZOS_BACKUP_FILE" offbox:zos-backups/'
   ```
3. Test immediately:
   ```bash
   sudo /opt/zos/zos/deploy/backup/backup.sh
   ```
4. **Store `ZOS_BACKUP_KEY` and `ZOS_SECRET_KEY` in your password manager.** Without `ZOS_BACKUP_KEY` the backup cannot be decrypted; without `ZOS_SECRET_KEY` encrypted vault credentials and module secrets cannot be decrypted after restore.

---

## Updating ZOS

`install.sh --update` takes an encrypted backup first, reuses your saved edge and domain from `/opt/zos/zos/.env`, fetches the requested ref (or pulls updated digest images), runs database migrations, upgrades installed modules (preserving all schemas, secrets, and manual tier/policy overrides), and recreates containers:

```bash
sudo bash /opt/zos/zos/deploy/install.sh --update --ref zos-v3.1.2
```
*(Replace `zos-v3.1.2` with a newer release tag when upgrading, omit `--ref` to refresh your persisted ref or `--local-source`, or pass `--ref main` to track the development branch.)*

---

## Moving an Existing ZOS to This Server

Follow the full migration and restore runbook in [TESTING.md § 11 (Moving ZOS to a New Host)](TESTING.md#11-moving-zos-to-a-new-host).

> **Note for pre-M12 Postgres volumes:** If you are upgrading an existing pre-M12 Postgres volume in place where `pg_trgm` was originally installed in schema `public`, relocating `pg_trgm` into the shared `extensions` schema requires Postgres superuser privileges once. On bundled Docker Compose Postgres, `deploy/postgres/migrate-existing-to-app-role.sh` handles this automatically (and `install.sh` runs it for you). On managed hosted Postgres without superuser, ensure `pg_trgm` is installed in schema `extensions` before installing the `memory` module.

---

## Edge Alternatives (`--edge cloudflared` / `--edge none`)

- **Cloudflare Named Tunnel (`--edge cloudflared`):**
  1. **Containerized token mode (recommended for Cloudflare):** Create a remotely-managed Named Tunnel in Cloudflare Zero Trust, set its Public Hostname (`zos.<your-domain>`) service URL to **`http://edge-filter:8000`**, put `CLOUDFLARED_TOKEN=<token>` in `/opt/zos/zos/.env`, and run:
     ```bash
     sudo bash /opt/zos/zos/deploy/install.sh --domain zos.<your-domain> --edge cloudflared --ref zos-v3.1.2
     ```
     `edge-filter` publishes no host ports and `cloudflared` is network-isolated on `zos_edge` so traffic can only reach `edge-filter:8000` (enforcing the allowlist, blocking `/healthz`, and allowing `/actuator` WebSockets for a remote Mac actuator).
  2. **Host-side locally-managed mode:** Leave `CLOUDFLARED_TOKEN` unset and configure host `~/.cloudflared/config.yml` from `deploy/cloudflared/config.yml.example` (proxies allowlisted paths including `/actuator` to `http://127.0.0.1:8787`, catch-all `404`).
  3. Optionally add Cloudflare Access on browser-only paths (`/admin*`, `/oauth/authorize`, `/connect/*`) while leaving `/mcp`, `/.well-known/*`, `/oauth/register`, `/oauth/token`, and `/actuator` bypassing Access.
- **Local / rehearsal (`--edge none`):**
  ```bash
  cd /path/to/lowercaseLabs/zos
  sudo bash deploy/install.sh --edge none --local-source .
  ```
  Binds only `127.0.0.1:8787` with no edge proxy.

---

## Appendix — Optional Google Modules (`gmail` & `gcal`)

Gmail and Google Calendar run as two isolated egress modules (`gmail` and `gcal`) built from the shared `modules/google` image. Each module performs its own Google OAuth flow with narrower scopes and stores its refresh token encrypted inside its own Postgres schema (`mod_gmail` / `mod_gcal`).

1. **👤 Create a Google Cloud OAuth Web Client:**
   - Open [Google Cloud Console](https://console.cloud.google.com/), create or select a project, and enable **Gmail API** and **Google Calendar API** (plus **Google Calendar MCP** if you plan to use the `mcp` backend).
   - Under **Google Auth Platform → Audience**, choose **External** and publish (or add yourself as a Test User).
   - Under **Clients → Create Client → Web application**, add **both** Authorized Redirect URIs (using `my-zos.exe.xyz` or `zos.<your-domain>`):
     - `https://my-zos.exe.xyz/admin/m/gmail/callback`
     - `https://my-zos.exe.xyz/admin/m/gcal/callback`
   - Copy the **Client ID** and **Client Secret**.
2. **Install the modules:**
   - Open **`https://my-zos.exe.xyz/admin/modules`**, click **Install** on **Gmail** and **Google Calendar**, and enter `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` for each (stored encrypted in the ZOS vault).
   - Or install via CLI:
     ```bash
     cd /opt/zos/zos
     docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
       uv run --no-sync zos module install gmail \
       --config GOOGLE_CLIENT_ID=<id> --config GOOGLE_CLIENT_SECRET=<secret>
     docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
       uv run --no-sync zos module install gcal \
       --config GOOGLE_CLIENT_ID=<id> --config GOOGLE_CLIENT_SECRET=<secret>
     docker compose --env-file .env -f deploy/docker-compose.yml restart zos
     ```
3. **👤 Connect in your browser:**
   - Open `https://my-zos.exe.xyz/admin/m/gmail/` and click **Connect Gmail**.
   - Open `https://my-zos.exe.xyz/admin/m/gcal/` and click **Connect Google Calendar** (you can also switch between `api` REST v3 and `mcp` Calendar MCP backends on this page).
   - Verify `/healthz` reports `gmail` (3 tools) and `gcal` (9 tools on `api`) online.
