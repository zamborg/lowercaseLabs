# ZOS — V0 build plan (executor handbook)

Status: final (review 06, 2026-11-23). Written for an executor agent to
work through task by task. Read `PLAN.md` and `SCHEMA.md` first. They are
the spec. This file is the order of work.

## Ground rules

1. **Stay inside the boundary.** ZOS is everything behind the MCP interface.
   Do not build agent runtimes, channels, schedulers, prompts/resources,
   or anything in `later/`. If a task seems to need one, stop and report.
2. **Build to the MCP spec**, using the official `mcp` Python SDK's
   **low-level server** (`mcp.server.lowlevel.Server`), not `FastMCP`
   decorators: tool lists are dynamic per client and per actuator state,
   and calls need our CallContext. SDK APIs move: check every API you use against the installed version (read its
   source or docs) before relying on it. Claude Code / Codex are manual
   smoke tests only, never design inputs.
3. **Every task ends with passing tests** (`uv run pytest`, `uv run ruff
   check`). Don't start the next task on a red suite. Each milestone ends
   with its scripted **Check**. Run it and paste the output into the milestone
   report.
4. **Record deviations.** Any choice not settled here goes in
   `zos/DECISIONS.md` (date, decision, why). Don't silently diverge from
   PLAN/SCHEMA. If you must, record it and flag it in the report.
5. **Privacy.** Never print, log, or commit content from Zubin's real data
   (Messages, Gmail, Calendar). Tests use fixtures. Live smoke tests assert on
   counts and shapes only. No secrets in git (`.env` is ignored).
6. **Git.** `zos/` is a subdirectory of the `lowercaseLabs` monorepo. Work on
   a branch per milestone (`zos/m1-core`, …), commit per task with clear
   messages. Don't push, open PRs, or merge unless asked.
7. **Human steps** (marked 👤) need Zubin: accounts, secrets, macOS
   permissions, Google console. When you reach one, stop and ask with exact
   instructions; continue with other tasks if possible.

## Stack

Python 3.12 · uv · `mcp` SDK (Streamable HTTP server + client) · Starlette /
FastAPI · psycopg 3 (async pool) · Typer (CLI) · `cryptography` (AES-GCM) ·
httpx (+ respx in tests) · pytest + pytest-asyncio · ruff. Local Postgres 17
for tests (Postgres.app or Docker); Neon dev branch only for deploy smoke.

## Layout

```
zos/
├── pyproject.toml   AGENTS.md (dev notes for coding agents)   DECISIONS.md
├── migrations/0001_init.sql
├── src/zos/
│   ├── config.py            # env: DATABASE_URL, ZOS_SECRET_KEY, ZOS_BASE_URL, ZOS_OWNER_PASSWORD_HASH
│   ├── db/                  # pool, migrator, queries
│   ├── auth/                # bearer + OAuth AS, CallContext
│   ├── router/              # registry, naming, routing, sessions, zos_status
│   ├── policy/              # tiers, decisions, annotations, elicitation
│   ├── audit.py
│   ├── providers/{base.py, memory.py, google/, actuator.py}
│   ├── vault.py
│   ├── web/                 # authorize + connect-google pages (minimal HTML)
│   ├── app.py               # ASGI app: /mcp, /actuator (WS), /oauth/*, /healthz
│   └── cli.py               # `zos` command
├── actuator/                # standalone `zos-actuator` package, own pyproject
│   └── src/zos_actuator/{main.py, messages.py, typedstream.py, shell.py}
├── deploy/{Dockerfile, fly.toml, launchd/*.plist}
└── tests/{unit, integration, e2e}
```

## Tool catalog (V0) and tier seeds

| Tool | Tier | Notes |
|---|---|---|
| `zos_status` | read | always on; lists providers + online state |
| `memory_get`, `memory_search` | read | |
| `memory_put`, `memory_delete` | write | |
| `mac_messages_search` | read | actuator only |
| `mac_contacts_search`, `mac_contacts_get`, `mac_messages_threads`, `mac_messages_read` | read | device (`zos-mac` 0.3.0, v3.2.5) |
| `mac_messages_send` | external | device (`zos-mac` 0.3.1, Shortcuts primary + AppleScript fallback, v3.2.7) |
| `mac_shell_run` | external | actuator only |
| `web_browser_snapshot`, `web_browser_take_screenshot`, `web_browser_find`, `web_browser_console_messages`, `web_browser_network_requests`, `web_browser_network_request`, `web_browser_wait_for` | read | device (`web`, `@playwright/mcp@0.0.86`, v3.2.6) |
| `web_browser_navigate`, `web_browser_click`, `web_browser_type`, `web_browser_fill_form`, `web_browser_file_upload`, `web_browser_evaluate`, … (18 tools) | external | device (`web`, `@playwright/mcp@0.0.86`, v3.2.6) |
| `gcal_list_events` | read | |
| `gcal_create_event`, `gcal_update_event`, `gcal_delete_event` | external | |
| `gmail_search`, `gmail_read` | read | |
| `gmail_send` | external | |

Annotations: tier sets `readOnlyHint` (read → true) and `openWorldHint`
(external → true); providers declare `destructiveHint` per tool (e.g.
`memory_delete`, `gcal_delete_event` → true).

---

## M0 — Scaffold and spikes

- **0.1 Scaffold** the layout above (empty modules), `pyproject.toml`,
  ruff/pytest config, `zos/AGENTS.md` (how to run tests, the ground rules),
  `DECISIONS.md`. Test: `uv run pytest` runs (zero tests OK).
- **0.2 Hosting spike** (`spikes/hosting/`, throwaway): minimal Streamable-HTTP
  MCP server with `ping` and a `toggle` tool that adds/removes a tool and
  emits `tools/list_changed`. Deploy to Fly (👤 Fly account/app). With an SDK
  client script, verify: connect, call, receive `list_changed`, then stay
  idle 120s and call again. Record keepalive/timeout settings needed behind
  Fly's proxy in DECISIONS.md.
- **0.3 Google spike** (`spikes/google/`, throwaway; 👤 Google Cloud project,
  OAuth client of type Web, consent screen External → **In production**,
  test with Zubin's personal account): loopback OAuth requesting
  `calendar.events`, `gmail.readonly`, `gmail.send`, `access_type=offline`,
  `prompt=consent`. Verify a refresh token is issued and still refreshes
  (re-check after 8 days, or record the policy as documented). Record in
  DECISIONS.md whether restricted Gmail scopes work for an unverified
  production app on a personal account. Fallback if blocked: Testing mode +
  documented weekly re-auth.

**Check:** DECISIONS.md has hosting and Google findings; spike scripts run.

## M1 — Core, policy, memory (solves the marquee problem)

- **1.1 DB.** Async pool; migrator (`schema_migrations`, applies
  `migrations/*.sql` in order, idempotent); `0001_init.sql` = SCHEMA.md
  exactly (minus OAuth tables if you prefer `0002`). Test: fresh DB migrates;
  second run is a no-op; key constraints hold (unique keys, CHECKs).
- **1.2 CLI.** `zos init` (principal, its single space, `memory` provider,
  tier seeds), `zos client add NAME --bearer` (prints token once, stores
  SHA-256), `zos token revoke`, `zos policy set [--client] PATTERN
  DECISION`, `zos migrate`. Test via Typer runner.
- **1.3 Auth.** Bearer middleware → immutable `CallContext(principal, space,
  client, trust_ceiling)`; revoked/expired → 401 with spec-compliant
  `WWW-Authenticate` including `resource_metadata="<ZOS_BASE_URL>/.well-known/oauth-protected-resource"`
  (OAuth discovery in M2 relies on it).
- **1.4 Router.** `ToolProvider` protocol (PLAN.md); registry; merged
  `tools/list` from online providers only; naming validation (`[a-zA-Z0-9_-]`,
  ≤48, deterministic truncate + 6-char hash); routing; unknown tool → error;
  `zos_status`. Track live MCP server sessions so the router can broadcast
  `tools/list_changed` (needed by M3). Unit-test naming and routing with fake providers.
- **1.5 Policy.** Tier resolution (longest literal match, unknown →
  external); decision resolution (client match → all-clients → tier
  default; `untrusted` turns `require_approval` into `deny`); annotations
  mapping. Exhaustive unit matrix.
- **1.6 Confirmation.** On `require_approval`: if the session declared the
  elicitation capability, elicit with the exact tool name and full canonical
  JSON args (never truncated); accept → execute; decline/cancel → denied.
  No capability → deny with a message naming the exact `zos policy set`
  override. Test with an SDK client that has an elicitation callback
  (accept and decline) and one without. `elicitation/create` takes a flat
  `requestedSchema` (e.g. `{approve: boolean}`); execute only when
  `action == "accept"` **and** `content["approve"] is True`.
- **1.7 Audit.** One `tool.called` event per call: client, provider, tool,
  decision, outcome, `args_digest` = SHA-256 of canonical JSON. No raw args.
- **1.8 Memory provider.** put (upsert by key), get, search (FTS rank,
  trigram fallback, tag filter, limit ≤ 50), delete.
- **1.9 App.** ASGI app mounting MCP at `/mcp`, `/healthz`; config from env.
- **1.10 E2E.** Start the app on a random port against local Postgres; two
  SDK clients with different bearer tokens: A `memory_put`, B `memory_get`
  and `memory_search` see it; tools carry annotations; a fake external tool
  triggers elicitation for A and is denied for a client without the capability;
  events rows exist for every call.
- **1.11 Deploy.** Dockerfile, `fly.toml` (M0 findings), migrations on
  release. 👤 Neon project + dev branch, Fly secrets. Smoke: `/healthz` and
  an SDK client `memory_put/get` against Fly.

**Check:** 1.10 e2e passes locally; Fly smoke passes. Optional manual
smoke: Claude Code and Codex configured with a bearer header share a memory.

## M2 — OAuth 2.1 for clients

- **2.1 Discovery.** `/.well-known/oauth-protected-resource` and
  `/.well-known/oauth-authorization-server` per the MCP authorization spec.
  Prefer the SDK's server-side auth support if the installed version has
  it. Otherwise implement minimally.
- **2.2 DCR** (`/oauth/register`) → `clients` row (kind `oauth`). Must accept
  public clients (`token_endpoint_auth_method="none"`).
- **2.3 Authorize** (`/oauth/authorize`): minimal owner page; owner
  authenticates with a passphrase (argon2 hash in `ZOS_OWNER_PASSWORD_HASH`),
  sees client name + redirect URI, approves. PKCE S256 required; codes
  single-use, 60s TTL (`oauth_codes`). Bind tokens to the resource
  (`resource` parameter / audience = ZOS_BASE_URL).
- **2.4 Token** (`/oauth/token`): code exchange; access tokens opaque 1h;
  refresh tokens 30d, rotated on use (`family_id`, `rotated_at`); reuse of
  a rotated token revokes the family.
- **2.5 Tests:** full flow with the SDK's OAuth client provider and a test
  hook that submits the authorize form; refresh rotation; revocation → 401.

**Check:** SDK client completes OAuth from discovery to tool call. Manual
smoke: Claude Code and Codex sign in via browser.

## M3 — Mac actuator

- **3.1 Core side** (`/actuator` WebSocket): bearer (kind `actuator`, bound
  to provider `mac`) checked on upgrade; ZOS runs an MCP **client** session
  over the socket → `initialize` → `tools/list` → provider online →
  broadcast `list_changed`. Ping every 20s. A new connection replaces the
  old one. On disconnect: offline + `list_changed`; in-flight calls fail
  `ACTUATOR_DISCONNECTED_MID_CALL`; later calls to its tools fail
  `ACTUATOR_OFFLINE`.
- **3.2 `zos-actuator` package:** MCP **server** over a WebSocket it dials;
  reconnect with jittered backoff; config `~/.zos-actuator/config.toml`
  (url, token). `zos actuator add mac` creates the `mac` provider row and mints
  its token (`tokens.provider_id`, kind `actuator`).
- **3.3 `mac_messages_search`:** local FTS5 index `~/.zos-actuator/messages.db`
  synced incrementally by `message.ROWID` from `chat.db`. Open chat.db as
  `file:…/chat.db?mode=ro` with `busy_timeout=2000` (never `immutable=1`).
  Text = `message.text`, else decode `attributedBody` (typedstream → NSString;
  write `typedstream.py` with fixture tests). Dates: Apple epoch (ns since
  2001-01-01) → ISO 8601. Returns handle (raw phone/email), is_from_me, date,
  chat identifier, text (capped). Params: `query`, `handle?`, `since?`,
  `limit ≤ 50`. Sync before each search (cheap: new ROWIDs only).
- **3.4 `mac_shell_run`:** dedicated tmux server `tmux -L zos`, started by
  its own launchd agent (so commands don't inherit the actuator's Full Disk
  Access). Per call: global mutex; if `#{pane_current_command}` isn't the
  shell → `PANE_BUSY` (param `reset=true` sends C-c first); run as
  write CMD to a per-call temp script, then send-keys only
  `source /tmp/zos/ID.sh >/tmp/zos/ID.out 2>&1; echo $? >/tmp/zos/ID.rc`
  (avoids quoting/multiline/`!` hazards, keeps cwd/env); if the pane died
  (e.g. the command ran `exit`), recreate it and report that; wait for RC
  with timeout (default 30s, max 300s); return exit code + output capped at
  32KB (head + tail). Params: `command`, `timeout_s?`, `reset?`.
- **3.5 Install.** `deploy/launchd/` plists for actuator + tmux, install
  script. Actuator runs from a venv at a fixed path
  (`~/.zos-actuator/venv`). Document granting Full Disk Access to that venv's
  resolved Python binary, and re-granting after Python upgrades. 👤 grant FDA.
- **3.6 Tests:** fixture chat.db built in-test (schema subset, incl. an
  `attributedBody` row); shell tests on a real tmux; e2e with core + a fake
  actuator: SDK client receives `list_changed`, tools appear/disappear,
  offline and mid-call errors are correct. Real-device smoke: search returns
  a count > 0 (print counts only).

**Check:** e2e passes; on the Mac, stopping the actuator removes `mac_*`
tools for a connected client and restarting restores them.

## M4 — Google (first-party upstream)

- **4.1 Vault:** AES-GCM, key from `ZOS_SECRET_KEY` (base64, 32 bytes),
  `key_version`; decrypt only in memory; per-credential `asyncio.Lock`
  around refresh. Tests: round-trip, wrong key fails, concurrent refresh
  calls hit Google once.
- **4.2 Connect:** `/connect/google` (owner-authenticated) → Google OAuth
  (scopes from M0) → callback stores one `credentials` row and the `gcal` and
  `gmail` providers that share it. Tools vanish (provider offline) if the credential is revoked.
- **4.3 Calendar:** `gcal_list_events(time_min, time_max, calendar_id =
  "primary", query?)`, `gcal_create_event`, `gcal_update_event`,
  `gcal_delete_event`. Explicit time zones on all times.
- **4.4 Gmail:** `gmail_search(query, limit ≤ 25)` → `messages.list` then
  `format=metadata` fetches with concurrency ≤ 5 (from, to, subject, date,
  snippet, ids); `gmail_read(message_id)` → text/plain preferred, else
  HTML stripped, capped; `gmail_send(to, subject, body,
  reply_to_message_id?)` builds MIME and, when replying, sets `threadId`,
  `In-Reply-To`, `References`.
- **4.5 Tests:** respx-mocked Google APIs incl. 429 backoff, token refresh,
  MIME threading. Live smoke: read-only, counts only.

**Check:** from an SDK client: list today's events, search Gmail (counts),
and `gmail_send` to Zubin's own address triggers elicitation before sending.

## Definition of done (V0)

All milestone checks pass; Claude Code and Codex both connect via OAuth and
see the same memory, Mac tools (while the actuator runs), and Google tools;
every call is in `events`; DECISIONS.md is current; nothing from `later/`
was built.
