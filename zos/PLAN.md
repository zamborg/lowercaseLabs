# ZOS — plan (V0)

Status: **design final** (reviews 05–06, 2026-11-23). Execution plan: `BUILD.md`.
Schema: `SCHEMA.md`. The earlier runtime-shaped design is parked in `later/`.

## What ZOS is

The "Zubin layer" for every agent he uses: a self-deployable **context and
routing engine for MCP calls**. Any MCP client (Claude Code, Codex, a chat
app, a local LLM UI) connects to one ZOS MCP endpoint. Behind it, ZOS hosts
memory and routes tool calls to upstream services and to actuators like his
Mac daemon, enforcing permissions and logging every call.

Agents run outside ZOS. ZOS never runs an agent.

**The boundary.** ZOS is everything behind the MCP interface and nothing in
front of it. Which harness, model, or UI the agent runs in, and how it's
configured (CLAUDE.md, AGENTS.md, skills, its own prompts), is the agent's
business. ZOS owes any MCP client exactly one thing: a correct,
spec-compliant MCP server. We design and test against the MCP spec, never
against one client's quirks.

Marquee problem: memories and MCPs diverged between Claude Code and Codex.
With ZOS both point at the same endpoint and see the same things.

## Schematic

```
   any MCP client (Claude Code · Codex · …)
                  │  OAuth 2.1 · Streamable-HTTP MCP
      ┌───────────┴────────────┐
      │       ZOS router       │  auth · policy · namespacing · audit
      └──┬─────────┬────────┬──┘
   hosted│ upstream│ actuator│
   memory   Google   Mac daemon (dials out over WSS, speaks MCP)
           (Calendar,
            Gmail; encrypted token vault)
                  │
            Postgres (Neon for Zubin; any Postgres for self-deploy)
```

## Components

- **MCP edge + auth.** Streamable-HTTP MCP. Clients sign in via MCP OAuth
  2.1 (PKCE + dynamic client registration) against ZOS's own single-user
  authorization server. Bearer tokens (`zos token create`) as fallback for
  headless clients and for actuators. Minimal web pages to approve client
  sign-ins and connect Google.
- **Router.** Merges tools from all online providers, namespaces them,
  filters by policy, routes calls, writes one audit event per call.
  Always-on `zos_status` tool lists which providers are live.
- **Providers — the one seam behind the router:**
  `ToolProvider { namespace, is_online(), list_tools(), call_tool(name, args, ctx) }`
  — `namespace` = provider slug = tool prefix (`memory_*`, `gcal_*`, …).
  - *hosted* — in-process, Postgres-backed (memory).
  - *upstream* — our own adapters over a remote API (Google), using
    credentials from the encrypted vault; tokens never exposed to agents.
    (Proxying a third-party remote MCP server is the same kind, later.)
  - *actuator* — user-run daemon dials out over WSS and speaks MCP as a
    server; ZOS is the MCP client on that socket.
- **Visibility.** Actuator tools are listed **only while connected**; ZOS
  emits `notifications/tools/list_changed` on connect/disconnect. Stale
  calls get `ACTUATOR_OFFLINE`; a socket lost mid-call returns
  `ACTUATOR_DISCONNECTED_MID_CALL` (the command may have run — don't retry blindly).
- **Policy — tool facts vs client decisions, kept separate.**
  - *Tier* is a global fact about a tool: `read | write | external`
    (unknown → external). It drives MCP tool annotations and the default
    decision. Never downgraded to grant access.
  - *Decision* per client: `allow | deny | require_approval`. Default from
    tier (read → allow, write → allow, external → require_approval);
    per-client overrides by tool pattern.
- **Confirmation.** `require_approval` is enforced by ZOS: it asks via MCP
  elicitation (showing the exact, untruncated call) when the client
  declares that capability; otherwise the call is denied with a reason that
  names the override to set. Annotations (`readOnlyHint`, `destructiveHint`,
  `openWorldHint`) are published as facts; what clients do with them is
  their business.
- **Hosted memory.** `memory_put / get / search / delete` (Postgres
  full-text + trigram). What agents store is up to them.
- **Audit.** Append-only `events`; args stored as digests by default.

## V0 providers — one per kind (Zubin, 2026-11-23)

- **hosted `memory`.**
- **upstream `gcal` + `gmail`** — two providers sharing one Google
  credential (one sign-in). No other services. They deliberately exercise
  the two upstream shapes (Zubin, 2026-11-24):
  - `gcal` — **MCP proxy**: ZOS is an MCP client to Google's remote
    Calendar MCP (`calendarmcp.googleapis.com/mcp/v1`) and re-exposes its
    tools namespaced, under ZOS policy and audit. Our REST adapter stays as a
    config-selected fallback while Google's server is in preview.
  - `gmail` — **API adapter**: our own tools over the Gmail REST API
    (Google's Gmail MCP has no send).
- **actuator `mac`** — exactly two tools:
  - `mac_messages_search` — read-only search over `~/Library/Messages/chat.db`.
  - `mac_shell_run` — run a command in a persistent local tmux session.

## Tool naming

`<prefix>_<tool>`, charset `[a-zA-Z0-9_-]`, prefixes contain no `_`, total
≤ 48 chars (deterministic truncate + 6-char hash on overflow). The portable
subset any MCP client and model API accepts. E.g. `memory_search`,
`gcal_list_events`, `gmail_search`, `mac_shell_run`, `zos_status`. Prefix
`zos` is reserved for router-owned tools.

## V1 seams (why V0 won't need rework)

The core never knows about agents. Anything later is either a **provider**
(behind the router) or a **client** (in front of it):
- Channels-as-buffer (spike later) → a provider with `messages_*` tools.
- A ZOS-hosted runtime / proactive agents → a separate service that is just
  another OAuth client.
- Async approvals for headless clients → a new strategy behind the existing
  `require_approval` decision.

Seams kept in V0: `clients.trust_ceiling` (`owner | untrusted`), monotonic
`events.id` (future stream cursor), three-valued decision, `ToolProvider`.

## Decided

- Python 3.12, uv, FastAPI/Starlette, official `mcp` Python SDK, psycopg 3,
  plain SQL migrations. One service + Postgres.
- Self-deploy = one container + any Postgres. Zubin: Fly + Neon.
- Upstream credentials: AEAD (AES-GCM) with key from env/Fly secret, never
  in DB; `key_version` for rotation; per-credential refresh lock.
- V0 has one space per principal; every client binds to it.

## Deferred

Everything in `later/`. Plus: MCP prompts/resources, third-party MCP
proxying, contact-name resolution for Messages, multi-user, vector search.
