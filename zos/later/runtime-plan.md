# ZOS — plan (v0, high level)

Status: design. No code yet. Last updated 2026-11-23 (design review 01 with
Gemini partner; full reviews in the Shepherd ZOS workstream artifact dir).

## What ZOS is

A personal agent operating system: one backplane that all of Zubin's agents
share for state, tools, and communication. Agents do the reasoning; ZOS owns
data, permissions, and connections to the outside world.

ZOS is a fresh project. It is conceptually descended from `zass`
(`../zass`) but shares no code and has no dependency on it.

## Schematic

```
   Interactive agents                 Dispatched agent (in core)
   Claude Code · Codex                own tool loop · LLMProvider
          │ MCP                                 │
 ┌────────┴─────────────────────────────────────┴────────┐
 │                        ZOS core                       │
 │  Gateway + policy ── tools (namespaced, core-pinned)   │
 │  Dispatcher ── runs ledger · schedules · approvals    │
 │  Stores ── threads/messages · memory · artifacts      │
 │  events (append-only audit + feed)                    │
 └───┬──────────────────────────┬─────────────────────┬──┘
     │ HTTPS in / SSE out       │ WSS (MCP)           │
  Channels                   Bridges              Neon Postgres
  Telegram · Email           Mac · Sandbox        (state tables =
  iOS · Web                                        authority)
```

## Components

- **Interactive agents.** Claude Code, Codex connect as MCP clients while
  Zubin works. They touch ZOS state only through gateway tools.
- **Dispatched agent.** ZOS's own minimal tool loop (`build_prompt →
  llm.complete → gateway.execute`, stops on `pending_approval`). Thin
  `LLMProvider` edge; ZOS-canonical message format in Postgres, never a
  provider's wire format. This is what answers channel messages and schedules.
- **Dispatcher.** In core. Inbound message or schedule fire → enqueue a
  `run` (Postgres worker, `FOR UPDATE SKIP LOCKED`, leases). **One active run
  per thread**; messages that arrive mid-run are drained into one batched
  follow-up run. Explicit cancel. Thread routing = explicit `agent_profile`
  on the thread, default for new threads.
- **Schedules.** `schedules` table (`once` | `cron`, explicit timezone).
  Firing writes `schedule_fired`, enqueues a run, advances `next_run_at` in
  one transaction; idempotency `(schedule_id, scheduled_for)`.
- **Approvals.** Async, never a held-open call. Gateway hashes
  `(tool, canonical_args, run_id)`, writes `approvals`, returns
  `pending_approval`; turn ends. On approve, **core executes the stored call**
  and enqueues a follow-up run to report. The LLM never re-emits args.
- **Policy.** Every tool has a tier (read / write / external); core pins tiers
  (bridge-declared tiers are advisory; unknown → external). **Provenance from
  day one:** events and runs carry `trigger_trust` (`owner` | `untrusted`, set
  by core from the authenticated sender); a thread that ever received
  untrusted input stays tainted. Untrusted runs: write needs approval,
  external denied or approval, sensitive reads gated.
- **Channels (talk).** HTTPS `POST /v1/events` inbound with idempotency
  `(channel, external_id)`; outbound via SSE cursor (`after=<event_id>`) or
  push. Webhook channels (Telegram, email) are in-process adapters. Carry
  approval prompts.
- **Bridges (do).** Dial out over WSS, auth on upgrade, then speak **MCP**
  (`initialize`, `tools/list`, `tools/call`, `list_changed`, `ping`). Any stdio
  MCP server becomes a bridge with a tiny shim. Core namespaces tools
  (`mac__imessage_send`). Offline bridge: tools stay listed, calls fail fast
  `BRIDGE_OFFLINE`. Sandbox is a tool provider, never the agent's runtime.
- **Connectors** (channels + bridges) share auth/enrollment only: one
  `connectors` table, hashed scoped tokens, `last_seen_at`.
- **Stores.** Threads/messages (canonical history for dispatched turns),
  memory (minimal, tools-only), artifacts (plain table, no versioning in v0).
- **Context assembly.** `load_snapshot(db, run_id, now) -> ContextSnapshot`
  (I/O) then `build_prompt(snapshot, budget) -> PromptSpec` (pure; no DB,
  clock, env). PromptSpec = provider-agnostic blocks with cache-breakpoint
  flags. Golden tests incl. byte-identical cached prefixes across turns. Each
  run records loaded event ids + prompt hash + token counts (replayable).
  Layers, stable → volatile: tool catalog (sorted; offline tools unchanged) ◆
  system + profile prompt ◆ pinned memory ◆ thread summary + verbatim
  history ◆ then an uncached run frame: now/tz, channel, trust banner,
  offline bridges, pending approvals, ~80-token active-threads list (owner
  runs only), trigger events. Artifacts appear as stubs, never inline.
- **History compaction.** Hysteresis, not sliding window: append verbatim
  until a high-water mark (~24k tok), then post-turn async cheap-model
  summarization folds down to a low-water mark (~8k verbatim). Summaries
  append-only (`thread_summaries`, thread points at latest). Tool outputs
  >8KB are offloaded to artifacts at write time (preview + URI in message).
- **Memory.** Pinned facts (always loaded, ≤1.5k tok) + archival
  (`memory.search` via Postgres FTS/trigram, `get`, `put`, `delete`). Scoped
  `(principal, space, profile_id NULL)`: NULL = shared ("one Z"), set =
  profile-private ("many agents"). Memory writes are `write` tier; pinned
  writes always need owner approval.
- **Cross-thread.** `threads.list_recent / search / read` tools (FTS), plus
  the active-threads list in the run frame. Disabled for untrusted runs (no
  exfiltration via inbound mail).
- **Tool catalog.** Profile allowlist × policy; denied tools omitted,
  approval-required kept. Soft cap ~30. `tools.enable(namespace)` only once
  we exceed it.
- **Identity.** Principal (owns data), agent profile (config, not a
  credential), client (authenticated caller bound to principal + profile),
  connector (I/O). One `api_tokens` table (subject client|connector, hashed,
  scopes, expiry, revoked_at), issued by CLI. Dispatched agent has no token.
  Every entry normalizes to an immutable `CallContext(principal, space,
  client, profile, run, trust)`; policy sees only CallContext + pinned tiers.
  Interactive clients = owner trust; external tier still needs approval.
- **Postgres (Neon).** State tables are the authority; `events` is an
  append-only audit + notification feed (**not** event sourcing). Every row
  carries `principal` + `space`.

## Decided

- New project; ignore zass's implementation. Neon Postgres is source of truth.
- Everything above under Components.
- Hosting: single always-on Fly machine running one Python service
  (uv, FastAPI, FastMCP). Wake-ups in-process (`asyncio.Event` after commit;
  sleep until earliest `next_run_at`). No `LISTEN/NOTIFY` until 2+ replicas.
  Disable Neon scale-to-zero if cold start hurts; don't keep it awake by polling.
- LLM provider: thin interface with Anthropic (direct, for caching) and
  OpenRouter (model choice) adapters.

## Open — vision (Zubin)

1. First use case: messaging assistant (Telegram → acts on Mac) vs shared
   substrate for interactive agents. Decides v0 build order.
2. Proactive agents (briefs, follow-ups, inbox watching) in v0, or reactive only?
3. One assistant everywhere ("Z") vs several agents per job/thread?
4. Is it OK that channel messages are answered by ZOS's own agent loop, with
   Claude Code / Codex as interactive clients only?

## Open — technical (next review)

None. Schema: `SCHEMA.md` (review 03). Build order: `BUILD.md` (review 04).

## Deferred (not v0)

Hosted multi-model agent mesh, agent-to-agent delegation, multi-user spaces
in practice, self-modifying modules, vector search, artifact versioning,
dynamic content taint-tracking, multi-replica core.
