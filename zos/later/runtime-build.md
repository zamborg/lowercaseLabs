# ZOS — v0 build order

Status: agreed (review 04, 2026-11-23). Branch pick waits on Zubin's vision
answer ("what comes first"). M0–M1 are shared and can start regardless.
Every milestone demo becomes a scripted e2e check — we verify ourselves.

## Shared

- **M0 — Skeleton, DB, deploy.** uv project, SQL migrator +
  `0001_init.sql` (all of SCHEMA.md), `zos` CLI (`init`, `token
  create/revoke`), FastAPI `/healthz` on Fly against Neon. Deploying in M0
  de-risks hosting before anything depends on it.
  *Demo:* `zos init` + token + `/healthz` live on Fly.
- **M1 — Gateway core, policy, stores.** `CallContext`, pinned
  `tool_policies`, approvals (JCS hash, atomic `pending → executing`),
  memories (FTS/trigram, pinned), artifacts (>8KB offload), threads/messages,
  event writer (lock + fanout). `zos approve <id>` as the first approval UI.
  *Demo (integration):* owner `memory.put` ok; pinned/untrusted write →
  `pending_approval`; approve executes stored args exactly once.

## Branch A — messaging assistant first

- **A2 — Dispatched agent + `zos chat`.** `load_snapshot` / pure
  `build_prompt`, runs worker (SKIP LOCKED, leases, coalescing),
  LLMProvider (Anthropic + OpenRouter). CLI chat as the first channel.
  *Demo:* multi-turn chat using memory; kill mid-run → lease recovery.
- **A3 — Telegram + live approvals.** Webhook, idempotency, inline
  Approve/Reject → core executes + continuation run.
- **A4 — Bridges.** WSS + MCP, upgrade auth, namespacing, ping,
  `BRIDGE_OFFLINE`; `zos-bridge` shim wrapping Mac iMessage/Contacts.
  *Demo:* "text Mom I'm late" → Approve → sent; sleeping Mac → clean error.
- **A5 — Schedules, compaction, provenance, MCP HTTP** for Claude Code/Codex.

## Branch B — substrate first

- **B2 — MCP HTTP gateway** (tokens → CallContext; memory/threads/artifacts/
  `approval.status`). *Demo:* Claude Code writes, Codex reads.
- **B3 — Bridges** (= A4), tools exposed to interactive agents.
- **B4 — Telegram as approval channel** for interactive calls.
- **B5 — Dispatched agent, schedules, compaction, provenance** (= A2 + A5).

Both branches converge on the same system.

## Layout

```
zos/
├── pyproject.toml          # FastAPI, FastMCP, psycopg 3, httpx
├── migrations/             # 0001_init.sql, ...
├── src/zos/
│   ├── db/                 # pool, migrator, repos, event writer
│   ├── gateway/            # CallContext, policy, approvals, built-in tools
│   ├── context/            # load_snapshot + pure build_prompt
│   ├── dispatcher/         # runs worker, schedules, compaction, llm/
│   ├── transport/          # FastAPI, MCP HTTP, bridge WSS, SSE, telegram
│   └── cli.py
├── bridge/                 # standalone zos-bridge (own pyproject, tiny deps)
└── tests/{unit,integration}
```

## Tests

- **Unit (no DB):** build_prompt goldens (byte-identical cached prefixes),
  JCS hashing, policy matrix, fake LLMProvider.
- **Integration:** local Postgres 17 (matches Neon), real migrations,
  ephemeral DB — run coalescing, double-approve race, FTS, cursor order,
  WSS reconnect.
- **Neon:** dev branch smoke before deploy only (pg_trgm, pooled vs direct,
  cold start). Never pytest against prod.
