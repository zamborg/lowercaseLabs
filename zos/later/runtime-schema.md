# ZOS — v0 Postgres schema

Status: agreed design (review 03, 2026-11-23). Base draft by Gemini; the
"Resolutions" section below overrides anything it conflicts with.

## Resolutions (override the draft below)

1. **principal/space placement:** every domain/operational row carries
   `(principal_id, space_id)` (keeps future row-level security trivial);
   auth/connector tables carry `principal_id` only (§6.1 adopted).
2. **Runs:** separate partial uniques for `running` and `queued` per thread,
   with coalescing upsert on the queued row (replaces "drain on finish").
   The claim query must skip threads that already have a `running` run.
3. **Orphan approvals (§6.2):** `approvals.thread_id/run_id` nullable;
   `principals.default_channel_connector_id` routes approval prompts and
   proactive output; interactive clients poll via `approval.status(id)` MCP
   tool. Approval for an interactive call does not enqueue a run.
4. **Proactive delivery (§6.5):** `schedules.target_connector_id` NULL →
   falls back to the principal default channel.
5. **Args:** store RFC 8785 JCS text + SHA-256; execute from that text only
   (§6.3 adopted). No `threads.latest_summary_id` (§6.4 adopted).
6. **Event cursor gaps:** identity ids are allocated at INSERT, not COMMIT,
   so concurrent txns can commit out of order and an SSE reader at
   `after=N` could skip an event forever. v0: insert the `events` row as the
   last statement of each txn while holding one in-process async lock until
   commit; live delivery is in-process fanout after commit; the cursor is
   only for reconnect catch-up. Revisit at 2+ replicas.

---

## 1. Conventions & extensions

- **Extensions:** `pg_trgm` (fuzzy memory search). Built-in `gen_random_uuid()` (or app-generated UUIDv7) for entity PKs.
- **Monotonic cursors:** `BIGINT GENERATED ALWAYS AS IDENTITY` for `events.id` and `messages.id` (safe `>` cursors for SSE and compaction watermarks).
- **Timestamps:** `TIMESTAMPTZ NOT NULL DEFAULT now()`.

---

## 2. Tables (16 total)

### Identity & auth

**`principals`** — *tx: CLI bootstrap*
- `id UUID PK`, `slug TEXT UNIQUE NOT NULL`, `display_name TEXT NOT NULL`, `timezone TEXT NOT NULL DEFAULT 'UTC'`, `created_at`

**`spaces`** — *tx: CLI bootstrap*
- `id UUID PK`, `principal_id UUID NOT NULL FK → principals`, `slug TEXT NOT NULL`, `name TEXT NOT NULL`, `created_at`
- `UNIQUE (principal_id, slug)`

**`agent_profiles`** — *tx: CLI / admin*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `slug TEXT NOT NULL`, `provider TEXT NOT NULL CHECK (provider IN ('anthropic','openrouter'))`, `model TEXT NOT NULL`, `system_prompt TEXT NOT NULL`, `allowed_tools TEXT[] NOT NULL DEFAULT '{}'`, `high_water_tokens INT NOT NULL DEFAULT 24000`, `low_water_tokens INT NOT NULL DEFAULT 8000`, `is_default BOOLEAN NOT NULL DEFAULT false`, `created_at`, `updated_at`
- `UNIQUE (space_id, slug)`
- `UNIQUE (space_id) WHERE is_default` *(at most one default profile per space)*

**`clients`** — *tx: CLI client-add*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `default_space_id UUID NOT NULL FK → spaces`, `agent_profile_id UUID NOT NULL FK → agent_profiles`, `name TEXT NOT NULL`, `kind TEXT NOT NULL CHECK (kind IN ('mcp_interactive','internal'))`, `created_at`
- `UNIQUE (principal_id, name)`

**`connectors`** — *tx: CLI enroll; bridge connect/disconnect; channel SSE ack*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `kind TEXT NOT NULL CHECK (kind IN ('bridge','channel'))`, `slug TEXT NOT NULL` *(bridge tool namespace, e.g. `mac`)*, `config JSONB NOT NULL DEFAULT '{}'`, `online BOOLEAN NOT NULL DEFAULT false`, `last_seen_at TIMESTAMPTZ`, `last_event_id BIGINT NOT NULL DEFAULT 0` *(channel outbound cursor)*, `created_at`
- `UNIQUE (principal_id, kind, slug)`

**`api_tokens`** — *tx: CLI create/revoke; throttled auth touch*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `client_id UUID FK → clients ON DELETE CASCADE`, `connector_id UUID FK → connectors ON DELETE CASCADE`, `token_hash BYTEA UNIQUE NOT NULL` *(SHA-256)*, `scopes TEXT[] NOT NULL DEFAULT '{}'`, `expires_at TIMESTAMPTZ`, `revoked_at TIMESTAMPTZ`, `last_used_at TIMESTAMPTZ`, `created_at`
- `CHECK ((client_id IS NOT NULL)::int + (connector_id IS NOT NULL)::int = 1)` *(real FKs instead of polymorphic UUID)*
- `INDEX (client_id) WHERE revoked_at IS NULL`, `INDEX (connector_id) WHERE revoked_at IS NULL`

### Tools & policy

**`bridge_tools`** — *tx: bridge connect / `list_changed` sync*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `connector_id UUID NOT NULL FK → connectors ON DELETE CASCADE`, `raw_name TEXT NOT NULL`, `namespaced_name TEXT NOT NULL` *(`mac__imessage_send`)*, `description TEXT NOT NULL`, `input_schema JSONB NOT NULL`, `advisory_tier TEXT CHECK (advisory_tier IN ('read','write','external'))`, `removed_at TIMESTAMPTZ`, `updated_at`
- `UNIQUE (principal_id, namespaced_name)`, `UNIQUE (connector_id, raw_name)`

**`tool_policies`** — *tx: CLI / admin policy (separate from `bridge_tools` so bridge sync can never touch tiers)*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `space_id UUID FK → spaces` *(NULL = principal-wide)*, `tool_pattern TEXT NOT NULL` *(exact name or glob)*, `pinned_tier TEXT NOT NULL CHECK (pinned_tier IN ('read','write','external'))`, `sensitive_read BOOLEAN NOT NULL DEFAULT false`, `updated_at`
- `UNIQUE NULLS NOT DISTINCT (principal_id, space_id, tool_pattern)` *(unknown tool defaults to `'external'` in code)*

### Conversation & execution

**`threads`** — *tx: inbound channel; schedule fire; `/agent` route; run finish*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `channel TEXT NOT NULL`, `external_thread_id TEXT`, `title TEXT`, `agent_profile_id UUID NOT NULL FK → agent_profiles`, `tainted BOOLEAN NOT NULL DEFAULT false`, `last_processed_msg_id BIGINT NOT NULL DEFAULT 0`, `last_activity_at TIMESTAMPTZ NOT NULL DEFAULT now()`, `created_at`
- `UNIQUE (space_id, channel, external_thread_id) WHERE external_thread_id IS NOT NULL`
- `INDEX (space_id, last_activity_at DESC)` *(active-threads digest & `threads.list_recent`)*

**`messages`** — *tx: inbound channel; run turn; approval execution; schedule fire*
- `id BIGINT GENERATED ALWAYS AS IDENTITY PK`
- `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `thread_id UUID NOT NULL FK → threads ON DELETE CASCADE`, `run_id UUID FK → runs`
- `role TEXT NOT NULL CHECK (role IN ('user','assistant','tool','system'))`, `sender_trust TEXT NOT NULL CHECK (sender_trust IN ('owner','untrusted','system'))`
- `channel TEXT`, `external_id TEXT`
- `blocks JSONB NOT NULL`, `plain_text TEXT NOT NULL DEFAULT ''`, `fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', plain_text)) STORED`, `token_estimate INT NOT NULL DEFAULT 0`, `created_at`
- `UNIQUE (space_id, channel, external_id) WHERE channel IS NOT NULL AND external_id IS NOT NULL` *(inbound channel idempotency)*
- `INDEX (thread_id, id)`, `GIN INDEX (fts)`

**`thread_summaries`** — *tx: post-turn async compaction (append-only)*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `thread_id UUID NOT NULL FK → threads ON DELETE CASCADE`, `parent_summary_id UUID FK → thread_summaries`, `through_msg_id BIGINT NOT NULL FK → messages`, `summary_text TEXT NOT NULL`, `token_estimate INT NOT NULL`, `model TEXT NOT NULL`, `created_at`
- `INDEX (thread_id, through_msg_id DESC)` *(drops need for circular `threads.latest_summary_id` FK)*

**`runs`** — *tx: trigger enqueue; dispatcher claim; heartbeat; finish/suspend/cancel*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `thread_id UUID NOT NULL FK → threads`, `agent_profile_id UUID NOT NULL FK → agent_profiles`
- `status TEXT NOT NULL CHECK (status IN ('queued','running','completed','awaiting_approval','cancelled','failed'))`
- `trigger_kind TEXT NOT NULL CHECK (trigger_kind IN ('message','schedule','approval_result','drain'))`, `trigger_trust TEXT NOT NULL CHECK (trigger_trust IN ('owner','untrusted'))`, `trigger_event_ids BIGINT[] NOT NULL DEFAULT '{}'`
- `loaded_event_ids BIGINT[]`, `prompt_hash BYTEA`, `input_tokens INT`, `output_tokens INT`, `cached_tokens INT`
- `lease_owner TEXT`, `lease_expires_at TIMESTAMPTZ`, `cancel_requested BOOLEAN NOT NULL DEFAULT false`, `error TEXT`, `started_at TIMESTAMPTZ`, `finished_at TIMESTAMPTZ`, `created_at`
- **`UNIQUE (thread_id) WHERE status = 'running'`** *(invariant: one active run per thread)*
- **`UNIQUE (thread_id) WHERE status = 'queued'`** *(coalesces mid-run / burst triggers via `ON CONFLICT DO UPDATE`, taking `min` trust and appending `trigger_event_ids`)*
- `INDEX (created_at) WHERE status = 'queued'`, `INDEX (lease_expires_at) WHERE status = 'running'`

**`approvals`** — *tx: gateway policy intercept; user decision (`pending → executing` atomic claim); execution finish*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `thread_id UUID FK → threads`, `run_id UUID FK → runs`, `client_id UUID FK → clients`, `agent_profile_id UUID NOT NULL FK → agent_profiles`
- `tool_name TEXT NOT NULL`, `tool_call_id TEXT`, `args_canonical_json TEXT NOT NULL` *(exact JCS string core will parse & execute)*, `args_hash BYTEA NOT NULL` *(SHA-256 of `tool_name + args_canonical_json + actor_id`)*
- `status TEXT NOT NULL CHECK (status IN ('pending','executing','executed','rejected','expired','failed'))`
- `decided_by_principal_id UUID FK → principals`, `decided_at TIMESTAMPTZ`, `expires_at TIMESTAMPTZ NOT NULL`, `result_json JSONB`, `error TEXT`, `created_at`
- `CHECK (run_id IS NOT NULL OR client_id IS NOT NULL)`
- `INDEX (thread_id) WHERE status = 'pending'`
- *Exactly-once execution invariant:* decision tx does `UPDATE approvals SET status='executing', decided_at=now() WHERE id=$1 AND status='pending' AND expires_at > now() RETURNING *` before calling the tool.

### Schedules, memory, artifacts, events

**`schedules`** — *tx: tool/CLI CRUD; dispatcher fire (`FOR UPDATE SKIP LOCKED`)*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `thread_id UUID FK → threads`, `agent_profile_id UUID NOT NULL FK → agent_profiles`
- `name TEXT NOT NULL`, `kind TEXT NOT NULL CHECK (kind IN ('once','cron'))`, `cron_expr TEXT`, `timezone TEXT NOT NULL`, `prompt TEXT NOT NULL`, `enabled BOOLEAN NOT NULL DEFAULT true`, `next_run_at TIMESTAMPTZ`, `last_fired_at TIMESTAMPTZ`, `created_at`
- `CHECK (kind = 'once' OR cron_expr IS NOT NULL)`, `INDEX (next_run_at) WHERE enabled AND next_run_at IS NOT NULL`
- *Fire idempotency:* enforced via `events.idempotency_key = 'schedule:' || schedule_id || ':' || scheduled_for` in the same tx that advances `next_run_at`.

**`memories`** — *tx: `memory.put/delete` or approval-execution tx*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `profile_id UUID FK → agent_profiles` *(NULL = shared)*
- `key TEXT NOT NULL`, `content TEXT NOT NULL`, `tags TEXT[] NOT NULL DEFAULT '{}'`, `pinned BOOLEAN NOT NULL DEFAULT false`, `token_estimate INT NOT NULL DEFAULT 0`
- `fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', key || ' ' || content)) STORED`, `created_at`, `updated_at`
- `UNIQUE NULLS NOT DISTINCT (space_id, profile_id, key)`, `INDEX (space_id, profile_id) WHERE pinned`, `GIN INDEX (fts)`, `GIN INDEX (tags)`, `GIN INDEX (content gin_trgm_ops)`

**`artifacts`** — *tx: `artifact.write` or gateway >8KB tool-output offload*
- `id UUID PK`, `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `thread_id UUID FK → threads`, `run_id UUID FK → runs`
- `mime_type TEXT NOT NULL`, `title TEXT`, `summary TEXT`, `size_bytes INT NOT NULL`, `sha256 BYTEA NOT NULL`, `content_text TEXT`, `content_bytes BYTEA`, `blob_url TEXT`, `created_at`
- `CHECK ((content_text IS NOT NULL)::int + (content_bytes IS NOT NULL)::int + (blob_url IS NOT NULL)::int = 1)`

---

## 3. `events` envelope & v0 event types

**`events`** — *tx: appended in the same transaction as every state mutation (outbox)*
- `id BIGINT GENERATED ALWAYS AS IDENTITY PK`
- `principal_id UUID NOT NULL FK`, `space_id UUID NOT NULL FK`, `thread_id UUID`, `run_id UUID`, `client_id UUID`, `connector_id UUID`, `agent_profile_id UUID`
- `trust_level TEXT NOT NULL CHECK (trust_level IN ('owner','untrusted','system'))`
- `type TEXT NOT NULL`, `outcome TEXT NOT NULL DEFAULT 'ok' CHECK (outcome IN ('ok','denied','pending_approval','error'))`
- `idempotency_key TEXT`, `payload JSONB NOT NULL DEFAULT '{}'`, `created_at`
- `UNIQUE (space_id, idempotency_key) WHERE idempotency_key IS NOT NULL`
- `INDEX (space_id, id)` *(SSE cursor replay)*, `INDEX (thread_id, id) WHERE thread_id IS NOT NULL`

**v0 event types:**
- Threads/messages: `thread.created`, `thread.routed`, `thread.compacted`, `message.inbound`, `message.outbound`
- Runs: `run.queued`, `run.started`, `run.completed`, `run.awaiting_approval`, `run.cancelled`, `run.failed`
- Tools/policy: `tool.called`, `tool.succeeded`, `tool.failed`, `tool.denied`
- Approvals: `approval.requested`, `approval.approved`, `approval.rejected`, `approval.expired`, `approval.executed`
- Schedules/connectors/stores: `schedule.fired`, `connector.online`, `connector.offline`, `bridge.tools_synced`, `memory.written`, `memory.deleted`, `artifact.created`

---

## 4. Canonical `messages.blocks` format

Array of typed objects (`blocks JSONB NOT NULL`). Every `tool_call` in an `assistant` message MUST be followed by a `tool` message referencing `call_id` (including `approval_request`, which provider adapters translate into a valid `tool_result` so Anthropic/OpenAI never reject the continuation transcript with dangling `tool_use`):

```json
{"type": "text", "text": "..."}
{"type": "tool_call", "call_id": "tc_01", "tool": "mac__imessage_send", "args": {"to": "+1...", "body": "hi"}}
{"type": "tool_result", "call_id": "tc_01", "tool": "mac__imessage_send", "is_error": false, "content": [{"type": "text", "text": "ok"}]}
{"type": "artifact_stub", "artifact_id": "uuid", "uri": "artifact://uuid", "mime": "text/plain", "size_bytes": 42010, "preview": "head...\n...tail"}
{"type": "approval_request", "approval_id": "uuid", "call_id": "tc_01", "tool": "mac__imessage_send", "summary": "Send iMessage to +1..."}
{"type": "approval_result", "approval_id": "uuid", "call_id": "tc_01", "tool": "mac__imessage_send", "decision": "approved", "is_error": false, "content": [{"type": "text", "text": "sent"}]}
```
*(An `artifact_stub` can appear inside `blocks` or inside `tool_result.content` / `approval_result.content`.)*

---

## 5. Migration tooling

**Plain numbered SQL files (`migrations/0001_init.sql`, ...) + a ~40-line Python runner (`schema_migrations` table), using `psycopg` v3 async.** Skip SQLAlchemy/Alembic: we rely heavily on Postgres-specific DDL (`UNIQUE NULLS NOT DISTINCT`, partial unique indexes, `GENERATED ... STORED` tsvectors, `pg_trgm`, `FOR UPDATE SKIP LOCKED`) that ORM autogenerate mangles or ignores.

---

## 6. Awkward spots flagged in `PLAN.md`

1. **"Every row carries `principal` + `space`."** Breaks on `principals`, `spaces`, `connectors`, `clients`, and `api_tokens` — e.g., Zubin's Mac bridge serves *all* his spaces, not one. Rule should be: every **domain/operational** row carries `(principal_id, space_id)`; auth/connector tables carry `principal_id` (and `default_space_id` on `clients`).
2. **Interactive MCP approvals have no `thread_id` / `run_id`.** `PLAN.md` says external calls from Claude Code need approval, and approvals enqueue a follow-up run — but interactive MCP calls have no thread/run. We need: nullable `thread_id`/`run_id` on `approvals`, a **default owner notification channel** on `spaces` (or `principals`) to route orphan approval prompts, and an `approval.status(id)` MCP tool so Claude Code can check the outcome.
3. **Canonical arg hashing vs `JSONB`.** Postgres `JSONB` normalizes numbers/keys differently than Python. Store `args_canonical_json TEXT` (RFC 8785 JCS) + its SHA-256 `args_hash`, and parse *that* exact string on execution — never hash `JSONB::text`.
4. **Circular FK if `threads` points at `thread_summaries`.** Drop `threads.latest_summary_id`; query latest summary via `INDEX (thread_id, through_msg_id DESC)` so compaction is a single `INSERT` with no circular FK.
5. **Channel outbound routing on `thread_id = NULL` schedules.** When a daily brief fires with no existing thread and creates one (`channel = 'schedule'`), which channel delivers it? Either `schedules` needs an explicit `target_channel` (e.g., `telegram`), or `spaces` needs a `default_channel` for proactive notifications.
