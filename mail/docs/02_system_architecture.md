# Services (docker-compose)

1. postgres: canonical DB.
2. redis: job queue + ephemeral caches.
3. mail-sync:
   - IMAP: pull headers/bodies, thread IDs, labels, read states.
   - SMTP: send mail.
4. api:
   - Auth (local dev token-based), REST endpoints.
   - Orchestrates triage actions, job creation, and state transitions.
5. agent:
   - LLM runtime + tools (summarize, draft, classify, propose snooze, update notes).
6. tui:
   - terminal client that calls the API.

# Core flow

- mail-sync ingests emails -> DB stores canonical email objects -> api exposes inbox views.
- user triages via TUI -> api records decision -> optionally enqueues agent jobs.
- agent produces drafts, classifications, notes updates -> api persists -> user approves/send.

# Runtime invariants

- No service directly writes to DB except through its own DAO layer.
- All agent outputs are persisted with provenance metadata:
  - model, prompt hash, tool calls, timestamps, input email IDs.
