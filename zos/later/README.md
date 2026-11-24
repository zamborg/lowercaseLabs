# Later layers (not V0)

Reviews 01–04 designed an agent runtime: dispatcher, runs, schedules,
channels (Telegram), our own agent loop, context assembly, compaction,
async approvals. Zubin's 2026-11-23 audit ruled that out of V0: ZOS V0 is
only a context-routing layer, and agents run outside it.

These files are kept as reference for V1+. Per the V0 seam principle, any
of it returns as either a **provider** (behind the router) or a **client**
(in front of it), never as a back door into V0. Expect to revise them
before reuse; they assume a core that owns agents.

- `runtime-plan.md`, `runtime-schema.md`, `runtime-build.md`
- Spike later: channels as a queryable message buffer (a provider) that
  Claude Code / Codex query.
