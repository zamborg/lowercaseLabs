# substrate

*(working name — rename freely)*

An infrastructure layer for **personal** AI: one authenticated, permissioned,
auditable, writable view of one person's life, exposed over a stable contract
so that any agent framework can consume it.

We are explicitly **not** building the agent. Frameworks churn; the substrate
shouldn't.

## Grounding

- Apple ecosystem
- Personal accounts only — no work identities
- Personal AI plans (Claude / OpenAI Pro), not API-billed infrastructure
- Claude Code as the first downstream consumer

## Phases

| Phase | State | Artifact |
|---|---|---|
| 1. Product scoping | in progress | `docs/00-research-brief.md` |
| 2. Technical design | not started | — |
| 3. MVP implementation | not started | — |

Phases will overlap on purpose — the point is to ping-pong until the primitives
are right while still shipping something that delivers real value.

## Docs

- [`docs/00-research-brief.md`](docs/00-research-brief.md) — landscape, Apple
  access-path reality check, protocol state, cost constraints, strawman
  primitives, open questions for Phase 1.

## Related work in this org

- `zass` (this repo) — filesystem-as-database assistant harness. Built, then
  starved: `state/` is effectively empty. The harness without the substrate.
- `ez-gz/Scheduler` — local control plane for agent CLIs, with schedule windows
  and usage budgets.
- `ez-gz/shepherd` — durable workstream/session organization over native agent
  CLIs.
