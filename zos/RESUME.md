# Resume notes (for the next Claude session)

Read `PLAN.md` first — it's the current design. This file is how to pick
the conversation back up.

## Roles

- **Zubin owns the vision.** Bring him only high-level / vision-alignment
  questions. He trusts us with every technical decision.
- **Opus (Claude Code) leads** and is Zubin's single point of contact.
- **Design partner:** Gemini 4 Neon in Antigravity CLI, running in Shepherd
  pane `27359a` (workstream ZOS). Talk to it with `shepherd send 27359a "..."`
  (single line; ask it to reply with a single sentinel word), read its pane
  with `shepherd peek`. Shepherd rules: `~/.shepherd/AGENTS.md`.
- **Shared files:** ZOS workstream artifact dir
  `~/.shepherd/workstreams/32eee62d-e739-4ce7-a2cf-2a1d4c79b65e/`
  (`notes.md`, `review-NN-*.md`).

## Review protocol

Brief → both write independent takes (no anchoring) → Opus reconciles →
partner responds → Opus folds decisions into PLAN.md.

## History

1. Reviewed `../zass` prototype; Zubin shared a Gemini "first-principles"
   design; we kept the event log, server-enforced tiers, thin interfaces,
   artifacts, principal/space columns; deferred mesh etc.
2. Fresh start as ZOS on Neon Postgres. Schematic v1: channels, bridges,
   gateway/policy, stores.
3. **Review 01 (2026-11-23):** dispatcher, approvals, connectors, hosting,
   provenance, runtime — all decided; see PLAN.md.
4. Reviews 02–04 designed context assembly, schema, build order for a
   runtime-owning core.
5. **Zubin's audit: scope creep.** V0 is only an MCP context-routing layer;
   agents (native Claude Code / Codex) run outside. Runtime docs moved to
   `later/`.
6. **Review 05 (re-scope):** new PLAN.md, SCHEMA.md, BUILD.md.
7. Zubin locked the boundary (ZOS = behind the MCP interface only) and the
   V0 providers (memory; gcal + gmail; Mac messages search + tmux shell).
8. **Review 06:** final executor-eye audit; fixes applied. **Design final.**
   BUILD.md is the executor handbook; Zubin spawns the executor.

## Status

Design final. Next: executor agent works BUILD.md from M0. Don't re-expand
scope: no runtime, channels, or Telegram in V0.

## Working with Zubin

- High-level, concise, lead with the recommendation.
- He talks by voice — transcripts are rough; confirm ambiguous terms briefly.
- Don't build until the design for that piece is agreed.
