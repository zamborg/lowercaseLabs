# Trust Engine

Thin trust proxy for Claude Code. It stores hook telemetry, defaults to zero trust, compiles repeated approvals into a generated `skill.md`, and exposes a local dashboard where you can pin runtime rules.

## What It Does

- Logs every Claude Code hook event that you wire into it
- Forces `PreToolUse` into `ask` by default when zero trust is enabled
- Lets you promote exact tool patterns into `allow once`, `allow always`, `allow-more-complex`, `escalate`, or `deny`
- Compiles a generated trust skill from observed history
- Serves a lightweight local dashboard at `http://127.0.0.1:8787`

## Why This Shape

Claude Code hooks are sufficient to build a trust proxy today, but native permission dialogs are not a perfect canonical approval log. This MVP treats hook telemetry as the source of truth and uses the proxy plus generated skill as the runtime layer.

The important limitation is that native manual permission choices are only partially inferable:

- `PostToolUse` and `PostToolUseFailure` imply the user approved execution
- `PermissionRequest` without a later tool outcome may mean deny, cancel, or abandonment
- If you need exact human labels for high-risk workflows, prefer semantic MCP tools and use elicitation

## Quickstart

1. Start the server:

```bash
npm start
```

2. Copy the hook example into your Claude Code settings:

```bash
mkdir -p .claude
cp examples/claude-settings.json .claude/settings.json
```

3. Run Claude Code from this project root so `node hooks/claude-trust-proxy.mjs` resolves correctly.

4. Open the dashboard:

[http://127.0.0.1:8787](http://127.0.0.1:8787)

## Data Layout

Generated data lives in `.trust-engine-data/`:

- `events.jsonl`: raw hook telemetry plus matched rule metadata
- `manual-rules.json`: runtime overrides you set in the dashboard
- `compiled-policy.json`: suggested rules derived from history
- `trust-skill.md`: generated trust skill for Claude

## Runtime Model

The trust proxy applies decisions in this order:

1. Exact manual rule match
2. If zero trust is enabled, `PreToolUse` returns `ask`
3. `PermissionRequest` is logged and only auto-resolved when a matching manual rule says `allow` or `deny`

The three user-facing trust states are:

- `allow once`: allows one future matching call, then falls back to escalation
- `allow always`: always auto-allows that exact fingerprint
- `allow-more-complex`: never auto-allows; it remains an escalation-class pattern and influences the generated skill

## Optional Claude-Backed Builder

By default the skill compiler is heuristic. If you want `claude -p` to rewrite the generated skill:

```bash
TRUST_ENGINE_USE_CLAUDE=1 npm run build:trust
```

## Sensible Next Steps

- Replace exact-match fingerprints with domain-specific MCP semantics for email, messages, and calendar
- Add identity and multi-user separation to the event store
- Replace JSONL with a database once you want server-side search and analytics
- Add a dedicated escalation workflow instead of relying only on native permission dialogs
