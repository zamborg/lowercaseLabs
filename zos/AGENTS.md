# ZOS — Agent Development Notes

Read `PLAN.md`, `SCHEMA.md`, and `BUILD.md` before changing anything.

## Running checks

All commands run from `zos/`:

```bash
uv run pytest
uv run ruff check
```

Actuator package (from `zos/actuator/`):

```bash
uv run pytest
```

## Ground rules

1. **Stay inside the boundary.** ZOS is strictly everything behind the MCP interface and nothing in front of it. Do not build agent runtimes, channels, schedulers, prompts/resources, or anything in `later/`.
2. **Build to the MCP spec.** Use `mcp.server.lowlevel.Server` (not `FastMCP` decorators) so tool lists are dynamic per client and actuator state and calls receive `CallContext`. Verify SDK APIs against the installed `mcp` source before relying on them. Claude Code / Codex are manual smoke tests only, never design inputs.
3. **Green tests per task.** Every task must end with passing `uv run pytest` and `uv run ruff check`. Each milestone ends with its scripted **Check**.
4. **Record decisions.** Log every unsettled design choice or deviation in `DECISIONS.md` (date, decision, why). Never silently diverge from `PLAN.md` or `SCHEMA.md`.
5. **Privacy.** Never print, log, or commit content from real personal data (Messages, Gmail, Calendar). Tests use fixtures; live smoke tests assert counts and shapes only. Never commit secrets (`.env`, Google client secrets, tokens).
6. **Git.** `zos/` lives inside the `lowercaseLabs` monorepo. Work on one branch per milestone (`zos/m0-scaffold`, `zos/m1-core`, ...), commit per task with clear messages. Do not push, open PRs, or merge unless asked.
7. **Human steps.** Steps requiring accounts, secrets, macOS permissions, or Google Cloud console (marked 👤 in `BUILD.md`) must be delegated to Zubin with exact instructions.
