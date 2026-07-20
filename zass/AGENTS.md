# zass — agent instructions

Read `CLAUDE.md` in this directory and follow it exactly. It is the operating
manual for zass, the personal assistant this repo implements; it applies to
every agent runtime (Codex, Claude Code, or anything else), not just Claude.

Runtime notes for non-Claude agents:

- The internal MCP server is started with:
  `uv run --project mcp/zass zass-mcp` (stdio transport).
  For Codex CLI, register it in `~/.codex/config.toml`:

  ```toml
  [mcp_servers.zass]
  command = "uv"
  args = ["run", "--project", "/absolute/path/to/zass/mcp/zass", "zass-mcp"]
  ```

- If you cannot run MCP servers, operate on `state/` directly — it is plain
  markdown with YAML frontmatter and is the source of truth. Match the
  frontmatter schemas of existing files.
- Skills/playbooks are markdown workflows in `.claude/skills/*/SKILL.md`.
  Read and follow them when the user invokes the corresponding command name.
