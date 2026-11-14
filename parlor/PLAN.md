# parlor (working title)

Local, native Mac chat app for persistent agents (personal assistants through roleplay characters). Fully local. Mockups: `mockups/index.html`.

## Shape
- One SwiftUI app. Core is a Swift package with no UI imports, driven by both the app and a headless `parlor-cli`.
- Three runtimes behind protocols: LLM (MLX, in process, actor), image worker (child process, MLX), MCP servers (stdio/http children via official Swift SDK).
- Files are the database under `~/parlor/`: `agents/*.md` (frontmatter + persona), `chats/<agent>/log.jsonl` + `summary.md`, `media/`, `models/`, `mcp.json` (Claude Desktop format), `CLAUDE.md` for external agents. FSEvents watcher reloads live.
- One chat per agent. Prune = delete messages, cut after a point, compact now.
- `AgentLoop` is a ~150 line protocol implementation: stream -> tool calls -> ToolHub -> repeat; Compactor summarizes oldest turns when tokens exceed `compact_at * context`.
- Image generation is just `image.*` tools in ToolHub. Nothing special-cased in the loop.
- Fresh chat snapshots the tool list; `tools/list_changed` adds a "tools updated" divider to open chats.

## Decisions
- Not Open WebUI (Python/Svelte web server; fights native + simple).
- MLX via `mlx-swift-lm`; weights from `mlx-community` on Hugging Face via `swift-huggingface`.
- Image: Swift MLX port of Z-Image Turbo / FLUX.2 Klein in a child process; Python `mflux` as fallback behind the same worker interface. ~20-40 s per image on M1 Max.
- Not sandboxed, Developer ID distribution (needed to spawn MCP servers).
- Defaults: macOS 26, Swift 6, no telemetry, HF downloads are the only network use. First models: Qwen3 30B-A3B 4-bit (assistants), Llama 3.2 3B 4-bit (characters), Z-Image Turbo (images).

## Phases (each has a self-run gate; no "please try it")
0. Spike (2-3d): streaming MLX chat + zass MCP tool round trip. Gate: model calls tools correctly 9/10 scripted prompts.
1. Core (~1w): Store, AgentLoop, Compactor, ToolHub, LLM actor, `parlor-cli`. Gate: CLI test, 60 messages, compaction, tool call, survives relaunch.
2. Chat + agents UI (~1w): sidebar, streaming chat, tool chips, agent editor with live file preview, prune. Gate: XCUITest flow + screenshots vs mockups.
3. Models (~4d): HF search, resumable download, memory gauge, load/unload on agent switch. Gate: interrupt/resume download, assign, chat; gauge within 10% of Activity Monitor.
4. Tool mounts (~4d): mcp.json editor, status/restart, per-agent allowlist, list_changed. Gate: add server mid-chat, next turn uses it.
5. Images (~1w): worker, `image.generate`, inline render, prompt/seed disclosure. Gate: "draw X" -> image in log < 40 s, worker memory freed on idle.
6. Make it ours: templates, agent pack import/export, folder CLAUDE.md, optional app-hosted MCP for external Claude Code.

## Risks
- Small local models' tool-calling reliability (phase 0 gate).
- Memory: 17 GB LLM + 12 GB image on 64 GB; image worker unloads on idle; warn before downloads that won't fit.
- Swift MLX API churn: pin versions, wrap behind our protocols.
- Swift image ports are community projects: worker is a process boundary so the fallback is cheap.

## Status (2026-11-13)
Built and verified on an M1 Max / 64 GB:
- Phase 0 gate: Qwen3 4B tool-calling on the real zass MCP server, 9/10 scripted prompts (`parlor-cli spike`).
- Phase 1 gate: unit/flow tests with a scripted model: file-watch agent creation, tool round trip, permission, 60-message compaction, relaunch persistence, prune (`Tests/CoreTests.swift`). Real-model compaction checked through the CLI (name recalled after the summary replaced the history).
- Phases 2-4 in the app: chat, agent editor with live file preview, prune mode, models (HF search, resumable download, memory gauge), tools (mcp.json editor, add server, restart, list_changed marker). Verified by launching the real app and capturing windows; live reload checked by editing agent files while the app ran.
- Phase 5: `ImageProvider` + `parlor-imagegen` worker (Z-Image Turbo, int8) embedded under `Contents/Resources/ImageWorker`.

Not verified: click-driven UI flows (XCUITest needs Developer Mode, off on this Mac and enabling needs sudo), and light mode.
