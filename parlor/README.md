# parlor

Local, native Mac chat app for persistent agents. Personal assistants to roleplay characters, one chat per agent,
fully on-device: MLX language models, a local image model, and MCP tools.

Design mockups and plan: `mockups/index.html`, `PLAN.md`.

## Build and run

Requires macOS 26, Xcode 26, `xcodegen` (`brew install xcodegen`).

```bash
scripts/build-worker.sh          # image worker (own package; first build compiles MLX, takes minutes)
xcodegen generate
xcodebuild -project Parlor.xcodeproj -scheme Parlor -configuration Debug -destination 'platform=macOS' -derivedDataPath build build
open build/Build/Products/Debug/Parlor.app
```

First launch creates `~/parlor/` with two example agents. Open **Models** and download a language model
(Qwen3 4B is a good start) and, for images, Z-Image Turbo.

## Layout

| Path | What |
|---|---|
| `Packages/ParlorCore` | All logic, no UI: store, agent loop, compactor, MCP hub, MLX runtime, model library, image provider |
| `App/` | SwiftUI app |
| `CLI/` | `parlor-cli`: drives the same engine headless (`spike`, `chat`, `download`, …) |
| `ImageWorker/` | `parlor-imagegen` helper (Z-Image via MLX), separate package so its MLX version is independent |
| `Tests/` | Unit and flow tests (scripted model, no weights) |
| `UITests/` | XCUITest flow (needs Developer Mode, see below) |

## State on disk (`~/parlor`, override with `PARLOR_HOME`)

```
CLAUDE.md                 explains the folder to external agents (written on first run)
mcp.json                  mounted MCP servers (Claude Desktop format)
agents/<id>.md            frontmatter + persona; edit from anywhere, the app reloads live
chats/<id>/log.jsonl      the one chat for that agent
chats/<id>/summary.md     rolling compaction summary
media/YYYY-MM/            generated images
models/<org>/<name>/      MLX weights (PARLOR_MODELS overrides the location)
```

## Swapping the agent loop

`AgentLoop` (`Loop.swift`) is a protocol. `SimpleLoop` is stream → tool calls → results → repeat.
Pass another implementation to `ParlorEngine(loop:)`.

## Testing

```bash
xcodebuild -project Parlor.xcodeproj -scheme Parlor -destination 'platform=macOS' -derivedDataPath build test -only-testing:ParlorTests
build/Build/Products/Debug/parlor-cli spike      # real model + real MCP tool-calling gate
```

UI tests (`-only-testing:ParlorUITests`) need Developer Mode: `sudo DevToolsSecurity -enable` once.

Debug-only env hooks for screenshots: `PARLOR_DEMO=1` (scripted model), `PARLOR_ROUTE=agent:<id>|models|tools`,
`PARLOR_MODE=edit`, `PARLOR_AUTOSEND`, `PARLOR_AUTODOWNLOAD=repo@language,repo@image`.
