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

## Character chat

The **Character (scene harness)** template runs an agent through `harnesses/character.py`: it plays the agent as a person
(not an assistant), keeps their look in `appearance:` and the story in `scenario:`, and has a small "art director" step decide
when a picture is worth taking and write the image prompt, restating the appearance each time so the face stays consistent.
Pictures come when the user asks to see something, or every few replies (`images: auto|ask|off`, `image_every:`). Any
GGUF model works with it: download a GGUF repo from the Models screen (needs `llama-server`), then add sampler flags to the
provider's `service` command in `settings.json` (for example `--temp 1.17 --min-p 0.075 --top-k 50 --repeat-penalty 1.1`).

### Stable Diffusion image models

Besides Z-Image, the Models screen (Image tab) downloads a single-file Stable Diffusion checkpoint (SDXL, Illustrious, Pony and
similar) from a repo such as `Ine007/waiNSFWIllustrious_v140`, and draws with it through stable-diffusion.cpp's `sd-cli`
(bundled in the app, fetched by `scripts/fetch-sd.sh`; or put `sd-cli` in `~/.local/bin`). One `sd-cli` run per image, so nothing stays in
memory. Sampling lives in `parlor-image.json` beside the weights: V-prediction checkpoints (a name with `vpred` or `v-pred`) get
`prediction: v`; defaults are Euler a, 28 steps (20 for `detail: fast`), CFG 6, CLIP skip 2, 832×1216 portrait, with quality
tags added to the prompt and a negative prompt. Set an agent's `image_model:` to the repo id. The character harness writes
Danbooru-style tags for these models automatically. Gated repos need `hf auth login` and access granted on the model page.

## Packaging

```bash
scripts/package.sh
```

Builds a Release `Parlor.app` with the image worker inside, `parlor-cli`, and `dist/Parlor-<version>.dmg` (the version is `MARKETING_VERSION` in `project.yml`). Without options it is ad-hoc signed: it runs on the Mac that built it, and on another Mac the first launch needs right-click → Open (or System Settings → Privacy & Security → Open Anyway), because it is not notarized.

To distribute without that prompt you need a paid Apple Developer account:

```bash
xcrun notarytool store-credentials parlor          # once: Apple ID, team ID, app-specific password
SIGN_ID="Developer ID Application: Your Name (TEAMID)" NOTARY_PROFILE=parlor scripts/package.sh
```

That signs with the hardened runtime, notarizes the dmg and staples the ticket. Models are not bundled: a fresh install starts with an empty parlor folder at `~/parlor` and downloads models from the Models screen.
