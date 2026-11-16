# parlor

Local, native Mac chat app for persistent agents. Personal assistants to roleplay characters, one chat per agent,
fully on-device: MLX language models, a local image model, and MCP tools.

Contracts: `SPEC.md`. Plan: `PLAN-V3.md` (`PLAN-V2.md` is superseded). Mockups: `mockups/index.html`.

## Build and run

Requires macOS 26, Xcode 26, `xcodegen` (`brew install xcodegen`).

```bash
scripts/build-worker.sh          # image worker (own package; first build compiles MLX, takes minutes)
xcodegen generate
xcodebuild -project Parlor.xcodeproj -scheme Parlor -configuration Debug -destination 'platform=macOS' -derivedDataPath build build
open build/Build/Products/Debug/Parlor.app
```

First launch creates `~/parlor/` with three example agents (Mabel, Pip and Builder, which makes more agents). Open **Library** and
download a language model (Qwen3 4B is a good start) and, for images, Z-Image Turbo. Each installed model gets a host on its own.

## Layout

Three layers, each made of JSON nouns and services (see `SPEC.md`):

| Layer | What | Where |
|---|---|---|
| Runplane | engines (backends), hosts (an engine plus a model), the catalog, the gateway, the supervisor | `Packages/ParlorCore/Sources/ParlorCore/Runplane` |
| Agents | harnesses (how an agent thinks), agents, rooms, conversations | `.../Agents`, `Agent.swift`, `Room.swift`, `Conversation.swift` |
| Control | verbs, the `parlor` MCP server, tiers and tokens | `.../Control`, `Commands*.swift` |

| Path | What |
|---|---|
| `Packages/ParlorCore` | All logic, no UI |
| `App/` | SwiftUI app: chat, agent editor, Library (models, hosts, engines), Tuning, System |
| `CLI/` | `parlor-cli`: drives the same engine headless (`verbs`, `call`, `chat`, `bench`, `spike`, …) |
| `ImageWorker/` | `parlor-imagegen` helper (Z-Image via MLX), separate package so its MLX version is independent |
| `Tests/` | Unit and flow tests (scripted model and a fake OpenAI server, no weights) |
| `UITests/` | XCUITest flow (needs Developer Mode, see below) |

## State on disk (`~/parlor`, override with `PARLOR_HOME`)

```
CLAUDE.md                     explains the folder to external agents (written on first run)
settings.json                 machine settings, tokens (hashes), gateway
mcp.json                      mounted MCP servers (Claude Desktop format)
engines/<id>/engine.json      backends; drop a folder in to add one
hosts/<id>.json               an engine plus a model; installed models get one automatically
harnesses/<id>/harness.json   how an agent thinks, beside its Python program
agents/<id>/agent.json        persona.md is the system prompt, memory.md what it remembers
chats/<id>/log.jsonl          the conversation; summary.md its rolling summary
media/YYYY-MM/                generated images
models/<org>/<name>/          weights (PARLOR_MODELS overrides the location)
```

Everything is a file you can edit; the app reloads live. One gateway (`http://127.0.0.1:<port>`) serves `/v1` (OpenAI-shaped
chat and images) and `/mcp` (the `parlor.*` verbs) to the app's own chats, harnesses and outside clients alike.

## Extending

- A new backend is an `engine.json` folder. A new way for an agent to think is a `harness.json` plus a program speaking the
  harness protocol (`SPEC.md` section 4; see `harnesses/CLAUDE.md` in your folder).
- The **Builder** agent (or any token with the configure tier) makes agents with `parlor.agents.validate`, `agents.try` and
  `agents.create`; hosts have `hosts.test` and `hosts.bench`.

## Testing

```bash
xcodebuild -project Parlor.xcodeproj -scheme Parlor -destination 'platform=macOS' -derivedDataPath build test -only-testing:ParlorTests
build/Build/Products/Debug/parlor-cli bench      # real model through the gateway: prompt-cache pattern across turns
```

UI tests (`-only-testing:ParlorUITests`) need Developer Mode: `sudo DevToolsSecurity -enable` once.

Debug-only env hooks for screenshots: `PARLOR_DEMO=1` (scripted model), `PARLOR_ROUTE=agent:<id>|models|tools|library|tuning`,
`PARLOR_MODE=edit`, `PARLOR_AUTOSEND`, `PARLOR_AUTODOWNLOAD=repo@language,repo@image`.

## Character chat

The **Character (scene harness)** template runs an agent through `harnesses/character/`: it plays the agent as a person
(not an assistant), keeps their look in `appearance:` and the story in `scenario:`, and has a small "art director" step decide
when a picture is worth taking and write the image prompt, restating the appearance each time so the face stays consistent.
Pictures come when the user asks to see something, or every few replies (`images: auto|ask|off`, `image_every:`). Any
GGUF model works with it: download a GGUF repo from the Library (needs `llama-server`); parlor makes a host for it. Sampler
flags live in the host's `settings`/`defaults` (the Tuning screen edits them and benches the result).

### Stable Diffusion image models

Besides Z-Image, the Library (Models, Image tab) downloads a single-file Stable Diffusion checkpoint (SDXL, Illustrious, Pony and
similar) from a repo such as `Ine007/waiNSFWIllustrious_v140`, and draws with it through stable-diffusion.cpp's `sd-cli`
(bundled in the app, fetched by `scripts/fetch-sd.sh`; or put `sd-cli` in `~/.local/bin`). One `sd-cli` run per image, so nothing stays in
memory. Sampling is the host's `settings` (`parlor-image.json` beside the weights holds the checkpoint's sampler, steps, size and prompt tags): V-prediction checkpoints (a name with `vpred` or `v-pred`) get
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

That signs with the hardened runtime, notarizes the dmg and staples the ticket. Models are not bundled: a fresh install starts with an empty parlor folder at `~/parlor` and downloads models from the Library.
