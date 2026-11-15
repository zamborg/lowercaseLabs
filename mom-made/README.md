# mom-made

A little Mac studio for making useful web apps. Talk on the left. Use what you made on the right.

The first local implementation is runnable: independent React/Cloudflare Worker apps, persistent D1, isolated Chromium previews, Codex chat, native menus, and a menu bar item.

## Run

From this directory, with Node 22+ for dependency installation:

```sh
npm ci
npm start
```

Use **Chat** to ask questions or plan. Choose **New App** when you're ready to build. When an app is open, drag the divider to resize its chat pane. Replies support Markdown. Enter sends, and Shift-Enter adds a line.

App servers use Electron's bundled Node. The source checkout requires Node and npm to install dependencies and build the app. The Apple Silicon preview DMG includes the app runtime. Install and sign in to a supported assistant separately, then select it in **Settings → Assistants**.

Closing the window stops local servers and agent work. The menu bar item reopens the workspace. Quit exits everything. On reopen, open tabs and local databases return; interrupted agent turns do not restart automatically. Open **Apps** in the Studio toolbar to find or reopen any project, including closed tabs.

The default data directory is `~/Library/Application Support/mom-made`. Set `MOM_MADE_DATA` to use a separate test workspace. Each app has its own source folder and `.wrangler/state`; Files reveals it in Finder.

## Verify

```sh
npm run typecheck
npm test
npm run spike
npm run spike:codex
```

`spike` opens a temporary Electron window and exercises real local Worker/D1 and browser operations. `spike:codex` makes small real model requests using your Codex login, including a declined approval and interruption. Both write results under ignored `.spike/`. Tests verify persistence, filesystem identity, IPC validation, stale navigation handling, transport reconnect, and process-tree cleanup after SIGKILL.

## What v1 does

Choose **New App** and describe it in plain words, watch it appear, use it, then circle what you want changed ("Show me what to change") and the marked-up picture goes to the agent. The agent can look at the running app through a private MCP tool set (screenshot, page text, console, logs, click, type). **Publish** deploys a snapshot to Cloudflare workers.dev with your own Wrangler login: it creates the D1 database once, applies migrations, deploys, and checks the live URL. Local D1 schema changes are checked-in migrations, applied before each start.

Every assistant gets the bundled [web design skill](skills/web-design/SKILL.md) for new and existing projects. It covers layout, phone-friendly design, accessible controls, and checking the running preview.

## Settings

Open **Settings** in the Studio toolbar or press Cmd-comma.

- **Assistants** shows installation and connection status. Choose the active assistant and save a default model and reasoning level for each one. Chat options can override the defaults until the active assistant or its saved preference changes.
- **Cloudflare** checks the connection and opens Cloudflare's own sign-in flow. Publishing uses the existing Wrangler connection.
- **Admin** lists local apps and their recent deployment history. Open an app in Studio, find its source files, or open its live site.

Each assistant implements the same interface in `src/core/agents/`.

- **Codex** via `codex app-server` (verified live; supports approvals, compact).
- **Claude Code** via the official CLI's `stream-json` mode and its own sign-in. mom-made does not display Claude Code approval prompts. Tested with a fake CLI.
- **Pi + OpenRouter** via Pi's RPC mode. Electron `safeStorage` encrypts the saved key. The desktop checks the key with OpenRouter and supplies it to Pi for model requests. Tested with a fake RPC process and the real Pi model catalog. [Pi installation](https://pi.dev/docs/latest/quickstart) · [OpenRouter key check](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key)
- **Pi + Local model** connects Pi to a running OpenAI-compatible model server on this Mac. It keeps its own sessions and can read page text and use the preview tools. Visual markup requires an assistant with image support.

Use `MOM_MADE_CODEX`, `MOM_MADE_CLAUDE`, or `MOM_MADE_PI` to point to a specific binary. To add another assistant, implement `Agent` in `src/core/agents/types.ts` and register it in `src/desktop/main.ts`.

## Package for macOS

`npm run dist:dmg` creates an Apple Silicon developer build for local testing. macOS may block a browser download of an ad hoc build with an “app is damaged” alert.

For a shareable DMG, install a [Developer ID Application certificate](https://developer.apple.com/help/account/certificates/create-developer-id-certificates/) with its private key in Keychain, then save [notarization credentials](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow) in a Keychain profile named `mom-made` using `xcrun notarytool store-credentials mom-made`.

```sh
APPLE_KEYCHAIN_PROFILE=mom-made npm run dist:release
```

Run this on macOS 14 or later. The command signs and notarizes the app and disk image, then writes the verified DMG to `release/distribution/`. Before sharing it, download the candidate in a browser and open it from Applications on a clean Mac or macOS user account. This exercises Gatekeeper with a browser-downloaded copy.

### Flint on Apple Silicon

Recursive provides [Flint INT4 Q4_1](https://huggingface.co/RecursiveSI/flint-int4-gguf) for Apple Silicon with Metal. The weights are about 26 GB and the publisher specifies at least 48 GB of unified memory. Its [NVFP4 build](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf) targets NVIDIA CUDA.

Install [Pi](https://pi.dev/docs/latest/quickstart) and a [llama.cpp release with Flint support](https://github.com/ggml-org/llama.cpp/releases). In **Settings → Assistants → Pi + Local model**, expand **Run Flint on Apple Silicon** for a command that downloads the weights separately and starts a local server. Once the server finishes loading, choose **Save and check**, then **Use assistant**. The default setup uses `http://127.0.0.1:8080/v1`, model ID `flint`, and a 65,536-token context. If you change the server's context limit, set the matching value in Settings.

## Not yet verified or done

- Set `MOM_MADE_DRY_RUN=1` to build and check a deployment without publishing it. Pi with local Flint was verified live. Live turns with Claude Code and Pi + OpenRouter still need verification.
- Skills management, secret entry, file attachments, verification of the Developer ID release build, and per-project dependency management are pending. Projects share the node modules bundled with mom-made.
- This is a trusted local developer tool, not a sandbox.

- [Design](DESIGN.md)
- [Implementation plan](IMPLEMENTATION.md)
- [Implemented contracts and MCP boundary](CONTRACTS.md)
- [Validation results and remaining work](VALIDATION.md)
