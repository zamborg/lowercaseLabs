# mom-made

A little Mac studio for making useful web apps. Talk on the left. Use what you made on the right.

The first local implementation is runnable: independent React/Cloudflare Worker apps, persistent D1, isolated Chromium previews, Codex chat, native menus, and a menu bar item.

## Run

From this directory, with Node 22+ for dependency installation:

```sh
npm ci
npm start
```

Create an app, then describe what you want in the chat. Cmd-Enter sends. Model and reasoning choices come from your installed Codex. Use Sign in if needed; the app uses Codex's own authentication. `MOM_MADE_CODEX=/absolute/path/to/codex` overrides binary discovery.

App servers use Electron's bundled Node rather than a global Node executable. The developer launcher still requires npm. This is a source checkout, not yet a signed standalone distribution.

Closing the window stops local servers and agent work. The menu bar item reopens the workspace. Quit exits everything. On reopen, open tabs and local databases return; interrupted agent turns do not restart automatically. Closing a tab preserves the app in the library on the empty studio screen.

The default data directory is `~/Library/Application Support/mom-made`. Set `MOM_MADE_DATA` to use a separate test workspace. Each app has its own source folder and `.wrangler/state`; Files reveals it in Finder.

## Verify

```sh
npm run typecheck
npm test
npm run spike
npm run spike:codex
```

`spike` opens a temporary Electron window and exercises real local Worker/D1 and browser operations. `spike:codex` makes small real model requests using your Codex login, including a declined approval and interruption. Both write results under ignored `.spike/`. Tests verify persistence, filesystem identity, IPC validation, stale navigation handling, transport reconnect, and process-tree cleanup after SIGKILL.

## Current boundary

This is the milestones 0–1 local vertical slice. Full markup feedback, mounted MCP tools, Skills management, secure secret entry, publishing, Claude Code, and Pi remain subsequent milestones. Capture currently freezes a saved screenshot; it does not yet send visual feedback to Codex.

The coordinator currently runs in Electron main, with app runtimes isolated behind watchdog processes. Moving orchestration to a utility process remains a later hardening step. The UI presents one workspace transcript; Codex threads are stored per app. Cross-app conversation summaries are not implemented.

Generated apps currently share this checkout's pinned dependency cache through a `node_modules` symlink. Their package manifests contain exact dependency versions; to use an app independently, remove that symlink and install dependencies in its folder. Managed per-project dependencies and a packaged toolchain installer remain release work.

This is a trusted local development tool: agents and generated tooling can execute local code. A working directory is not an OS sandbox. Cloud credentials from the parent environment are excluded from managed runtime and Codex environments.

- [Design](DESIGN.md)
- [Implementation plan](IMPLEMENTATION.md)
- [Implemented contracts and MCP boundary](CONTRACTS.md)
- [Validation results and remaining work](VALIDATION.md)
