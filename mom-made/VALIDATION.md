# Local slice validation

Executed on macOS / Apple Silicon, November 13, 2026. Dependency versions are pinned in package-lock.json.

## Tested versions

- Electron 45.3.2, bundled Node 24.22.2
- Codex CLI 0.166.0; JSON schema generated from the installed binary
- Vite 8.3.8, Cloudflare Vite plugin 1.69.1, Wrangler 4.162.0
- React 19.3.1, Hono 4.13.18

## Results

- TypeScript check and production desktop build passed.
- Six regression tests passed, including actual child/grandchild termination after the owner received SIGKILL.
- Real local Worker health check passed.
- A D1 row survived complete local server shutdown and restart.
- The actual WebContentsView was inspected and accepted native text input. Node `require` was unavailable inside the preview.
- Preview capture, hidden live view, drawing in a shell canvas, and live-view restoration passed without losing input state.
- Codex authenticated through its existing local login, streamed a completed reply, reconnected and resumed its thread, emitted an approval request that was declined, and completed an explicit interruption with status `interrupted`.
- The desktop shell loaded with the privileged preload API and displayed the live model catalog and app tab.
- Codex edited the first app while the selected preview was the second app. Inspection after HMR confirmed the change in the first app and no matching change in the second.
- After quitting and launching the desktop again: both app tabs, seven transcript messages, selection of the second app, first-app source changes, and a saved D1 row were restored.

Machine-readable raw results are in ignored `.spike/results.json`, `.spike/codex.json`, and `.spike/reopen.json`. Temporary test app folders are kept below `.spike/`, separate from the normal workspace.

## Issues found and fixed

- Build output initially retained source subdirectories; switched to explicit named entry points.
- Old Codex process exit callbacks could invalidate a newly opened connection; callbacks are now scoped to their child process. Reconnect is covered by a regression test.
- Multiple callers could initiate concurrent preview loads for the same app; startup now deduplicates the entire server-plus-preview operation.
- Capturing a fully hidden Chromium surface can fail. Capture tools must target an available display surface; the local UI captures the visible selected app. Parent WebContents captures do not include native child preview views.
- The desktop now has its own userData/single-instance namespace.
- Private build/deploy credentials are excluded from child process environments.

## Limits and outstanding acceptance work

This is a developer build. No cloud resource was created or deployed. No signing/notarization, standalone installer, managed per-project dependency install, broad accessibility audit, memory budgets, or eight-app resource test has been completed.

Codex basic chat, approvals, stop, resume, and compaction are wired. Rich question/permission forms are not yet supported; unknown requests receive an explicit unsupported response. Third-party auth and expired-session recovery need additional product testing.

The preview security check demonstrates no Node access in the page; it is not an exhaustive Electron security audit. Coding-agent OS isolation, remote bindings, cloud publishing, secrets, arbitrary external file grants, and MCP authorization are separate work.

Full window-close/reopen stress, crash-loop recovery, busy-port races, and sleep/wake need broader regression coverage before release. The local server watchdog was tested for owner death; Codex subprocess tree hardening remains separate from the server watchdog.
