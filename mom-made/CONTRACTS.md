# Local contracts — v1

Implemented wire validation lives in `src/core/contracts.ts`; proposed MCP inputs and result envelopes live in `src/core/tools.ts`. The MCP stdio server belongs to milestone 2 and is not mounted in this slice.

## Ownership

- The desktop owns one default workspace. Each app gets a UUID and an ordinary folder under `workspaces/default/apps/<uuid>` in the selected data directory.
- `mom-made.json` is a versioned, secret-free manifest. The absolute path is held in desktop metadata, not the manifest.
- Codex runs with that app folder as `cwd`, workspace-write sandbox, on-request approvals, and explicit app instructions. The visible workspace transcript combines messages; each app has its own provider thread. Cross-app context summaries remain future work.
- Submitted turns capture the target AppRecord before asynchronous connection/start work. Preview selection is independent of the turn target. One editing turn runs at a time.
- SQLite records transcript projections and message events in the same transaction. App registry and selected tab persist separately. An interrupted active turn is surfaced on recovery; no agent work resumes automatically.
- Local D1 state is below the app's `.wrangler/state`; closing a tab stops its runtime and preserves that folder.

## Desktop IPC

Only the trusted top-level shell frame can call `state` or `command`. Commands are a discriminated union; no arbitrary shell execution or provider RPC forwarding is exposed. Preview pages have no preload bridge and no Node integration.

Creation returns an app ID. Runtime startup is asynchronous and progresses through starting, ready, failed, stopping, and stopped. Errors reach the shell with logs and a retry path. A readiness check hits `/api/health`. Bounds are validated integer coordinates supplied by the trusted layout renderer. Capture returns a saved PNG path, an in-memory data URL for display, and the navigation generation.

## MCP contracts

Use explicit app IDs on every app-scoped request. Browser observation/action requests include a navigation generation. A stale generation returns `STALE_NAVIGATION`, with retryable=true; the caller observes again rather than replaying a click against a different page. `apps.list` is workspace-scoped. Inputs reject unknown fields.

Results are versioned success/error envelopes. Errors have a stable code, human-readable message, and retryable flag. Future mutating tools must include an operation ID and return the recorded result for a retried operation. Publish authorization binds account, environment, source hash, and plan ID. It is not inferred from a generic MCP connection.

The first tools will be apps.list, apps.describe, apps.logs, browser.capture, and browser.inspect. Click/type tools will require a view control lease and fresh element references. Files, secrets, and publish tools must be implemented behind their own services before being offered.

## Harness protocol

Codex 0.166.0 was checked using its generated JSON schema. The adapter uses LF-delimited records, request correlation, bounded record size, connection-scoped callbacks, timeouts, initialize/initialized handshake, and separate server requests and notifications. Its environment is limited to local process essentials; cloud/API tokens from the desktop environment are not passed through. Authentication uses Codex's own local sign-in.

Unknown server requests receive an explicit unsupported response. Command and file-change approval requests receive native product UI. The current approval UI supports allow once/decline; richer question/permission forms are future work. CLI schema upgrades need fresh fixtures and integration checks.
