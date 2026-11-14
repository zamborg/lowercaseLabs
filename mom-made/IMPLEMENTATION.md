# mom-made — implementation handoff

Read DESIGN.md first. Build a working vertical slice before expanding providers. This document describes proposed contracts, not existing implementation.

## Modules

```text
apps/desktop/          Electron main, trusted renderer, native menus and dialogs
packages/contracts/    IPC, persistence schemas, normalized agent events
packages/coordinator/  workspaces, turn queue, durable operation journal
packages/harnesses/    Codex, Claude Code, Pi adapters
packages/runtime/      process supervisor, toolchain manager, health checks
packages/browser/      main-process view broker, inspection, captures, feedback
packages/mcp/          per-run stdio bridge and scoped tools
packages/cloudflare/   connection checks, resource reconciliation, PublishService
packages/templates/    versioned React/Worker starter and optional D1/R2 recipes
packages/skills/       canonical library, snapshots, harness-specific projection
```

Prefer a small TypeScript monorepo with pinned dependencies. Keep desktop metadata distinct from generated application D1 databases. No production cloud mutations are needed to implement the first three milestones.

## Contracts to agree before implementation

```ts
// Product contracts; provider wire types remain in their adapters.
type RunContext = {
  workspaceId: string;
  targetAppIds: string[];
  conversationId: string;
  turnId: string;
  cwd: string;
  attachmentIds: string[];
  skillSnapshotIds: string[];
  policyId: string;
};

type AgentCapabilities = {
  resume: boolean;
  cancel: boolean;
  compact: boolean;
  imageInput: boolean;
  interactiveApprovals: boolean;
  effortLevels: string[];
};

interface HarnessAdapter {
  probe(): Promise<{ version: string; capabilities: AgentCapabilities }>;
  accountState(): Promise<{ state: 'ready' | 'login-required' | 'error' }>;
  start(context: RunContext): Promise<{ sessionId: string }>;
  resume(sessionId: string, context: RunContext): Promise<void>;
  send(input: AgentInput): Promise<void>;
  interrupt(turnId: string): Promise<void>;
  answer(requestId: string, answer: UserAnswer): Promise<void>;
  compact(sessionId: string): Promise<void>;
  events(): AsyncIterable<AgentEvent>;
  close(): Promise<void>;
}
```

`AgentInput`, `UserAnswer`, and `AgentEvent` are defined during milestone 1. Events need workspace/conversation/turn/session IDs, adapter source ID when available, local monotonic sequence, timestamp, event type, and typed payload. Types include text delta, tool start/update/end, file changes, approval request, question, usage, compaction, and turn completed/failed/interrupted. Persist the event and transcript projection transactionally; coalesce visual deltas without losing durable ordering. Unknown provider events must not crash the UI. Bound payload sizes and handle stdout backpressure and fragmented JSON records.

Main IPC exposes specific methods such as `apps.start`, `browser.capture`, and `publish.prepare`; never arbitrary shell or arbitrary IPC forwarding. Long operations return operation IDs. A publish plan includes an immutable source hash, account/environment, resource mappings, migrations, missing secret names, requested changes, and authorization status. `publish.execute(planId)` rejects stale or mismatched authorization. Retrying the same operation reconciles its journal.

Persistent entities: Workspace, App, Conversation, HarnessSession, Turn, AgentEvent, FileGrant, Attachment, SkillRevision, SkillSelection, Annotation, RuntimeIntent, CloudConnection, ResourceMapping, PublishPlan, Deployment. Secrets are opaque Keychain references. Use schema versions and transactional metadata migrations; back up before upgrades.

## Delivery sequence

### 0 — prove the difficult boundaries

Make disposable technical spikes before styling the production shell:

- Stream one Codex turn into a custom UI; handle one approval, interruption, and session resume using the installed protocol version.
- Load a local app in WebContentsView, capture it, inspect it, and perform a real interaction in that same browser session.
- Hide the native preview, annotate its frozen capture, then restore it without losing page state.
- Start a Worker template with local D1, write a row, terminate its full process tree, restart, and read the row.
- Force-kill the desktop parent and prove its managed server descendants exit.

Exit: known protocol versions and a written result for each spike. If a browser API cannot support the needed behavior, resolve that before building the complete shell. Do not assume a separate Playwright browser is equivalent.

### 1 — usable local studio

Implement workspace metadata, template creation, supervisor, preview tabs, single conversation, and Codex adapter. Include empty, starting, failed, and interrupted states. Model/effort options come from capabilities. Create and preserve ordinary app folders. Add native menu shortcuts and basic tray lifecycle.

Exit: a user asks for a small app, uses it, asks for an edit, opens another app, quits, and reopens both with local data intact. Run targets stay fixed when preview selection changes. Fresh installation works without global Node or terminal PATH assumptions.

### 2 — the visual feedback loop

Implement the owned MCP bridge, browser control, captures, element selection, and frozen-frame markup. Connect feedback to submitted turns and post-change evidence. Add private attachments and explicit asset import. Handle stale navigation references and human takeover.

Exit: circle a form button, request a change, receive an updated preview and evidence, then resolve or reopen the comment. Verify drawing never clicks the underlying app.

### 3 — durable state and reusable taste

Add D1/R2 template capabilities, migration management, Skills page, revision snapshots, and per-app secret entry. Store Keychain references and keep secrets out of chat, screenshots, archives, logs, and frontend bundles. Show read/write file grants clearly.

Exit: a skill changes design behavior on the next turn; a server-side API call uses a local secret without including it in the client build; a private reference stays out of exported source.

### 4 — reliable publishing

Implement Cloudflare settings, account/resource checks, immutable build staging, resource journal, secret upload, migrations, publish, verification, and history. Mount documentation MCP and optionally authorize the API MCP. Add saved destination and workers.dev first; custom hostname second. UI and agent invoke the same service.

Exit: deploy a static app, an app with D1, and one using a secret to a designated test account. Repeat publish without duplicating resources. Interrupt and recover each deployment phase. Verify production remains independent of local data and local shutdown. Record cleanup instructions and remove only explicitly identified test resources when authorized.

### 5 — bring another harness

Add Claude Code and Pi using the same adapter contract and replay fixtures. Keep provider authentication modes distinct. Verify each supports image/feedback context and approval delivery before advertising full support; capability-gate missing controls. Switching harness generates a visible handoff, not a pretend resume.

Exit: the same build/inspect/change flow works with all supported providers. Failure to authenticate has a usable recovery path and never silently falls back to another billable provider.

### 6 — Mac release quality

Add signed/notarized packaging, managed toolchain installation/update recovery, accessibility pass, keyboard coverage, dark appearance, crash diagnostics, bounded logs, and performance measurement. Fresh-account test on supported Apple Silicon/macOS versions; decide Intel support separately. Use native file dialogs and real menus, not browser imitations.

Suggested performance budgets are acceptance targets to measure, not claims: shell interactive within 2 seconds on the reference machine with warm disk; cached preview ready within 5 seconds; do not block the UI on server startup; no duplicate runtimes after reopen; bounded idle/background CPU. Record memory with 1, 3, and 8 apps and choose a disclosed policy if all-open restoration is too expensive.

## Required regression scenarios

| Scenario | Pass condition |
| --- | --- |
| Quit/reopen after a database write | Row, source, tab order, and chat remain; no old server survives |
| Busy port / server crash / sleep-wake | Clear state and retry; no unrelated process killed or duplicate spawned |
| App switched during agent turn | Edits and tools remain on submitted targets |
| Main or coordinator crash | Durable events recover; interrupted turns are not reported successful or automatically resumed |
| Preview tries privileged IPC | Rejected; no credential/file/system API becomes accessible |
| Attachment symlink escapes grant | Rejected by canonical-path policy |
| Markup followed by HMR | Original capture remains valid; fresh evidence linked to new revision |
| Human touches page during agent action | Agent control lease yields; no stale follow-on clicks |
| Cloud credentials expire | Local app continues working; publish reconnects explicitly |
| Publish interrupted after creating D1 | Retry discovers same recorded resource; no duplicate database |
| Code rollback after migration | UI reports data is not rolled back and checks compatibility |
| Agent edits during publish | Deployment uses recorded immutable snapshot |
| Provider changes schema or emits malformed JSON | Adapter fails clearly without corrupting transcript |
| Secret entered or tool errors | Value absent from transcript, diagnostics, frontend output, and export |

Test contracts and state transitions with deterministic fixtures. Use real integration tests for process-tree ownership, browser isolation, local persistence, and publishing; mock-only coverage is insufficient for those boundaries. Do not add a broad test suite for decorative spacing.

## Initial scope boundary

Ship the entire local create/edit/inspect/reopen loop plus one dependable publish path before adding framework choices, autonomous swarms, public extension marketplaces, collaboration, cloud-hosted agents, or general infrastructure management. Preserve adapters and module boundaries so those can be added without rewriting the studio.

Implementation kickoff: “Implement milestones 0 and 1 from mom-made/DESIGN.md and IMPLEMENTATION.md. Treat the documented contracts as proposed until checked against installed versions. Preserve unrelated workspace changes. Do not deploy cloud resources during the local milestones. Report spike results and a runnable local vertical slice.”
