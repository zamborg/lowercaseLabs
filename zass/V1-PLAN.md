# zass V1 planning handoff

Updated: 2026-11-17 (America/Los_Angeles).

## Goal and planning status

Zubin wants a personal AI tools and services layer that he can test from
Claude Code, Codex, and a locally run Pi agent using OpenRouter. Ideally a
hosted web app lets users inspect their content and manage controls.

Opus and Zubin will work through high-level service boundaries before V1
implementation. This document captures the repository spot-check and
proposed scope; recommendations below are not settled architecture decisions.
No hosting platform, storage migration, account model, or deployment has
been selected or authorized by this plan.

## Current implementation

zass is a working local prototype, not yet the complete V1 described above.

- `mcp/zass/zass_mcp/server.py`: Python FastMCP server, stdio transport,
  23 tools, primarily returning Markdown strings.
- `mcp/zass/zass_mcp/store.py`: Markdown/YAML storage for todos, notes,
  people, memory, journal, and inbox. `ZASS_STATE_DIR` overrides the default
  repository-local `state/` directory.
- `mcp/zass/zass_mcp/apple.py`: local macOS Contacts search and iMessage
  sending through AppleScript, with arguments passed through argv.
- `CLAUDE.md` and `AGENTS.md`: operating rules, continuity practices, and
  external-action confirmation instructions.
- `.claude/skills/`: brief, capture, triage, and weekly-review workflows.
- `.mcp.json`: registers the internal zass server for Claude Code.
  Codex configuration is documented in `AGENTS.md`.
- `connectors/README.md`: optional Gmail, Calendar, and iMessage-history
  setup recipes. These connectors are not registered in `.mcp.json`.
- `scripts/setup.sh`: dependency sync, store tests, and server import check.

The repository currently has no zass web frontend, authenticated service
API, user-account isolation, or hosted deployment implementation.

## Evidence from the spot-check

### Verified

- A real MCP stdio client initialized the server and discovered 23 tools.
- `todo_add`, `todo_list`, `note_add`, `search`, and `briefing` returned
  successfully against a temporary state directory.
- The existing store suite reported **9 passed, 1 failed**. The failure in
  `test_todo_lifecycle` assumes the fixed due date `2026-08-01` is in the
  future. It now falls in the past, so the zero-day filter includes it.
- A disposable fixture demonstrated that
  `memory_forget("../outside")` can delete a file outside the memory
  collection. `find_by_id` joins an unvalidated ID directly into a path.
  The test affected temporary fixtures only, not personal state.

### Inspected, but not validated end-to-end

- Claude Code and Codex configuration paths exist; actual agent sessions
  have not been tested against zass in this review.
- macOS integrations exist; no Contacts query or message send was performed.
- The neighboring `mom-made/src/core/agents/pi-mcp.ts` registers an MCP
  server through a Pi extension. Its Pi adapter also has OpenRouter
  integration. This is a candidate reference, not proof that zass works
  with the intended Pi installation or version.
- No external Gmail/Calendar connection was tested.
- No concurrency, authentication, deployment, or multi-user tests exist in
  the inspected zass suite. The smoke check is not an exhaustive tool test.

## Gaps to address before release

1. **Path containment.** Validate IDs and collection names; enforce resolved
   paths staying within their intended roots. Review every read, update,
   delete, and search route, including symlink behavior. Add regression tests.
2. **Concurrent writes.** Current writes are direct, with no locking or
   version checks around read/modify/write. Multiple clients can lose updates.
   Choose a single write authority and an explicit conflict strategy.
3. **Enforced permissions and approvals.** Outgoing-action confirmation is
   currently a prompt-level rule. `imessage_send` sends immediately when
   called. Design service-enforced grants and approvals for external actions;
   record execution outcomes automatically rather than relying only on an
   agent to call `journal_log` afterward.
4. **Portable agent behavior.** MCP registration alone does not establish
   that every client loads the operating manual and playbooks. Specify what
   belongs in tool contracts, server instructions, and client setup.
5. **Stable validation.** Make date tests deterministic, validate input
   schemas and error behavior, and cover the MCP boundary in addition to
   store functions.
6. **Web/service foundations.** Define authentication, isolation, content
   APIs, access revocation, secrets handling, backup/export, and deployment
   before exposing a hosted service.

## Proposed V1 scope for discussion

Start with one private personal workspace used by all three clients, then
expand to multiple users after that workflow is reliable.

Include:

- Shared todos, notes, people, memory, inbox, journal, and search.
- Repeatable setup and smoke scenarios for Claude Code, Codex, and Pi with
  OpenRouter.
- A web interface for browsing/editing content and viewing activity.
- Explicit client access grants and revocation, with defined action controls.
- A supported backup/export and restore path.

Defer unless the planning discussion identifies a compelling V1 need:

- Broad connector catalog, autonomous outbound actions, public signup,
  billing, team collaboration, and complex background scheduling.

Decide separately whether macOS messaging, Gmail, and Calendar are required
for the first private release. Existing connector code is not itself a
reason to include an unvalidated capability in V1.

## Proposed service boundaries

- **Agent clients own reasoning and conversation.** zass provides durable
  personal state and authorized operations across interchangeable clients.
- **One application layer owns mutations.** MCP and the web UI use the same
  validation, authorization, persistence, and audit behavior.
- **Local capabilities execute locally.** Hosting the web app does not give
  a server access to macOS Contacts or Messages. A local bridge, if included,
  needs explicit pairing, capability grants, and offline behavior.
- **Content and execution history have distinct purposes.** The journal can
  preserve assistant continuity; an execution audit should reliably identify
  the client, operation, approval where applicable, and outcome.

## Decisions for Zubin and Opus

1. **Authoritative state:** local-first with a hosted interface, or hosted
   state with optional local connectors? What works while offline?
2. **Release audience:** private single-user V1, invite-only individual
   accounts, or public multi-user service?
3. **Storage contract:** retain Markdown as the writable source of truth,
   or use a transactional store with Markdown import/export? How do direct
   file edits interact with permissions and conflict detection?
4. **Client transport:** local stdio, remote MCP, or both? How do credentials,
   workspace selection, and client revocation work?
5. **Action boundary:** which tools are read-only, content-mutating, or
   externally consequential? Where are approvals presented and enforced?
6. **Web controls:** exactly which controls ship—client grants, connector
   configuration, approval queue, audit history, export, deletion?
7. **Connector ownership:** does zass broker external services or do agents
   connect independently? Independent connectors bypass zass's controls.
8. **Workflow delivery:** what minimum context and playbooks must each client
   load, and how are those kept consistent across runtimes?

## Proposed implementation sequence and acceptance checks

### 1. Harden the local core

Fix path containment, deterministic date tests, input validation, and the
chosen write-conflict strategy. Tests should demonstrate rejected escapes,
preserved data on failures, and defined concurrent-update behavior.

### 2. Prove the three-client workflow

From each actual client: connect, capture an item, create/update a todo,
save/retrieve a note and memory, search, and obtain a briefing. A second
client must observe persisted changes. Keep external sends disabled or
mocked during routine validation. Record client versions and setup steps.

### 3. Build the authenticated web experience

Use the same application layer as MCP. Verify content visibility and edits,
access revocation, and activity history. If V1 has multiple users, test
cross-user access denial through both web and MCP interfaces before release.

### 4. Add selected external capabilities

Validate connector setup, permission denial, unavailable local bridge,
approval binding to exact action arguments, and failure/retry behavior.
Avoid duplicate external effects on retries; log actual execution outcomes.

### Release gate

A fresh setup can run the supported clients and web app; data survives a
restart and a backup/restore exercise; access controls behave as documented;
all selected connectors have explicit validation results. Document known
limitations rather than treating the initial stdio smoke check as completion.

## Next planning output

Resolve the decisions above into an agreed architecture, V1/non-V1 scope,
data and permission model, and a short milestone backlog. Then implement
against that agreed boundary. This handoff does not prescribe a framework
or hosting vendor.
