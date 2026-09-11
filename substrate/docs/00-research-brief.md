# Research Brief — Personal AI Infrastructure

Status: input to Phase 1 (product scoping). Not a design doc.
Date: 2026-09-11

---

## 0. The thesis this research supports

Everyone in this space is building the **harness** — gateway, channels, skills,
memory, cron. Two projects (OpenClaw, Hermes) now define that shape, both have
enormous mindshare, and both are converging on the same primitives. The harness
is therefore not defensible and not interesting: it will be commoditized by
whoever ships the next agent framework.

What nobody owns is the layer *underneath*: a single-user, authenticated,
permissioned, auditable, **writable** view of one person's life, with a stable
contract that outlives any agent framework. That is the thing worth building,
and it matches the stated goal — defer agent choices to downstream consumers.

The rest of this doc is the evidence.

---

## 1. Prior art: what the field has converged on

### OpenClaw
Self-hosted gateway. Primitives: **Gateway** (control plane: sessions, routing,
policy), **Channels** (WhatsApp / Telegram / Discord / Signal / iMessage /
Slack / WebChat), **Sessions** (DMs share the main session, each group gets its
own), **Nodes** (paired iOS/Android devices contributing camera, screen, voice),
**Skills / plugins** (channel, provider, tool). ~235K stars.

Then, February 2026: three critical advisories in two weeks. RCE via prompt
injection (CVE-2026-25253). A passive scan found 42,665 exposed instances; of
5,194 verified, **93.4% had auth bypass**. The skill marketplace had 341
malicious skills out of 2,857 reviewed — including the single most-downloaded
one. Banned by major Korean tech companies. Karpathy: "400K lines of vibe coded
monster."

**Read:** the architecture caused the crisis. Channel-first design + an
unsandboxed agent with ambient credentials + an open plugin marketplace is a
structural vulnerability, not a bug that got patched. Any personal
infrastructure layer has to invert every one of those choices.

### Hermes Agent (Nous Research, MIT, Feb 2026)
Primitives are **state-shaped**, which is the more interesting half:
- **Profiles** — independent agents, each with own config, gateway, cron.
- **SOUL.md** — identity/voice document.
- **Memory** — SQLite + FTS5 "holographic" fact store; pluggable providers
  (honcho, mem0, supermemory, byterover, hindsight, openviking, retaindb).
- **Skills** — procedural memory the agent distills from completed tasks.
- **Cron** — built in, with delivery to any channel.
- **Terminal backends** — local, Docker, SSH, Modal, Vercel Sandbox.

~32K stars by April 2026, 600+ indexed skills, migration from OpenClaw.

**Read:** eight competing memory providers behind one plug-point means there is
*no standard* for personal memory. That's a gap, and it's adjacent to ours.

### Daniel Miessler's "Personal AI Infrastructure"
Seven components: Intelligence (model + scaffolding), Context (3 memory tiers:
session / work / learning), Personality, Tools (skills + MCP + prompts),
Security, Orchestration (hooks + agent tiers), Interface. Useful taxonomy,
entirely harness-side, zero connectivity story. Confirms the gap rather than
filling it.

### Your own prior art (important)
- **`zass`** — filesystem-as-database assistant harness, 23 MCP tools, markdown
  state, runs on the agent subscription. Well-designed. **State dir: 2 journal
  entries, empty todos/notes/people.** The harness got built; the substrate
  didn't, so nothing had gravity and it died.
- **`Scheduler`** — local control plane for agent CLIs: queue, schedule windows,
  **usage budgets**, worktrees, run records.
- **`shepherd`** — durable workstream/session organization over tmux-hosted
  Claude Code / Codex.

You have already built the harness, the scheduler, and the orchestrator. The
missing piece is exactly the one under discussion. `zass`'s empty state dir is
the single best argument for this project's framing.

---

## 2. The Apple reality check

This is the part that actually constrains the product, so it's worth being
precise. Access paths, ranked by durability:

| Path | Reaches | Needs | Durability |
|---|---|---|---|
| **CalDAV / CardDAV over iCloud** (app-specific password, 2FA required) | Calendar, Reminders, Contacts | any Linux box | **High.** Documented, still current as of mid-2026, no Mac, no TCC |
| **EventKit / AppleScript / JXA / `shortcuts` CLI** | + Notes, Mail, Messages, Shortcuts, Files, Maps | always-on Mac, Full Disk Access, per-app TCC grants | Medium. Breaks on OS updates; TCC attribution is fiddly (servers use `disclaim` shims so the helper becomes its own TCC-responsible process) |
| **`chat.db` read + Beeper `platform-imessage` / BlueBubbles** | iMessage send/receive, reactions, edits, attachments | always-on Mac, FDA | Medium-low. **No official API in 2026.** ~95% parity on Tahoe; known breakage on message edits and group icons |
| **HealthKit** | Health, workouts, sleep | **native iOS app** | Hard gate. No backend API. Data never leaves the phone unless an app you ship moves it |
| **App Intents** | whatever apps expose | iOS 26.1+/macOS Tahoe 26.1+ | **Rising.** The strategic one |

### The App Intents / MCP development is the most important fact here
- SiriKit is **deprecated as of WWDC 2026**. App Intents is the only sanctioned
  path into a third-party app.
- Apple began shipping **MCP support inside the App Intents framework** in the
  iOS/iPadOS/macOS 26.1 developer betas, and has shipped its own MCP servers
  (`xcrun mcpbridge` in Xcode 26.3).
- App Intents already backs Siri, Spotlight, Shortcuts, widgets, Focus.

Translation: **Apple is turning the OS into an MCP host.** If we speak MCP, the
Apple surface gets strictly better over time without us rewriting anything. That
is a strong argument for MCP as our external contract even where we use
CalDAV/AppleScript internally today.

### The topology consequence
Scope decides hardware, mechanically:
- Calendar + Reminders + Contacts only → **any Linux box**. No Mac.
- \+ Notes / Mail / iMessage → **always-on Mac** (mini), with FDA and TCC pain.
- \+ Health → **a native iOS app we ship**. Non-negotiable.

So "do we build an iOS app" isn't an interface question. It's a data question,
and Health is the forcing function.

---

## 3. Protocol: MCP is now actually load-bearing

The **2026-07-28 spec** is a real infrastructure upgrade, not a toy protocol
revision:
- **Stateless core** + Multi Round-Trip Requests (MRTR) — elicitation and
  sampling no longer need a persistent bidirectional stream. Server-initiated
  requests only while the server is processing a client request.
- **Auth hardening**: servers MUST implement OAuth 2.0 Protected Resource
  Metadata (RFC 9728); clients MUST implement Resource Indicators (RFC 8707) so
  a malicious server can't harvest tokens meant for another; RFC 9207 issuer
  validation; Dynamic Client Registration deprecated in favor of client metadata
  documents (CIMD).
- Header-based routing, cacheable list results, formal extensions framework.

Resource Indicators + PRM are precisely the mechanism for "this consumer gets
scoped access to this slice of my life." That's our permission model, already
specified, already implemented in Tier-1 SDKs.

**Read:** MCP is the external contract. Stateless + MRTR also means elicitation
("confirm this write?") is now a first-class protocol feature — which is the
approval primitive we need.

---

## 4. The economics constraint (don't skip this)

Anthropic announced in May 2026 that Agent SDK usage and `claude -p` would move
off subscription limits onto a separate credit pool at API rates — then
**paused it on June 15, 2026**. Today programmatic usage still draws from the
subscription. But the intent was stated publicly and the pause is a pause.

Implications for an always-on, proactive system:
1. Model access must be a **swappable provider** from day one. Never hardcode.
2. **Token budget is a first-class resource.** A proactive daemon burns tokens
   24/7 with no human in the loop. Your `Scheduler` already has usage gates —
   that concept belongs in the infra layer, not bolted on later.
3. Favor architectures where the expensive model is called *rarely and
   deliberately* — deterministic triggers, cheap filters, LLM only at the
   decision point. Cost discipline is a design constraint, not an optimization.

---

## 5. Where the actual difficulty is

The hard parts of this are unglamorous and none of them are agent problems:

1. **Identity resolution.** The same human is a phone number in `chat.db`, an
   email in a calendar invite, a contact card, and "Mom" in conversation. Nothing
   works until entities are unified.
2. **Personal data is a change stream, not a corpus.** Calendars mutate, events
   move, reminders complete. RAG-over-documents is the wrong shape. You need
   sync cursors, deltas, and an append-only event log.
3. **Write safety.** Reads are easy and cheap to get wrong. Writes send a
   message to a real person, decline a real meeting, delete a real note. Needs:
   idempotency keys, preview/diff, confirmation thresholds, undo, and a full
   audit log. This is where the value *and* the danger both live.
4. **Prompt injection with ambient credentials.** OpenClaw's crisis in one line:
   an agent reading untrusted content (emails, messages, web) while holding
   write access to everything. The infra layer must assume its consumer is
   compromised and enforce policy independently.
5. **Reconciliation.** Two sources disagree about the same event. Someone has to
   pick, and "the LLM decides at read time" is not an answer.

---

## 6. Strawman primitives (to argue about in Phase 1)

Deliberately small. Everything else is downstream.

1. **Source** — a typed connector with a sync cursor. Pull + push. Knows how to
   resume.
2. **Entity** — the normalized objects: `Person`, `Event`, `Task`, `Message`,
   `Place`, `Doc`, `Metric`. Cross-source identity resolution lives here.
3. **Timeline** — append-only event log of everything that happened. Substrate
   for both memory and triggers. The one thing that must never lie.
4. **Memory** — *derived*, opinionated, small: facts, preferences, commitments.
   Explicitly separate from Timeline. (Do not build a memory framework; there
   are eight.)
5. **Action** — typed, effectful, with preview → confirm → execute → undo, an
   idempotency key, and an audit record. The crown jewel.
6. **Trigger** — time + event predicates that wake a downstream consumer. Cheap,
   deterministic, no model in the loop until it fires.
7. **Policy** — per-consumer capability scopes and human-in-the-loop thresholds.
   Enforced in the infra layer, never in the agent.

Consumers (Claude Code, a future app, a cron job, someone else's framework) get
these over MCP with scoped tokens. They bring their own reasoning.

---

## 7. Opinions I'd defend

- **Own the trust surface, rent the habit surface.** Build the approvals /
  audit / undo / review interface ourselves — that's where trust is won or lost
  and it can't be a Telegram message. Ride iMessage or Telegram for the
  conversational surface at v0 because habit beats polish. Revisit when Health
  forces an iOS app anyway.
- **Write-capable from day one, narrowly.** A read-only life dashboard is a toy
  and dies exactly like `zass/state/` died. Pick two writes, make them
  bulletproof.
- **The Timeline is the product.** Memory, triggers, and briefings are all views
  over it. If the log is right, everything downstream is replaceable — which is
  the whole point.
- **Don't build:** a skills marketplace, a memory framework, an agent loop, a
  channel gateway. All four are commoditized and two of them are how OpenClaw
  got owned.

---

## 8. Open questions for Phase 1

1. **Host.** Is there an always-on Mac available? This decides ~40% of scope.
2. **Domains in v0.** Calendar + tasks? Comms? Health? Finance? (Health ⇒ iOS app.)
3. **Read-only or write from day one?** I argue write. Which two writes?
4. **Single-user forever, or eventually 1→N?** Changes auth and tenancy deeply.
5. **Interface at v0** — own app, or ride iMessage/Telegram?
6. **North star:** what's the one request that, working end-to-end, makes this
   feel real? That's the MVP spec.

---

## Sources

- [OpenClaw docs](https://docs.openclaw.ai/) · [repo](https://github.com/openclaw/openclaw)
- [The OpenClaw Security Crisis (Coral)](https://www.coral.inc/blog/2026-03-07-openclaw-security-crisis-2026) · [SecurityWeek](https://www.securityweek.com/openclaw-security-issues-continue-as-secureclaw-open-source-tool-debuts/) · [Kaspersky](https://www.kaspersky.com/blog/openclaw-vulnerabilities-exposed/55263/) · [Giskard](https://www.giskard.ai/knowledge/openclaw-security-vulnerabilities-include-data-leakage-and-prompt-injection-risks)
- [Hermes Agent docs](https://hermes-agent.nousresearch.com/docs/) · [repo](https://github.com/nousresearch/hermes-agent) · [awesome-hermes-agent](https://github.com/0xNyk/awesome-hermes-agent)
- [Daniel Miessler — Personal AI Infrastructure](https://danielmiessler.com/blog/personal-ai-infrastructure)
- [WWDC26 Apple Intelligence guide](https://developer.apple.com/wwdc26/guides/apple-intelligence/) · [Build intelligent Siri experiences with App Schemas](https://developer.apple.com/videos/play/wwdc2026/240/)
- [Apple MCP in App Intents (9to5Mac)](https://9to5mac.com/2025/09/22/macos-tahoe-26-1-beta-1-mcp-integration/) · [AppleInsider](https://appleinsider.com/articles/25/09/22/ios-26-could-get-a-major-ai-boost-with-the-model-context-protocol)
- [iCloud CalDAV settings](https://cli.nylas.com/guides/icloud-caldav-settings) · [Apple: app-specific passwords](https://support.apple.com/en-us/102654)
- [Beeper platform-imessage](https://github.com/beeper/platform-imessage) · [Apple-MCPs](https://github.com/JonathanRReed/Apple-MCPs) · [macos-mcp](https://github.com/krmj22/macos-mcp)
- [HealthKit](https://developer.apple.com/documentation/healthkit) · [What you can and can't do with HealthKit data](https://www.themomentum.ai/blog/what-you-can-and-cant-do-with-apple-healthkit-data)
- [MCP 2026-07-28 spec](https://modelcontextprotocol.io/specification/2026-07-28) · [release post](https://blog.modelcontextprotocol.io/posts/2026-07-28/) · [WorkOS on auth changes](https://workos.com/blog/mcp-2026-spec-agent-authentication)
- [Use the Claude Agent SDK with your Claude plan](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan) · [Anthropic pauses June 15 change](https://www.digitalapplied.com/blog/anthropic-claude-credit-overhaul-june-15-2026)
