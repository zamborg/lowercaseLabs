# parlor V2: push / pull

V1 is built and in daily use (see `PLAN.md`). V2 is not more features. It is the smallest set of primitives
that makes today's features and the ones on the push list below cheap, so each new idea is a config or a small
file rather than new core code.

**Rule for every item:** it either maps onto an existing primitive (cheap, do it), forces one new primitive
(justify it here first), or goes to the parked list.

## The primitives (the pull)

Five things, nothing else. Everything in the app is one of these or a composition.

| # | Primitive | Is today | V2 change |
|---|-----------|----------|-----------|
| 1 | **Home**: files are the database, watched live | agents, chats, media, models, `mcp.json`, `settings.json` | add `conversations/`; stop reseeding agents when the folder is empty |
| 2 | **Runtime**: `LanguageModelRuntime`, image worker, one `settings.json` | single-slot prompt cache, one resident model | prompt-cache slots per conversation (LRU, memory budget) |
| 3 | **Conversation**: participants, an append-only log with a speaker on every line, a scheduler that decides who speaks next | `Room` = one agent, one user, hard-wired | the big one. A 1:1 chat is a conversation with `[agent, user]`. Two agents talking, a group room and "agent + silent observer" are the same type with a different scheduler |
| 4 | **Turn**: run one agent on a transcript with a tool allowlist, stream events | `Room.turn` + `SimpleLoop` + `buildHistory` + `Compactor`, all tied to `Room` | pull it out of `Room` as a pure function of (agent, transcript, tools, settings). Compaction becomes a transcript transform |
| 5 | **Tool surface**: MCP, plus the app exposing itself over MCP | `ToolProvider` (only image is in-process) + MCP client; per-agent allowlist | one extension seam, not two (see below) |

UI shrinks to four screens that map 1:1 to these: **Conversations** (chat view, generalised), **Agents**
(editor), **Library** (models + the performance console), **Tools**.

## The extension question ("modules")

Zubin's ask: modules that are easy to create, mountable on chosen agents, and that local models could write.

Two ways to do it; V2 picks one.

- **A. A Swift `ParlorModule` framework.** In-process, fast, full engine access. But every module needs a
  rebuild, a local model can't safely write one, and it is a second extension mechanism beside MCP.
- **B. MCP only, plus the app hosts one MCP server of itself.** A module is an MCP server in
  `~/parlor/modules/<name>/` (any language; `uv run` for Python) mounted via `mcp.json`. The app's own MCP
  server (`parlor`) exposes the primitives: `agents.list`, `conversation.create / post / read / run`,
  `agent.ask`. A module that needs the fabric calls those. Mounting on an agent is the allowlist we already
  have (`rooms.*`). A local model writes a folder and a one-line config; the watcher mounts it.

**Recommendation: B.** It is the pull. It reuses what exists, it is the only option a local model can author,
and it is also how an external Claude Code drives the app (the original goal). In-process Swift stays only for
what must be (the image worker today). Cost: a few ms per call, and the host MCP server needs a localhost
transport with a per-session token.

Trust note: a module is arbitrary code run unsandboxed (same as any MCP server today). Newly dropped modules
should ask before first launch rather than auto-mount.

## Rooms as the proof

"Two agents chat" and "a chat room" are not a module in V2, they are primitive 3 plus a scheduler:

- **Scheduler** is one tiny protocol: given the log and participants, return who speaks next or stop.
  Shipped: `roundRobin(maxTurns)`, `userDriven`, `mention` (speaker named in the last message).
- **Turn** is primitive 4 with a different transcript framing: "You are X in a group with Y. Transcript: …
  Reply as X."
- Agents can start a room through the host MCP (`conversation.create` + `run`), so "Mabel, get Pip and Babe to
  settle this" works with no extra code.
- Known costs to design for, not discover: alternating personas thrashes a single prompt cache (hence cache
  slots), and participants on different models pay a model swap per turn (the scheduler should batch turns per
  model, or the UI should warn).

## Feedback ledger (push, with disposition)

| Push | Disposition |
|------|-------------|
| Delete / disable MCP servers | done |
| HF search error | done (API dropped `usedStorage`); live test guards it |
| Renaming an agent leaves old file id | done; id follows name on commit |
| `[Image shown]` leaking into replies | done (tool result wording) |
| Control console for speed / memory / tokens | done (Performance, `settings.json`) |
| Efficient LLM and image servers | prompt cache done (≈7× faster follow-up turns); image precision / steps / size in settings |
| PyTorch-only / non-Z-Image repos give cryptic errors | done (preflight refuses with a reason) |
| Other image models (SDXL, Flux, SD1.5) | primitive 5: ship as an MCP server, not a core backend |
| GGUF models | parked (user is fine pinning it); would be a runtime backend, a separate decision |
| Two agents chatting / rooms | primitive 3 + schedulers (above) |
| Modules mountable on chosen agents, local-model-authored | primitive 5, option B |
| Model choice / guardrails | user's to play with; nothing in the app filters. Not an app feature |
| Light mode, Release build with embedded worker | verification debt below |

## Tighten-up list (V1 debt, cheapest first)

1. Seeding: `seedIfEmpty` recreates Mabel and Pip whenever `agents/` is empty, so deleting every agent resurrects
   them. Seed once, remember with a marker file.
2. `PARLOR_*` debug env hooks in the app: keep, but isolate behind one `DebugHooks` type so they stop leaking.
3. The "tools updated" marker is not persisted; after a relaunch the first diff is lost. Persist the tool set
   in the chat folder.
4. `image_model` lives on the agent but only applies to the built-in tool. Move it to the image tool's own
   config so a swapped-in image MCP server doesn't leave a dead field.
5. Pip's seed model (Llama 3.2 3B) is not installed on a fresh machine; seeding should fall back to an
   installed model or prompt for the download.
6. Single-slot prompt cache and `Room`-bound compaction: resolved by primitives 2 and 4.
7. Warnings: the Swift 6 sendable warning in `ModelLibrary`; the image worker still builds `-Onone`
   (upstream crash under `-O`), so re-test when `z-image-swift` updates.
8. Per-turn tool schema cost: every offered tool's JSON is re-sent each step. Measure and consider only
   offering tools a request might need once a mount has many.

## Verification debt (from V1)

Now covered: core flows (11 tests), the click-driven UI flow (create, chat, rename, prune, relaunch), real-model
prompt-cache correctness (`parlor-cli bench`). Still open: light mode, a Release build running a real image
generation, the Models and Tools screens under UI test, multi-hour memory behaviour (idle unload).

## Order of work

1. **Conversation + Turn refactor** (primitives 3 and 4), no new features. Gate: all V1 tests and the UI test
   still pass with a 1:1 chat as the only conversation; relaunch and prune unchanged.
2. **Schedulers + group rooms UI.** Gate: two scripted agents converse N turns and the log attributes each line;
   then the same on a real model (cache slots measured with `bench`).
3. **App-hosted MCP (`parlor` server) + `modules/` folder.** Gate: a throwaway Python module, written by a
   local model from the folder guide alone, mounts and is called by an agent mid-chat.
4. Tighten-up list and verification debt, interleaved.

## Open decisions for Zubin

- **B over A for modules?** (recommended: yes)
- **Refactor first, rooms second?** (recommended: yes; rooms built before the refactor would be a throwaway)
- **Cache slots:** cap by memory budget or by count? (recommended: memory budget, default a fraction of free RAM)
