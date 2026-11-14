# parlor V2: push / pull

V1 is built and in daily use (see `PLAN.md`). V2 is not more features. It is the smallest set of primitives
and specs that makes today's features, and everything on the push list, a config or a small file rather than
new core code. The end state is a local platform for hosting, authoring and running agents, built from five
primitives and four specs, nothing else.

**Rule for every item:** it maps onto an existing primitive (cheap, do it), forces a new primitive (justify it
here first; none has so far), or goes to Later or Non-goals.

**Compatibility rule:** V2 reads every V1 file in `~/parlor` unchanged. New fields are optional and missing
ones inherit. No migration step, and the app stays usable daily after every step of the order of work.

## Decided

| Decision | Source |
|----------|--------|
| Extension seam is MCP only, plus an app-hosted `parlor` MCP server. No Swift module framework | agreed in discussion |
| Rooms and agent-to-agent chat are a Conversation scheduler, built after the Conversation + Turn refactor | agreed in discussion |
| Remote model providers allowed, off by default, badged, marked in the agent header | Zubin, explicit |
| Behaviour settings (tool steps, sampling, limits) live on the agent; memory and resource settings stay global | Zubin's push |
| Agent permission tiers (read, act, configure, destructive) | endorsed ("this is awesome") |
| No parallel generation: serialise per resource, overlap across resources | proposed, not objected to |
| Prompt edits apply on the next message (already true); old replies and the summary are not rewritten | Zubin: "that's chill" |

All dated 2026-11-13. "Proposed, not objected to" items are the first to revisit if they chafe.

## The five primitives

| # | Primitive | V1 today | V2 |
|---|-----------|----------|----|
| 1 | **Home**: files are the database, watched live | `agents/`, `chats/<agent>/`, `media/`, `models/`, `mcp.json`, `settings.json` | adds `modules/`, `workspaces/`, `logs/`; a chat folder can hold a `conversation.json`; seed once |
| 2 | **Runtime**: model backends and every supervised process | one resident MLX model, one cache slot, an image worker, MCP children launched ad hoc | services (state, policy, cost, controls), a job queue per resource, residency and cache slots by memory budget, a provider registry |
| 3 | **Conversation**: participants, an append-only log with a speaker on every line, a scheduler | `Room`: one agent and one user, hard-wired | 1:1 chat is a conversation of `[agent, user]`; rooms, agent-to-agent and autonomous runs are the same type with another scheduler |
| 4 | **Turn**: one agent produces one reply from a transcript, streaming events | `Room.turn` + `SimpleLoop` + `buildHistory` + `Compactor`, tied to `Room` | a function of (agent, transcript, tools, resolved settings) plus an injected runtime and tool executor, with no `Room` state. The built-in loop is the default *harness*; others plug in via the Turn protocol |
| 5 | **Tool surface**: everything an agent or outside client can do | MCP client, in-process image tool, per-agent allowlist | one command layer behind the UI, the `parlor` MCP server and the CLI; permission tiers; module manifests |

## 1. Home

```
~/parlor/
  CLAUDE.md                folder guide; also the module and harness authoring guide
  settings.json            machine settings, provider registry, service policies
  mcp.json                 mounted MCP servers (Claude Desktop format) + lifecycle fields
  agents/<id>.md           persona + front matter (model, provider, tools, tier, harness, overrides)
  chats/<id>/              one conversation per folder
    log.jsonl              append-only; lines gain an optional `speaker` (absent = the folder's agent or user)
    summary.md             the built-in harness's rolling summary
    conversation.json      optional: participants, scheduler. Absent = 1:1 with agent <id>
  media/YYYY-MM/           images and other outputs
  models/<org>/<name>/     MLX weights + .parlor-model.json (model-scope defaults)
  modules/<name>/          MCP servers with module.json
  workspaces/<agent>/      cwd for a harness's long-running work
  logs/<service>.log       one rotating log per service
```

- Keeping `chats/` rather than adding `conversations/` is the pull: a V1 chat already is a V2 conversation, so
  nothing moves.
- **Writers.** The app is the only writer of `log.jsonl`; agents, harnesses and modules add to a conversation
  through `parlor.conversations.post`. Everything else may be hand-edited; the app writes atomically and
  reloads on change.
- Every path derives from `ParlorHome`, which is what the "Your data" links read.
- Seed Mabel and Pip once, remembered by a marker, not "whenever `agents/` is empty".

## 2. Runtime: services, queues, providers

**Services.** Anything the app supervises: the MLX runtime (in-process; start/stop = load/unload), the image
worker, every MCP server and module, external harnesses, local provider servers (e.g. llama.cpp), the Runtime
API.

| Field | Meaning |
|-------|---------|
| state | stopped, starting, running, idle, failed (with last error) |
| policy | always on, on demand, or stop after N idle minutes; restart on crash with backoff, `failed` after 3 |
| cost | memory, queue depth, last used |
| controls | start, stop, restart, open log |

`mcp.json` entries gain `lifecycle` and `idleMinutes`, so a mount no longer means "launched at app start".

**Image worker becomes an MCP server** (open decision 4). It stays a separate process, because its MLX pin
differs, but it speaks MCP stdio instead of its own JSON lines. MCP image content is already rendered inline.
Every tool is then the same kind of thing, and another image backend is a second `mcp.json` entry. The model
choice and the per-agent size and steps move into its tool arguments and config, which retires the agent's
`image_model` field.

**Queues.** One job queue per resource (each provider, the image service). Priority: the user's foreground reply,
then room turns, then background jobs (summaries, autonomous runs). Jobs for the loaded model run first, with a
fairness bound. Queue depth is visible. Cancelled jobs are dropped.

**Memory and cache.**
- Several models stay resident while they fit (LRU). The budget counts services too, since a harness, a
  llama.cpp server and the image worker share unified memory with the models.
- Each resident MLX model has several prompt-cache slots. **A request takes the slot with the longest common
  prefix**, not a slot keyed by conversation id. That one rule makes the cache work for chats, rooms, summaries
  and stateless Runtime API clients alike.
- Image renders may overlap a reply, and both slow down; a setting makes them wait instead.

**Providers** (`settings.json` → `providers`):

| Field | Meaning |
|-------|---------|
| id, kind | `mlx` (built in, the default) or `openai-compatible` |
| baseURL, keyRef | endpoint; the key lives in the Keychain, never in a file |
| models | from `/v1/models`, or listed by hand |
| locality | `local` or `remote`; remote is off by default and badged |

- An agent gains an optional `provider:` field (default `mlx`). It is separate from `model` because V1 model
  values are repo ids that already contain a slash.
- GGUF: run a llama.cpp server as a service and register it as a local provider. The prompt cache applies only
  to the MLX provider.

**Runtime API.** The reverse direction: the app serves its models over a 127.0.0.1-only OpenAI-compatible
endpoint (chat completions, streaming, tool calls) with per-client tokens. It is off by default, except that
launching an external harness starts it for that harness's token. Requests go through the same queues and cache
slots as the app's own. One OpenAI wire codec serves both directions.

## 3. Conversations, rooms and schedulers

- **Scheduler** is one protocol: given participants and the log, return who speaks next, or stop. V2 ships
  `userDriven` (today's chat), `roundRobin(maxTurns)`, `mention` (whoever the last message names) and
  `autonomous` (an agent continues without a user prompt, bounded by a step budget).
- **Framing for groups.** The Turn sees the transcript with speakers named ("You are X in a group with Y and Z").
  Other agents' lines arrive as user-role text attributed to their speaker.
- Agents start rooms through the `parlor` server (`conversations.create` + `run`), so "Mabel, get Pip and Babe to
  settle this" needs no code.
- **Costs to design for.** Alternating personas would thrash one cache (fixed by prefix-matched slots), and
  participants on different models pay a swap per turn (the queue prefers the loaded model; the room UI warns
  when participants span models).
- **No user present.** Destructive actions in an autonomous run are held, not executed, and surface as a notice
  in the conversation.

## 4. Turns and harnesses

**The line.** A harness owns how the model is prompted, when tools are called, how context is trimmed or
summarised, and any memory or planning. The app owns the log, the UI, tool execution, permission tiers and model
hosting.

**One event model, two transports.** The Turn's events (`text.delta`, `tool.call`, `tool.result`, `status`,
`turn.done {stats}`, `error`) are one Swift type.
- The built-in loop (`SimpleLoop` + `Compactor`) implements the Turn in-process against that type and calls the
  MLX runtime directly. It is not a subprocess, and it pays no serialisation cost.
- External harnesses get the same events as JSON lines over stdio.
- If the built-in loop needs something the protocol can't carry, the protocol is wrong.

**Turn protocol (external).**
- app → harness: `turn.start {agent, transcript, tools, settings}`, `tool.result`, `turn.cancel`
- harness → app: `text.delta`, `tool.call`, `status`, `turn.done`, `error`
- **Tools run in the app.** The harness sends `tool.call`; the app checks the allowlist and tier, runs it,
  records it (and any image) in the log, and returns `tool.result`. One enforcement point.
- **The model is reached through the Runtime API.** At launch the app injects `OPENAI_BASE_URL` and a per-process
  token, so an existing OpenAI-style harness works unmodified.
- **Work after the turn.** A harness running long jobs posts results later through
  `parlor.conversations.post`, which is tiered like any other call.
- **Failure.** If a harness crashes mid-turn, the partial text is kept (as Stop does today), an error line is
  logged, and the service restart policy applies.
- **ACP.** Before finalising, evaluate adopting ACP (Agent Client Protocol, JSON-RPC over stdio) instead of our
  own format. Use it if it carries the event set above without contortion.

**External harnesses.** An agent's front matter says `harness: <command>`. The app runs it as a supervised
service with `cwd` at `workspaces/<agent>/` and its stderr in `logs/`. tmux is a launch detail (the command can
be `tmux new -d …`); the dashboard offers "Open in Terminal". A harness keeps its own derived state (summary,
memory) in its workspace; the conversation log stays the app's.

**Cost.** Prefix-matched cache slots help any harness that keeps its prompt prefix stable. A harness that
reshuffles history pays full prefill every turn, and nothing on our side fixes that.

## 5. Tool surface: one command layer

**State is files, actions are tools, and every front end calls the same verbs.** The UI, the `parlor` MCP server
and `parlor-cli` are three front ends onto one set of engine verbs. A feature with no verb is unfinished.

- File-backed state (`settings.json`, `agents/*.md`, `mcp.json`) changes through validated writers. Bad values
  are refused with a reason; good ones apply live. Hand edits still work.
- Actions with no file behind them are the server's real value: load or unload a model, start or stop a
  service, download, run a conversation, prune, compact, reveal.
- `parlor-cli` today builds its own `ParlorEngine`, so it loads a second copy of a model. In V2 it becomes a
  thin client of the server whenever the app is running, and keeps its standalone mode for tests.

**Tool groups** (`parlor.<group>.<verb>`): `agents` (list, read, create, update, delete), `conversations`
(create, post, read, run, prune, compact), `models` (list, search, download, load, unload), `providers` (list,
test), `services` (list, start, stop, restart, logs), `settings` (get, set, reset), `tools` (list, mount,
unmount, enable), `home` (paths, reveal).

**Tiers.** Every caller (agent or token) carries a tier, which gates the verbs.

| Tier | Allows | Default |
|------|--------|---------|
| none | `parlor` verbs not offered | character template |
| read | list and read anything | assistant template |
| act | run conversations, post, load models, start services | opt-in |
| configure | edit agents, settings, mounts | opt-in |
| destructive | delete, prune, uninstall | always asks the user in the UI |

- **For agents**, a tier is an allowlist entry (`parlor.read`, `parlor.act`, …), so there is no new mechanism.
- **For launched processes** (modules, harnesses), the injected token *is* the identity and carries the tier.
- **An external Claude Code** gets a token from the System screen, with the tier you pick.

**Transport.** 127.0.0.1 only, with a token. Children the app launches get `PARLOR_MCP_URL` and their token in
the environment.

**Module manifest** (`modules/<name>/module.json`): name, one-line description, launch command, what it needs
(network, files, `parlor` tier), an optional settings schema rendered as a form, and a default lifecycle.
- A newly dropped module shows its manifest and asks before first launch.
- `CLAUDE.md` in Home is the authoring guide a local model reads.

## Settings scopes: machine, model, agent

**The test:** does it change what *this agent* says or does, or what the *machine* spends?

| Scope | Lives in | Settings |
|-------|----------|----------|
| Machine | `settings.json` | prompt cache, KV precision, attention window, prefill step, model unload timer, image precision, service policies, memory budget, queue policy, providers |
| Model | `.parlor-model.json` | recommended sampling, max context |
| Agent | `agents/<id>.md` | model, provider, harness, tier, tools, temperature, top-p, repetition penalty, reply limit, context size, compaction threshold, keep-verbatim, summary length, max tool steps, tool result limit, tool timeout, image size and steps, thinking |

- **Resolution.** Effective settings = agent, else model, else machine. The agent's context size is clamped to
  the model's maximum. Settings are resolved once and passed to the Turn as values.
- **Editor.** The agent editor shows overrides with the inherited value as placeholder; clearing a field
  inherits again.
- **V1 already does part of this.** V1 agents carry context, compaction threshold, temperature and thinking, so
  this extends an existing pattern.
- **Memory knobs stay global.** An attention window or KV quantisation on one agent would disable the shared
  prompt cache for that agent and make memory unpredictable.

## Agents and characters

`kind` changes only the sort order, the creation defaults and an editor picker; the turn loop never reads it.
- V2 has one `Agent` type. Assistant and character become *creation templates* (starter persona, tool list,
  tier, context, temperature, model).
- The sidebar groups agents by a free `group` field. A V1 `kind:` line is read as the initial group.

## UI: four screens

| Screen | Holds | Primitives |
|--------|-------|-----------|
| Conversations | chats and rooms; per-conversation menu (reveal log, open summary, compact, prune) | 3, 4 |
| Agents | editor: persona, model and provider, tools and tier, harness, overrides; flags a missing model | 4, 5 |
| Library | models (search, download, installed), providers | 2 |
| System | services dashboard with logs, queue and memory; tokens; machine settings (today's Performance console); mounts and modules; "Your data" links | 1, 2, 5 |

Today's Tools screen folds into System, because a mount is a service (open decision 5). "Your data" uses Reveal
in Finder for folders and Open for files.

## The four specs (`SPEC.md`)

Written before code; each has a conformance test the built-in implementation must pass.

| Spec | Between | First implementation |
|------|---------|----------------------|
| Turn protocol | app ↔ harness | the built-in loop (in-process, same event type) |
| Tool surface | app ↔ tools and clients (MCP, `parlor` verbs, tiers, module manifest) | the image server, the `parlor` server |
| Runtime API | harness ↔ our models (OpenAI-compatible) | a scripted test harness |
| Provider spec | app ↔ model backends | the MLX runtime; then a llama.cpp server |

Plus the file formats in Home: agent front matter, `conversation.json`, the log line, `module.json`,
`settings.json`.

## Later (wanted, not V2)

- A built-in terminal (PTY) for harness sessions.
- Sandboxing harnesses and modules. Until then they run unsandboxed like any MCP server today, limited only by
  their workspace and tier.
- "Reset chat context": clear the summary and keep the log.
- Offering only the relevant tools per request once mounts are large.

## Non-goals

- Parallel generation or batching.
- GGUF or non-Z-Image image backends in the core.
- Multi-user use or network access beyond 127.0.0.1.
- Sync.
- A Swift module framework.
- Rewriting old replies when a persona changes.

## Order of work

Each step ships, keeps the app usable daily, and has a gate that must pass before the next starts. The order
assumes the recommended answers to the open decisions.

0. **`SPEC.md`.** Gate: the built-in loop, the image tool and a 1:1 chat can each be described in it without
   exceptions; Zubin reviews.
1. **Conversation + Turn refactor**, with resolved settings as the Turn's input (file-level overrides only, no
   UI). No new features. Gate:
   - all V1 tests and the UI test pass;
   - relaunch, prune and `parlor-cli bench` are unchanged;
   - the built-in loop passes the Turn conformance test.
2. **Runtime core.** Queues, memory-budgeted residency and prefix-matched cache slots. Gate:
   - two chats prompting at once show a queue and both finish;
   - alternating between two chats keeps both cached (`bench`).
3. **Rooms.** Schedulers, `conversation.json`, speaker lines, room UI. Gate:
   - two scripted agents converse N turns with correct attribution;
   - then the same on a real model, with cache hits measured.
4. **Services.** Dashboard, `lifecycle` on mounts, `logs/`, restart policy, image worker as MCP. Gate:
   - the image tool works unchanged through MCP;
   - stopping and starting each service from System works;
   - a crashed MCP server shows `failed` with its error.
5. **Command layer, `parlor` server, tiers, modules, thin CLI.** Gate:
   - every UI action maps to a verb, checked against a list in `SPEC.md`;
   - an external Claude Code with an `act` token runs a room;
   - a Python module written by a local model from `CLAUDE.md` alone mounts after approval and is called
     mid-chat;
   - a `read`-tier agent is refused a write.
6. **Providers, Runtime API, external harnesses.** Gate:
   - a llama.cpp GGUF model answers through a provider;
   - an unmodified OpenAI-style script harness runs as an agent, its tool calls are enforced by tiers, and its
     output looks identical in chat.
7. **Settings scopes UI, agent templates, tighten-up and verification debt**, interleaved with earlier steps
   where they touch the same code.

## Feedback ledger (push, with disposition)

| Push | Disposition |
|------|-------------|
| Delete / disable MCP servers; HF search error; rename leaves old id; `[Image shown]` leak; cryptic non-MLX errors | done in V1 |
| Control console; efficient LLM and image servers | done (Performance, prompt cache ≈7× faster follow-ups); becomes System → machine settings |
| Other image models (SDXL, Flux, SD1.5) | an MCP server; step 4 makes the built-in one the same kind |
| GGUF models | a llama.cpp server as a local provider (step 6) |
| Two agents chatting, chat rooms | Conversation schedulers (step 3) |
| Modules mountable on agents, authored by local models | MCP + manifest + `CLAUDE.md` guide (step 5) |
| Modules that are controllable | lifecycle, manifest, tiers, approve on first launch (steps 4–5) |
| Every control exposed to the app as tools | one command layer; `parlor` server (step 5) |
| Parallel models and prompts | queues, residency budget, no parallel generation (step 2) |
| Service dashboard; choose what spins up | services in System (step 4) |
| Bring your own harness; harness gets its own process; host, author and run agents to a spec | Turn protocol + Runtime API + workspaces (step 6); `SPEC.md` (step 0) |
| Register LLMs as a backbone | provider registry (step 6) |
| Tool steps and similar belong to the agent | settings scopes (steps 1 and 7) |
| Agent vs character | one type, templates (step 7) |
| Links to logs and data | "Your data" in System; per-conversation menu |
| Prompt changes hot-applied? | already: next message; the cache re-prefills once |
| Reset chat context | Later |
| Model choice and guardrails | the user's call; nothing in the app filters |

## Tighten-up list (V1 debt)

1. Seeding resurrects Mabel and Pip whenever `agents/` is empty (fixed by the Home rule above).
2. Pip's seed model is not installed on a fresh machine: seed with an installed model or prompt to download.
3. `PARLOR_*` debug hooks: isolate behind one `DebugHooks` type.
4. The "tools updated" marker is not persisted, so the first diff after a relaunch is lost. Store the tool set
   in the chat folder.
5. The Swift 6 sendable warning in `ModelLibrary`. The image worker builds `-Onone` (upstream crash under `-O`);
   re-test when `z-image-swift` updates.
6. Tool schemas are re-sent every step: measure, since it matters more with external harnesses.
7. Babe points at an uninstalled model (`MythoMax`); the editor should flag agents whose model is missing.

Resolved by the design: the single-slot cache (step 2), `Room`-bound compaction (step 1), the dead
`image_model` field (step 4).

## Verification debt

Covered: 11 core tests, the click-driven UI flow, real-model prompt-cache correctness (`bench`).

Still open from V1: light mode, a Release build running a real image generation, the Models and Tools screens
under UI test, multi-hour idle unload.

New for V2: each spec's conformance test, and a soak test with two chats, a room and the image service under one
memory budget.

## Open decisions for Zubin

1. **`SPEC.md` before the refactor?** Recommended: yes.
2. **Rooms (step 3) before services and the command layer (steps 4–5)?** Recommended: yes; rooms need only steps
   1–2. Swap if driving the app from Claude Code matters more first.
3. **Residency and cache slots capped by memory budget** (recommended; default a fraction of RAM) or by count.
4. **Image worker as an MCP server** (recommended) vs keeping its own protocol behind a service entry.
5. **Four screens, with Tools folded into System** (recommended) vs keeping Tools separate.
