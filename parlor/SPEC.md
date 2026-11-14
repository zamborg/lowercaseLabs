# parlor specs (v0.1)

The contracts behind `PLAN-V2.md`. The four specs plus the Home file formats. Code implements these; when the
code and this file disagree, one of them is a bug.

- **Status:** v0.1, written before step 1 of the order of work. Every section says which step implements it.
- **Versioning:** every wire format and every JSON file carries `"version": 1`, except files V1 already wrote
  without one, where a missing version means 1. A reader ignores unknown keys. Breaking changes bump the version.
- **Words:** MUST and MUST NOT are requirements for conformance. Anything else is guidance.

Contents:
1. Home: file formats
2. Turn spec: the in-process harness and the external protocol
3. Tool surface: tools, `parlor` verbs, tiers, modules
4. Runtime API: serving our models
5. Provider spec: consuming model backends
6. Conformance tests
7. Step 1 contract (the refactor), for execution
8. Settled defaults for the open decisions

---

## 1. Home: file formats

Root: `~/parlor`, or `$PARLOR_HOME`. Compatibility rule: every V1 file is read unchanged, new keys are optional,
and a missing key inherits.

```
~/parlor/
  CLAUDE.md                 folder guide; authoring guide for modules and harnesses        V1
  settings.json             machine settings, providers, services                          V1, extended
  mcp.json                  mounted MCP servers                                            V1, extended
  agents/<id>.md            one agent                                                      V1, extended
  memory/<id>.md            the agent's durable memory, shared across its conversations    step 7
  chats/<id>/log.jsonl      one conversation's log                                         V1, extended
  chats/<id>/summary.md     the built-in harness's rolling summary                         V1
  chats/<id>/conversation.json  participants and scheduler                                 step 1 (read), 3 (write)
  chats/<id>/tools.json     last tool set seen, for the "Tools updated" marker              step 1
  media/YYYY-MM/<uuid>.<ext>  generated and attached media                                 V1
  models/<org>/<name>/      weights + .parlor-model.json                                   V1, extended
  modules/<name>/           an MCP server + module.json                                    step 5
  workspaces/<agent>/       cwd for an external harness                                    step 6
  logs/<service>.log        rotating service logs (1 MB × 3)                               step 4
  .parlor-seeded            marker: starter agents were created once                       step 1
  .git/                     automatic history of hand-editable state                       step 7
```

**Writers.** The app is the only writer of `log.jsonl` and `summary.md`. Everything else may be edited by hand
or by an external tool. The app writes atomically (temp file and rename), except for the append to
`log.jsonl`, and reloads on an FSEvents change.

**History (step 7).** The app keeps `~/parlor/.git`. It commits after each app write to `agents/`,
`memory/`, `settings.json`, `mcp.json`, `modules/*/module.json` and `conversation.json`, with the actor (UI,
agent id, token name) as the author. `models/`, `media/`, `logs/`, `workspaces/` and `chats/*/log.jsonl` are
git-ignored. This is the undo for edits made by configure-tier agents. Restoring is `parlor.home.restore`.

### 1.1 Agent: `agents/<id>.md`

YAML-style front matter, then the persona as the body. `id` is the file name, and it follows `name` on rename
(V1 behaviour).

| Key | Type | Default | Since |
|-----|------|---------|-------|
| `name` | string | id | V1 |
| `description` | string | "" | V1 |
| `kind` | `assistant` \| `character` | assistant | V1; from step 7 read only, as the initial `group` |
| `group` | string | from `kind` | step 7 |
| `model` | repo id | `mlx-community/Qwen3-4B-4bit` | V1 |
| `provider` | provider id | `mlx` | step 6 |
| `image_model` | repo id | none | V1; retired in step 4 (moves to the image tool's config) |
| `tools` | list of patterns | [] | V1 |
| `tier` | `none` \| `read` \| `act` \| `configure` | none | step 5 |
| `harness` | command line | built in | step 6 |
| `context` | int (tokens) | 16384 | V1 |
| `compact_at` | 0–1 | 0.7 | V1 |
| `temperature` | number | 0.6 | V1 |
| `thinking` | bool | false | V1 |
| `color` | int | 0 | V1 |
| `top_p` | number | machine | step 1 |
| `repetition_penalty` | number | machine | step 1 |
| `max_reply_tokens` | int, or `none` | machine | step 1 |
| `max_steps` | int | machine | step 1 |
| `tool_result_chars` | int | machine | step 1 |
| `tool_timeout` | int (seconds) | machine | step 1 |
| `keep_recent` | int | machine | step 1 |
| `summary_tokens` | int | machine | step 1 |
| `image_size` | `{fast: int, detailed: int}` | machine | step 4 |
| `image_steps` | `{fast: int, detailed: int}` | model default | step 4 |

The writer MUST preserve keys it does not understand. V1 `fileText()` rewrites from known fields only, so step
1 adds a pass-through of unknown keys.

**Tool patterns:** `*`, `server.*` or `server.tool`, matched against the qualified name (V1 `Agent.allows`).
Tier entries are not patterns; they live in `tier`.

### 1.2 Log line: `chats/<id>/log.jsonl`

One JSON object per line, ISO-8601 dates, slashes unescaped. These are V1 `ChatMessage` fields plus `speaker`.

```json
{"id":"UUID","role":"user|assistant|tool|marker","text":"…","speaker":"mabel",
 "toolCalls":[{"id":"…","name":"server__tool","arguments":{}}],
 "toolCallID":"…","toolName":"server__tool","isError":false,"seconds":1.2,
 "images":["media/2026-11/x.png"],"createdAt":"2026-11-13T10:00:00Z"}
```

- `speaker` (step 1): an agent id, or `"user"`. If it is absent, a `user` line's speaker is `"user"` and any
  other line's speaker is the conversation's only agent. A reader MUST accept lines without it. A writer MUST
  set it on new `user`, `assistant` and `tool` lines.
- `images` on a `user` line are attachments (reserved for the step that adds vision input). They are paths
  relative to Home.
- Unknown keys are kept on rewrite (prune).

### 1.3 Summary: `chats/<id>/summary.md`

V1 format: front matter `through: <message UUID>`, then the summary as the body. It is owned by the built-in
harness. An external harness keeps its own state in its workspace.

### 1.4 Conversation: `chats/<id>/conversation.json`

```json
{ "version": 1,
  "title": "Settle it",
  "participants": [ {"agent": "mabel"}, {"agent": "pip"}, {"user": true} ],
  "scheduler": { "kind": "roundRobin", "maxTurns": 6 },
  "triggers": [] }
```

- If the file is absent, the conversation is `participants: [{agent: <folder id>}, {user: true}]` with
  `scheduler: {kind: "userDriven"}`. Every V1 chat is this case.
- `scheduler.kind` is one of:
  - `userDriven`: the agent replies to each user message.
  - `roundRobin`: agents take turns, up to `maxTurns`; a user message restarts the count.
  - `mention`: the agent named in the last message speaks; otherwise the first agent speaks.
  - `autonomous`: the first agent continues without a user message, for at most `stepBudget` turns, then stops.
- `triggers` is a list of `{ "every": "30m" | "2h" | "1d", "at": "08:00" }` (clock; `at` is local time of day, and
  `every` then counts whole days, default `1d`) or `{ "onFile": "<glob relative to the parlor folder>" }`. Each
  trigger starts a run of the scheduler with no user message, so it suits `autonomous` rooms or rooms with history.
  Clock triggers count from when the app first sees them; a time that passes while the app is closed is not made up
  (last runs live in `chats/<id>/triggers.json`). A file trigger waits 30 s between runs and ignores the room's own
  folder, `logs/` and `.git/`, so an agent writing files cannot start itself in a loop.
- Folder ids for new conversations are `<slug>-<4 hex>`, so they never collide with an agent id.

### 1.5 Tool set snapshot: `chats/<id>/tools.json` (step 1)

`{"version":1,"tools":["image.generate","…"]}` is the qualified names offered on the last turn. It replaces V1's
in-memory snapshot, so "Tools updated" markers survive a relaunch.

### 1.6 Memory: `memory/<id>.md` (step 7)

Markdown with no front matter, owned by the agent and shared across all its conversations. The built-in harness
puts it in the system prompt under "What you remember". Agents change it through `parlor.memory.read`,
`parlor.memory.append` and `parlor.memory.replace`; their own memory needs tier `none` or higher, someone
else's needs `configure`. External harnesses get it in `turn.start` as `memory`.

### 1.7 Model metadata: `models/<org>/<name>/.parlor-model.json`

V1 has `{"kind":"language|image","repo":"…"}`. Extended (step 1, read only):

```json
{ "kind": "language", "repo": "mlx-community/Qwen3-4B-4bit",
  "defaults": { "temperature": 0.7, "top_p": 0.8, "context": 32768 },
  "capabilities": { "tools": true, "vision": false, "maxContext": 40960 } }
```

- `defaults` is the model scope of settings resolution.
- `capabilities.maxContext` clamps the agent's `context`.
- `capabilities.tools: false` makes the editor warn when the agent has tools; the Turn still offers them.
- All keys are optional.

### 1.8 Machine settings: `settings.json`

V1 `ParlorSettings` (`llm`, `image`, `loop`) is unchanged, with tolerant decoding. Additions:

```json
{ "memoryBudgetFraction": 0.6,
  "cacheSlotsPerModel": 4,
  "providers": [ {"id":"mlx","kind":"mlx"} ],
  "services": { "<service id>": {"lifecycle":"onDemand","idleMinutes":10} },
  "runtimeAPI": { "enabled": false, "port": 11500 },
  "parlorServer": { "enabled": true, "port": 11501 } }
```

`llm.topP`, `llm.repetitionPenalty`, `llm.maxReplyTokens` and all of `loop.*` become machine *defaults* that
agents override (step 1). Memory knobs stay machine-only: `promptCache`, `kvBits`, `kvGroupSize`,
`quantizedKVStart`, `maxKVSize`, `prefillStepSize`, `keepLoadedMinutes`, `image.precision`, `image.idleSeconds`.

### 1.9 MCP mounts: `mcp.json`

Claude Desktop format (V1: `command`, `args`, `env`, `cwd`, `url`, `disabled`). Additions (step 4) are
per-entry `lifecycle` (`always` | `onDemand`, default `always` for V1 compatibility) and `idleMinutes`.

### 1.10 Module manifest: `modules/<name>/module.json` (step 5)

```json
{ "version": 1, "name": "dice", "description": "Roll dice with notation like 2d6+1.",
  "command": "uv", "args": ["run", "server.py"],
  "needs": { "network": false, "files": "workspace", "tier": "none" },
  "settings": { "type": "object", "properties": { "maxDice": {"type":"integer","default":20} } },
  "lifecycle": "onDemand" }
```

- `settings` is a JSON Schema, rendered as a form. Values are stored in `modules/<name>/settings.json` and passed
  to the server as `PARLOR_MODULE_SETTINGS` (a JSON string).
- On first sight, the app records the manifest's hash in `settings.json → services` and asks before launching.
  If the hash changes, the app asks again.

---

## 2. Turn spec

A **turn** is one agent producing one reply (including its tool steps) from a conversation. A **harness** is
whatever does that.

### 2.1 Inputs

```swift
public struct TurnInput: Sendable {
    var agent: Agent
    var conversationID: String
    var participants: [Participant]      // for framing; 1:1 = [agent, user]
    var transcript: [ChatMessage]        // full log including markers; the harness filters
    var summary: ChatLog.Summary?        // built-in harness only
    var tools: [ToolDef]                 // already filtered by the agent's allowlist
    var settings: ResolvedSettings       // §2.4
    var lastPromptTokens: Int            // from the previous turn's stats, 0 if unknown
    var now: Date
}

public struct TurnContext: Sendable {
    var llm: LanguageModelRuntime         // queue and cache live behind this
    var tools: ToolExecutor               // the ONLY way a harness runs a tool
}

public protocol ToolExecutor: Sendable {
    /// Checks the agent's allowlist (and later its tier), applies the timeout and the result clip,
    /// and never throws: failures come back as `isError` output.
    func call(_ call: ToolCallRecord, agent: Agent, settings: ResolvedSettings) async -> ToolOutput
}

public protocol Harness: Sendable {
    func run(_ input: TurnInput, context: TurnContext,
             emit: @Sendable @escaping (TurnEvent) -> Void) async throws
}
```

### 2.2 Events

```swift
public enum TurnEvent: Sendable {
    case delta(String)                       // the whole visible text so far for this step (replace, not append)
    case status(String)                      // "Thinking", "Compacting", "Running image.generate"
    case toolStarted(ToolCallRecord)
    case message(ChatMessage)                // a finished assistant or tool line, to append to the log
    case summary(ChatLog.Summary, folded: Int) // the harness compacted; the app persists it and adds a marker
    case stats(LLMStats)
}
```

Rules:
- `delta` carries the cumulative visible text of the current step, with `<think>` blocks removed (as V1 does). A
  new step starts with the next `delta` after a `message`.
- A `message` with role `assistant` is emitted once per model step, including a step that only calls tools. An
  assistant message with empty text and no tool calls is not emitted.
- Every assistant tool call is followed by exactly one `tool` message with the same `toolCallID`, in order.
- The app sets `speaker` on lines from `message`. Harnesses need not.
- The turn ends when `run` returns. A thrown error is shown to the user. Partial `delta` text is kept as an
  assistant line on cancellation, but not on error (V1 behaviour).
- `turn.done` exists only on the wire (§2.5). In-process it is the function returning.

### 2.3 The built-in harness (behaviour to preserve exactly)

This is V1 `SimpleLoop` + `Compactor` + `Room.buildHistory`, moved, not changed.

1. **Compaction first.** It runs if `max(lastPromptTokens, estimate(history without tools)) > compact_at ×
   context`. The cut is at a user message, keeping `keep_recent` messages verbatim. The summary is generated
   with `useCache: false`. The harness emits `status("Compacting")`, then `summary(…)`.
2. **System prompt**, in this order:
   - the persona (or "You are <name>."),
   - a blank line and "Today is <full date>." (date only, which keeps the cache prefix stable),
   - if there are tools, the V1 tool-use instruction paragraph,
   - if there is a summary, "## Earlier conversation (summary)" and its text.
3. **History:** non-marker lines after `summary.through`, mapped by role.
4. **Loop:** up to `max_steps` model steps. The last step offers no tools. Each tool call goes through
   `context.tools.call`. Results are clipped to `tool_result_chars` (keeping 2/3 head and 1/3 tail, with the V1
   note).
5. **Request:** `temperature`, `max_reply_tokens`, `thinking`, and `LLMTuning(settings: resolved.llm)` where
   `resolved.llm` has the agent's `top_p` and `repetition_penalty`.

**Group framing (step 3, built-in harness):**
- For participants other than this agent, `assistant` lines become `user` lines prefixed `"<Name>: "`.
- The system prompt gains: "You are <Name>, in a conversation with <names>. Reply only as <Name>; do not write
  other participants' lines."
- The user's lines are prefixed `"User: "` only when there are two or more agents.

### 2.4 Settings resolution

```swift
public struct ResolvedSettings: Sendable, Equatable {
    var temperature, topP: Double
    var repetitionPenalty: Double?
    var maxReplyTokens: Int?
    var thinking: Bool
    var contextTokens: Int; var compactAt: Double
    var keepRecent, summaryTokens: Int
    var maxSteps, toolResultChars, toolTimeoutSeconds: Int
    var llm: ParlorSettings.LLM     // machine knobs with topP / repetitionPenalty / maxReplyTokens replaced
    static func resolve(agent: Agent, model: ModelMetadata?, machine: ParlorSettings) -> ResolvedSettings
}
```

- **Precedence:** an agent key if present, else the model's `defaults`, else the machine value.
- **Context clamp:** `contextTokens = min(agent context, model.capabilities.maxContext)` when the model gives one.
- **V1 agents:** they always write `temperature`, `context`, `compact_at` and `thinking`, so for them the agent
  value wins. That is intended.

### 2.5 External harness protocol (step 6)

The app launches `harness:` with `cwd = workspaces/<agent>/` and this environment:

| Variable | Value |
|----------|-------|
| `OPENAI_BASE_URL` | `http://127.0.0.1:<runtimeAPI.port>/v1` |
| `OPENAI_API_KEY` | a per-process token |
| `PARLOR_MCP_URL` | `http://127.0.0.1:<parlorServer.port>/mcp` |
| `PARLOR_TOKEN` | the same per-process token, which carries the agent's tier |
| `PARLOR_AGENT` | the agent's id |

Transport is JSON lines on stdin and stdout; stderr goes to `logs/harness-<agent>.log`. The process is
long-lived and serves turns one at a time per conversation.

app → harness:
```json
{"type":"turn.start","version":1,"turn":"<uuid>","conversation":"<id>","agent":{…front matter…, "persona":"…"},
 "participants":[…],"transcript":[…log lines…],"tools":[{"name":"server__tool","description":"…","schema":{}}],
 "settings":{…ResolvedSettings…},"memory":"…"}
{"type":"tool.result","turn":"…","callID":"…","text":"…","isError":false,"images":["media/…"]}
{"type":"turn.cancel","turn":"…"}
```

harness → app:
```json
{"type":"text.delta","turn":"…","text":"cumulative visible text"}
{"type":"status","turn":"…","text":"Searching"}
{"type":"message","turn":"…","role":"assistant","text":"…","toolCalls":[{"id":"…","name":"server__tool","arguments":{}}]}
{"type":"tool.call","turn":"…","callID":"…","name":"server__tool","arguments":{}}
{"type":"stats","turn":"…","promptTokens":0,"completionTokens":0,"tokensPerSecond":0}
{"type":"turn.done","turn":"…"}
{"type":"error","turn":"…","message":"…"}
```

- **Tool calls.** The harness MUST send `tool.call` and wait for `tool.result`. The app runs the call through
  the same `ToolExecutor`, logs the tool line and returns the result. The harness does not emit tool `message`s
  itself.
- **Work after the turn.** Output after `turn.done` goes through `parlor.conversations.post` with the token.
- **ACP.** Evaluate ACP before step 6. If ACP carries this event set, adopt it and map these names onto its
  messages; this section then becomes the mapping.

---

## 3. Tool surface

### 3.1 Tools

- `ToolDef {server, name, description, schema}`. The wire name is `server__tool` (what the model sees). The
  qualified name is `server.tool` (what humans, allowlists and logs use).
- `ToolOutput {text, images[], isError}`. Images are saved under `media/` and logged on the tool line. The model
  sees only the text.
- MCP image content blocks become images (V1). The built-in image tool's result text tells the model the image
  is already shown (V1 wording).

### 3.2 `parlor` server (step 5)

MCP over streamable HTTP at `127.0.0.1:<parlorServer.port>/mcp`, bearer token. For agents, the in-process
`parlor` provider exposes the same verbs, gated by the agent's `tier`. Each verb's tool is `parlor.<group>.<verb>`.

| Verb | Tier | UI action today |
|------|------|-----------------|
| `agents.list`, `agents.read` | read | sidebar, editor |
| `agents.templates` | read | New Agent menu |
| `agents.create` | configure | New Agent menu (any template); `harness` set by an agent or token asks the user first |
| `agents.update` | configure | editor save, rename |
| `agents.duplicate` | configure | Duplicate |
| `agents.delete` | destructive | Delete agent |
| `conversations.list`, `conversations.read` | read | sidebar, chat view |
| `conversations.create` | act | (rooms, step 3) |
| `conversations.post` | act | send a message |
| `conversations.run` | act | (rooms) |
| `conversations.stop` | act | Stop |
| `conversations.compact` | act | Compact now |
| `conversations.prune` (`ids` or `after`) | destructive | delete messages, cut after |
| `conversations.reset` | destructive | Clear chat |
| `models.list`, `models.search` | read | Models |
| `models.download`, `models.pause` | act | Download / Pause / Resume / Retry |
| `models.delete` | destructive | Delete model |
| `models.load`, `models.unload` | act | (System) |
| `providers.list`, `providers.test` | read | (Library) |
| `services.list`, `services.logs` | read | (System) |
| `services.start`, `services.stop`, `services.restart` | act | Restart server |
| `settings.get` | read | Performance |
| `settings.set`, `settings.reset` | configure | Performance fields, Reset to defaults |
| `tools.list` | read | Tools |
| `tools.mount`, `tools.unmount`, `tools.enable` | configure | add, delete, enable or disable a server |
| `home.paths` | read | Your data |
| `home.reveal` | read | Reveal in Finder (UI only; does nothing over the wire) |
| `memory.read`, `memory.append`, `memory.replace` | none | Memory in the agent editor (an agent's own; another agent's needs configure) |
| `home.history` | read | System → History |
| `home.restore` | destructive | System → History |

Rules:
- **Gate for step 5:** every UI action in the right-hand column calls the engine verb, and nothing else.
- **Destructive verbs** called by an agent or token show a confirmation in the app and block until it is
  answered. With no window open, or in an autonomous run, they are refused with "needs the user". Tokens never
  bypass this.
- **Writes are validated.** `settings.set` and `agents.update` refuse unknown keys and out-of-range values with
  a reason.

### 3.3 Tiers and tokens

The order is `none` < `read` < `act` < `configure`; `destructive` is never a tier, it always asks.
- **Agents:** an agent's `tier` gates `parlor.*` tools. Mounted MCP tools are gated by `tools` patterns as in V1.
- **Tokens:** created in System. Each has a name, a tier and an optional expiry. They are stored in the Keychain,
  with only names and hashes in `settings.json`.
- **Launched processes:** modules and harnesses get a per-process token with the tier from their manifest or
  agent.

### 3.4 Services (step 4)

A service is a supervised thing with an id:
- `llm:<provider>`
- `image`
- `mcp:<name>`
- `module:<name>`
- `harness:<agent>`
- `runtimeAPI`
- `parlorServer`

State machine: `stopped → starting → running ⇄ idle → stopped`, and any state → `failed`. A service restarts on
crash with backoff (1 s, 5 s, 30 s); after 3 failures inside 10 minutes it stays `failed`. Reported cost:
resident memory (from `proc_pid_rusage` for children; MLX active memory for the in-process runtime), queue depth
and last used time.

---

## 4. Runtime API (step 6)

An OpenAI-compatible endpoint at `127.0.0.1:<runtimeAPI.port>/v1`. Every request needs
`Authorization: Bearer <token>`. It is off unless enabled; launching an external harness starts it for that
harness's token.

- `GET /v1/models`: the installed language models of every enabled provider, as `provider/repo`.
- `POST /v1/chat/completions`: messages, `tools` (function calling), `temperature`, `top_p`, `max_tokens`,
  `stream`. Streaming is SSE with `data: {chunk}` lines and `data: [DONE]`. Requests map onto `LLMRequest` and go
  through the same queue (priority: a harness's foreground turn = user priority) and prefix-matched cache slots
  as the app's own.
- **Errors** are OpenAI-shaped `{error:{message,type}}`: 401 for a bad token, 404 for an unknown model, 503 when
  the model can't be loaded within the memory budget.
- **Not supported:** embeddings, images, audio, `n > 1`, logprobs. These return 400 with a reason.

---

## 5. Provider spec (step 6; the MLX provider is shaped to it from step 2)

```swift
public protocol LanguageModelRuntime: Sendable {          // V1 shape, kept
    func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error>
    func loadedModel() async -> String?
    func unload() async
}
public protocol ModelProvider: LanguageModelRuntime {
    var id: String { get }
    var locality: Locality { get }                         // .local | .remote
    func models() async throws -> [ProviderModel]           // id, capabilities
}
```

Provider entries in `settings.json → providers`:

| Field | Meaning |
|-------|---------|
| `id` | unique; `mlx` is reserved for the built-in provider |
| `kind` | `mlx` or `openai-compatible` |
| `baseURL` | for `openai-compatible` |
| `keyRef` | Keychain item name; never the key itself |
| `models` | optional list; else from `GET /models` |
| `locality` | `local` or `remote`. Remote providers start disabled, show a badge, and mark the header of any agent using them |
| `service` | optional command to run as a service (e.g. a llama.cpp server), started on demand |

Semantics every provider MUST meet: streamed text arrives as `LLMEvent.text`, tool calls as `LLMEvent.toolCall`,
and one `LLMEvent.stats` comes at the end. `cachedTokens` is 0 when the provider has no prompt cache.
Cancellation stops generation promptly.

**MLX specifics (step 2):**
- Several resident models, LRU, under `memoryBudgetFraction` × physical RAM; the budget counts services too.
- `cacheSlotsPerModel` prompt-cache slots per model. A request takes the slot with the longest common token
  prefix. A miss evicts the least recently used slot.
- `useCache: false` requests (summaries) never touch the slots.
- The KV-quantisation and attention-window rules from V1 still disable reuse.

---

## 6. Conformance tests

Each lives beside the existing core tests in `Tests/CoreTests.swift` (or a sibling file in that target) and
runs on the scripted runtime (`DemoRuntime`, or a test-local scripted runtime when exact requests must be
captured), with no model download.

| Test | Asserts |
|------|---------|
| `turnBuiltInMatchesV1` | for fixed transcripts, the built-in harness produces byte-identical `LLMRequest.messages` to V1 `buildHistory` (system prompt, history slicing after a summary) |
| `turnEventOrder` | assistant with tool calls → toolStarted → tool message per call, in order; the last step offers no tools |
| `turnAllowlist` | a disallowed tool call yields the V1 "not enabled" error output, via `ToolExecutor` |
| `turnCancelKeepsPartial` | cancel mid-stream appends the partial text as an assistant line |
| `compactionAsEvent` | over threshold, the harness emits `summary` before replying, and the app persists it plus the marker |
| `settingsResolution` | agent > model defaults > machine; context clamp; V1 agent files resolve to V1 values |
| `logSpeakerCompat` | V1 lines without `speaker` load; new lines carry it; prune keeps unknown keys |
| `agentFileRoundTrip` | unknown front-matter keys survive a save |
| `toolsSnapshotPersists` | the "Tools updated" marker appears after a simulated relaunch |
| `seedOnce` | deleting every agent and reloading does not recreate Mabel and Pip |
| `conversationDefault` | no `conversation.json` → userDriven 1:1; a present file is parsed (scheduler unused until step 3) |
| later steps | `queueOrdering`, `prefixSlotSelection`, `roomAttribution`, `tierRefusal`, `harnessWireRoundTrip`, `runtimeAPISchema` |

Plus the existing 11 tests, the UI test and `parlor-cli bench`, all unchanged.

---

## 7. Step 1 contract: Conversation + Turn refactor (for execution)

**Goal:** the same app, restructured along §2. No visible change except three fixes: tool snapshots survive a
relaunch, seeding happens once, and agent files keep unknown keys.

**Do:**
1. `Turn.swift`: `TurnInput`, `TurnContext`, `TurnEvent`, `ToolExecutor`, `Harness` (§2.1–2.2).
2. `BuiltInHarness` (rename `SimpleLoop`; keep `AgentLoop` as a deprecated typealias only if needed to compile)
   containing:
   - the loop;
   - compaction (moved from `Room.compactIfNeeded` / `compact`);
   - history building (moved from `Room.buildHistory`, as a `static func buildRequestMessages(_ input:)` so the
     conformance test can call it).
   `Compactor` stays a helper it uses.
3. `HubToolExecutor`: wraps `ToolHub`. It moves the allowlist check, timeout and clip out of the loop.
4. `Settings.swift`: `ResolvedSettings.resolve` (§2.4). `ModelMetadata` reads `.parlor-model.json` including the
   optional `defaults` / `capabilities`. `Agent` gains optional override fields (§1.1, step 1 rows) and an
   `extra: [String: Frontmatter.Value]` pass-through.
5. `Conversation.swift`: `Participant`, `SchedulerSpec`, `ConversationConfig.load(dir:)` with the §1.4
   default; a `Scheduler` protocol with `UserDriven` only.
6. `Room` becomes the conversation's state holder. It keeps:
   - the observable UI state, the log, the summary persistence and the prune operations;
   - `send` / `stop` / `compactNow`.
   Its `turn` builds a `TurnInput`, calls `deps.harness.run`, and handles `TurnEvent`s: it sets `speaker` on
   appended lines, persists `summary` events with the V1 marker text, and updates stats. `compactNow` calls the
   harness's compaction directly (`BuiltInHarness.compact(input:context:)`). The type name `Room` stays until
   step 3.
7. `ChatMessage.speaker: String?`, plus pass-through of unknown keys on rewrite (decode into a stored
   `[String: JSON]` of extras).
8. `tools.json` snapshot (§1.5), replacing `lastKnownTools() -> nil`.
9. The `.parlor-seeded` marker. `seedIfEmpty` creates the starter agents only if the marker is absent, then writes
   it. On an existing V1 home with agents present, write the marker without seeding.

**Don't:**
- touch the UI beyond what compiles;
- change the system prompt text, the cache behaviour or the log format beyond `speaker`;
- add schedulers other than `UserDriven`;
- start step 2.

**Gate:**
- all existing tests plus the §6 step-1 rows pass;
- the UI test passes;
- `parlor-cli bench` shows the same cache hit pattern as before (turns 2–4 mostly cached);
- relaunching the real app against `~/parlor` shows every existing chat intact;
- `git diff` on a V1 agent file after opening and saving it in the editor changes nothing.

**Never** `rm -rf` anything under `~/parlor`. Tests use a temporary `PARLOR_HOME`.

---

## 8. Settled defaults for the open decisions

`PLAN-V2.md` left five decisions open. This spec assumes the recommended answers:
1. Spec first.
2. Rooms (step 3) before services and the command layer.
3. Cache and residency capped by memory budget.
4. The image worker becomes an MCP server.
5. Tools folds into System.

It also settles the two Home-layout gaps raised after the plan:
- app-owned agent memory (`memory/<id>.md`);
- automatic git history for undo.

Both are reserved now and built in step 7. Overriding any of these changes the sections that name them.
