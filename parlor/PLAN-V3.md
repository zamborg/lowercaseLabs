# parlor V3: engines → hosts → agents

**What parlor is:** a local Mac app for building small character agents and assistant experiences. Underneath it is
a local, ML-first API host:

- You mount **engines**.
- You attach **models** to them as **hosts**, which speak standard APIs.
- You build **agents**. Each agent is a **harness** using those hosts and some **tools**.

Engines, harnesses and tools are folders with a manifest, so adding a new kind of thing is a drop-in, not new core
code.

**The principle: built-ins get no back doors.** Everything parlor ships uses the same public seams a user
extension would:

- the harness protocol,
- engine adapters,
- manifests,
- the gateway,
- control verbs.

If a built-in needs something a seam can't carry, the seam is wrong, so fix the seam. That is what makes every
seam solid: our own features exercise it every day.

**Three patterns carry the whole design.** Every feature is built from them, and adding something new means using
one of them, not inventing a fourth.

| Pattern | One type | Owns | To add something new |
|---------|----------|------|----------------------|
| **Services** | `Supervisor` + `ServiceSpec` | every process: hosts, harnesses, tool servers. Start, ready check, idle stop, restart, logs, memory | describe it as a `ServiceSpec`; it gets the dashboard, `services.*` verbs and restart policy for free |
| **Adapters** | `protocol Adapter` (one per wire protocol) | turning a backend's protocol into the gateway's standard API | a backend that speaks an existing protocol needs only a manifest. A new protocol needs one new adapter |
| **JSON nouns** | `protocol Noun: Codable` + `NounStore<T>` | files on disk, CRUD-plus-validate verbs, schema, file watching | a new noun is one struct, one schema constant and one `validate`; the store gives it files, verbs and `schema.get` |

```swift
protocol Noun: Codable, Identifiable where ID == String {
    static var folder: String { get }          // "hosts", "agents", "harnesses", "engines"
    static var schema: JSON { get }            // hand-written JSON Schema, returned by schema.get
    var version: Int { get }
    var extra: [String: JSON] { get set }      // unknown keys, kept on save
    func validate(in catalog: Catalog) -> [Issue]   // references and cross-field rules
}
// NounStore<T: Noun>: load folder → [T]; list/read/create/update(merge patch)/delete/validate;
// atomic writes; FSEvents reload; Issue = {code, path, message, hint}
protocol Adapter: Sendable {
    func chat(_ req: ChatRequest, host: HostInfo) -> AsyncThrowingStream<ChatEvent, Error>   // optional per capability
    func image(_ req: ImageRequest, host: HostInfo) async throws -> ImageResult
    func service(for host: HostInfo) -> ServiceSpec?                                          // nil: in-process or remote
}
```

Written by Opus for Sonnet to execute, step by step, with gates. V3 is a fresh start: existing agents, chats and
provider entries in `~/parlor` are discarded, not migrated (Zubin, 2026-11-15).

## Goals, and how V3 meets each

| # | Goal (Zubin's words) | How | Proven by (gate) |
|---|----------------------|-----|------------------|
| 1 | A solid runplane: add models and text generators, exposed and hosted through standard APIs | Engines (manifest + adapter) serve models as hosts. Every host is reachable at the gateway's `/v1/chat/completions` by its host id | Step 3: `curl` chats with an MLX host and a llama.cpp host. Step 5: LM Studio or Ollama added by manifest alone |
| 2 | The same for image generation | `/v1/images/generations` on the same gateway, over the `command` (sd.cpp) and `mcp-stdio` (Z-Image) adapters. The `image.generate` tool is a thin client of it | Step 3: `curl` gets an image from Z-Image and from an SD checkpoint |
| 3 | Agents and harnesses that are launchable, like `character.py` | Every agent is a harness process, including the default assistant (`chat.py`). Harness manifests carry a config schema and starter templates; the editor renders the form | Step 4: Mabel on `chat.py` and Wren on `character.py` work, and compaction, Stop and rooms all work over the protocol |
| 4 | Tools to manage the system's state ourselves | Control verbs over `/mcp`, regrouped by the nouns (`engines`, `models`, `hosts`, `harnesses`, `agents`…), with tiers. UI, CLI and agents call the same verbs | Step 6: Claude Code with a `configure` token creates a host and an agent, then chats with it, with no UI touched |
| 5 | Fix bugs like models not showing as downloaded | One `Catalog` answers every availability question. Nothing else checks paths or parses model names | Step 2: per-format tests; no "not downloaded" for an installed GGUF or SD model |
| 6 | Reasonable modules, so new things and services are easy | Three manifest types (engine, harness, tools) and four adapters. One `Supervisor` runs every process the same way | Step 5: a new engine and a new harness, from manifests and the guides alone, with no Swift change |
| 7 | Buttons and settings exposed in the UI by what we build (a "Gen image" button) | Harness manifests declare `actions` (buttons in the chat composer or agent settings). Engine and harness manifests declare settings schemas that render as forms. The UI hard-codes neither | Step 4: Wren's "Gen image" and "Selfie" buttons appear from `character`'s manifest alone. "Compact now" is the `chat` harness's own action |
| 8 | An agent in the app uses the parlor controls to make new agents, cleanly | Every noun is plain JSON with one shape (file = verb input = verb output), uniform verbs, `schema.get`, `validate` with fixable errors, and `agents.try`. A seeded **Builder** agent uses them | Step 6: in chat, Builder turns "a pirate who draws herself" into a working agent on the first `create`, with `try` shown before saving. A bad draft gets every error back at once, with paths |
| 9 | One superadmin pane to tune and optimise the text and image stack per model | **Tuning** screen: every host's engine settings, request defaults, machine knobs, live stats and a benchmark button, in one place | Step 6: change Stheno's context and the SD checkpoint's steps from Tuning, run `hosts.bench` on both, and see the before/after numbers |

## Why: the tendrils in V2

| Symptom | Cause | V3 answer |
|---------|-------|-----------|
| Models show "not downloaded" when they are | Availability is answered in seven places, each with its own idea of a model name | Catalog |
| Downloading a GGUF edits `settings.json` | Derived state stored as config (`registerGGUF`) | Auto hosts are computed, never written |
| Text and image are different worlds | Text goes runtime → router; image is a *tool* that branches on file format | Both are host capabilities behind the gateway |
| Five process supervisors | Provider process, image worker, MCP connection, harness process, loopback servers, plus an id-prefix `switch` | One Supervisor |
| `model:` is ambiguous | It holds a repo id, `provider/model` or `mlx/org/name`, routed by guessing | Agents name host ids |
| Harness settings are untyped | Built-in knobs are `Agent` fields; `character.py`'s live in `extra` with no schema | Harness config schema |
| The built-in loop is privileged | It runs in-process with access the protocol doesn't carry (compaction, summary) | It becomes `chat.py`; the protocol gains what it needed |
| Two servers, two ports | Runtime API and Parlor MCP | One gateway port: `/v1/*` + `/mcp` |
| `Engine.swift` is 800 lines | Registry, supervisor, router, seeder and watcher at once | Split by layer (step 7) |

## The nouns

```
Engine      a backend that can serve a capability        mlx, llama.cpp, stable-diffusion.cpp, z-image, openai-remote
  └─ Host   an engine serving a model at the gateway     "stheno" = llama.cpp + L3-8B-Stheno GGUF + flags
Model       weights on disk, with a format               models/<org>/<name>/ + .parlor-model.json
Harness     a program that runs an agent's turns         chat (chat.py), character (character.py)
  └─ Agent  a harness + persona + hosts + tools + config   agents/wren/ (agent.json, persona.md, memory.md)
Tools       MCP servers an agent may call                image.generate, parlor.*, mcp.json mounts, modules/
```

### Gateway

The gateway is an in-process Swift API (`Gateway`) with an HTTP front on `127.0.0.1:<settings.gateway.port>`.

**Routes:**
- `GET /v1/models` lists hosts and their capabilities.
- `POST /v1/chat/completions` (OpenAI: streaming, tools).
- `POST /v1/images/generations` (OpenAI shape plus `negative_prompt`, `seed`, `steps`, `shape`). Images are written
  to `media/YYYY-MM/`, and the response carries `data[].url = "file://…"` and `data[].parlor_path`.
  `response_format: "b64_json"` is honoured for outside apps that expect it.
- `/mcp`: control verbs, same port, same tokens.
- Later capabilities (`/v1/embeddings`, `/v1/audio/speech`, `/v1/audio/transcriptions`) are a new capability name
  plus a adapter mapping, with no new plumbing.

**Rules:**
- `model` is a host id, or `<host>/<model>` for a multi-model remote host.
- **Per-host queue.** Priority is the `X-Parlor-Priority` header (`user` | `room` | `background`, default `user`). A
  harness sends `background` for summaries.
- **Cancellation:** a client disconnect cancels its generation. Gate it: Stop in chat frees the host within 1 s.
- **Memory budget.** Before starting a host, the gateway checks resident cost against `memoryBudgetFraction`. If it
  is over, it stops idle hosts (LRU) first. It refuses with 503 and a reason if still over.
- The prompt cache (prefix-matched slots) stays inside the MLX adapter. Any client that keeps a stable prefix gets it.
- **Sampling parameters.** `temperature`, `top_p`, `min_p`, `top_k`, `repetition_penalty` and `max_tokens` are
  request parameters.
  - Each engine manifest lists the ones it honours in `params`. The adapter passes those through and drops the
    rest.
  - `stats.ignored` names any that were dropped, and Tuning shows the list per engine.
  - **Resolution:** the request value, else the host's `defaults`, else the engine's defaults. Harness config
    schemas never put a `default` on these keys. A harness forwards only what the agent set, so host tuning
    actually applies.

### Engines (`engines/<id>/engine.json`)

Built-ins ship in the app bundle and are seeded with the marker pattern (`parlor-engines-v1`, rewritten when marked
and different). A user engine is just a folder.

```json
{ "version": 1, "id": "llama.cpp", "title": "llama.cpp",
  "capabilities": ["chat"], "formats": ["gguf"], "adapter": "http-openai",
  "launch": { "command": "llama-server",
              "args": ["-m", "{model.file}", "--alias", "{host}", "--port", "{port}", "--jinja",
                       "-c", "{settings.context}", "-ngl", "99", "{settings.args...}"] },
  "health": "/v1/models",
  "params": ["temperature", "top_p", "min_p", "top_k", "repetition_penalty", "max_tokens"],
  "requires": { "binary": "llama-server", "hint": "brew install llama.cpp" },
  "settings": { "type": "object", "properties": {
      "context": { "type": "integer", "default": 8192, "minimum": 512 },
      "args": { "type": "array", "items": { "type": "string" }, "default": [] } } },
  "lifecycle": { "idleMinutes": 30, "startupSeconds": 600 } }
```

**Adapters** are the only Swift a backend needs:

| Adapter | Meaning | Engines |
|--------|---------|---------|
| `inprocess-mlx` | today's `MLXRuntime` (queue, slots, residency) | mlx |
| `http-openai` | a child or remote server that already speaks OpenAI; the gateway proxies | llama.cpp, openai-remote, LM Studio, Ollama |
| `command` | one templated process per request (`{prompt}`, `{negative}`, `{out}`, `{width}`, `{height}`, `{seed}`, `{steps}`) | stable-diffusion.cpp |
| `mcp-stdio` | a long-lived MCP worker whose tool is mapped onto the API (`launch` + `"tool": "generate"`) | z-image (`parlor-imagegen`, unchanged) |

**Template rules:**
- `{x...}` splices a list.
- A `?` suffix marks a value as optional. When it is unset, the value and the flag just before it are both
  dropped: `["--negative-prompt", "{negative}?"]`.
- Values are passed as argv, never through a shell.
- The `requires.binary` lookup checks the manifest's folder, the app bundle's `Resources/`, `~/.local/bin`,
  `/opt/homebrew/bin` and `/usr/local/bin`, in that order. That replaces `llamaServerPath` and
  `SDCppImage.binaryURL`.

Moving Z-Image to HTTP later is a manifest edit (`adapter: http-openai`), not a code change.

### Models and formats

`.parlor-model.json` holds `{repo, kind, format, file?}`. The format is decided at download and never guessed later:

| Format | What it is | Engine |
|--------|-----------|--------|
| `mlx-lm` | MLX safetensors language model | mlx |
| `gguf` | llama.cpp weights (one quant, shards together) | llama.cpp |
| `sd-checkpoint` | single-file SDXL-family safetensors + `parlor-image.json` | stable-diffusion.cpp |
| `z-image` | Z-Image MLX folder | z-image |

An engine matches a model when `engine.formats` contains the model's format. Downloads keep today's choosers
(`chooseGGUF`, `chooseCheckpoint`) and gated-repo hint.

### Hosts (`hosts/<id>.json`)

```json
{ "version": 1, "id": "stheno", "engine": "llama.cpp", "model": "bartowski/L3-8B-Stheno-v3.2-GGUF",
  "settings": { "context": 8192 },
  "defaults": { "temperature": 1.17, "min_p": 0.075, "repetition_penalty": 1.1 },
  "enabled": true }
```

- **Auto hosts.** Every installed model whose format has exactly one engine gets a host computed in memory and
  shown as *auto*. Its id is `slug(shortName)`, with `-2`, `-3`… for collisions, assigned in sorted repo order so
  ids are stable. Auto hosts are never written. Editing one writes `hosts/<id>.json`, which then wins over the
  auto host.
- **Remote hosts** (`openai-remote`) carry `url`, `keyRef` (Keychain) and optional `models`. They serve
  `<host>/<model>` and start with `enabled: false`. They are badged in the editor and the chat header.
- **Port** is assigned by the gateway from 8100 upward at start. It is never stored.
- **Rule:** a host is how a model *runs* (`settings`: memory, context, launch flags) and how it is asked by default
  (`defaults`: sampling). An agent's config overrides `defaults` per agent. See "Sampling parameters".

### Catalog: the single answer

`Catalog` is a pure value, rebuilt from `models/`, `engines/`, `hosts/`, `harnesses/` and the Supervisor's snapshot.
It is the only code that decides availability.

- `hosts(capability:)` returns `[HostInfo]`, which feeds pickers.
- `status(host:)` is one of `ready`, `running`, `starting`, `off`, `missing model`, `missing binary (<hint>)`,
  `unknown engine`, or `failed (<why>)`.
- `status(harness:)` is `ready`, `missing binary (python3: install the Command Line Tools)` or `bad manifest (<why>)`.
- `problem(agent:)` returns the first blocking problem across its harness, its required hosts and their statuses, or
  nil.
- `models` lists each installed model with format, size and the hosts that use it.

The Library screen, the agent editor, the chat header, `/v1/models` and `hosts.list` read the Catalog and nothing
else. Gate: a test asserts no other type calls `installedDirectory` or parses a model name.

### Harnesses (`harnesses/<id>/harness.json` + script)

Every agent runs a harness process. There is no in-process loop. Shipped harnesses are seeded with the marker
pattern, and a user harness is a folder.

```json
{ "version": 1, "id": "character", "title": "Character",
  "command": "python3", "args": ["character.py"],
  "needs": { "chat": "required", "image": "optional" },
  "config": { "type": "object", "properties": {
      "temperature": { "type": "number", "minimum": 0, "maximum": 2, "description": "blank uses the host's default" },
      "max_reply_tokens": { "type": "integer", "minimum": 16 },
      "appearance": { "type": "string", "format": "multiline" },
      "scenario": { "type": "string", "format": "multiline" },
      "images": { "enum": ["auto", "ask", "off"], "default": "auto" },
      "image_every": { "type": "integer", "default": 4, "minimum": 1 } } },
  "templates": [ { "title": "Character", "persona": "…", "tools": ["image.generate"], "config": {} },
                 { "title": "Scene", "persona": "…", "config": { "scenario": "…" } } ] }
```

- **`parlor_harness.py`** is a seeded helper, like `parlor_module.py`. It owns the protocol loop, the gateway client
  (streaming, cancel on `turn.cancel`), `call_tool`, history framing for rooms, and compaction helpers. `chat.py` and
  `character.py` are small programs on top of it.
- **`chat.py`** is the V2 built-in loop, ported:
  - system prompt order;
  - "Today is <date>";
  - tool paragraph;
  - summary section;
  - `max_steps` with no tools on the last step;
  - result clipping (2/3 head, 1/3 tail);
  - room framing (`<Name>: ` prefixes, "Reply only as X");
  - compaction at `compact_at × context`, cut at a user message, `keep_recent` verbatim.

  Its config schema is today's `AgentOverrides` plus `context`, `compact_at`, `temperature` and `thinking`.
- **Config schema subset** (validated by the app on save, in `agents.update`, and rendered as the editor form):
  - `type` (string, number, integer, boolean, array of string);
  - `enum`, `default`, `minimum`/`maximum`;
  - `format: multiline`;
  - `description`.

  Unknown keys in an agent's `config` are kept and passed through. `validate` reports them as `warning` issues, not
  errors, so a typo is visible but never blocks a save.
- **Protocol v2** (SPEC §2.5, bumped). It must carry everything the old built-in used.

  app → harness:
  - `turn.start` adds:
    - `hosts`: `{"chat": {"model": "stheno", "baseURL": "http://127.0.0.1:<port>/v1"}, "image": {…}}`;
    - `config` (resolved: schema defaults ← agent config);
    - `summary` (`{through, text}`, or null);
    - `lastPromptTokens`;
    - `action` (null for a user message; see Actions).

  harness → app:
  - `summary {turn, through, text, folded}`: the app writes `summary.md` and the "Earlier messages summarised"
    marker. The app still owns the log and the summary file; the harness owns the decision and the text.
  - Unchanged: `text.delta`, `status`, `message`, `tool.call`/`tool.result`, `stats`, `turn.done`, `error`.
  - env: `PARLOR_GATEWAY_URL`, `PARLOR_TOKEN`, `PARLOR_AGENT`, `PARLOR_WORKSPACE`, `PARLOR_HOME`. Drop
    `OPENAI_BASE_URL` and `PARLOR_MCP_URL` duplicates, or keep them as aliases of the gateway URL.
- **Actions: buttons declared by the harness.** The manifest lists `actions`, and the UI renders them; nothing is
  hard-coded per harness.

  ```json
  "actions": [
    { "id": "image", "title": "Gen image", "icon": "photo", "placement": "composer", "needs": "image",
      "hint": "Picture the current scene" },
    { "id": "selfie", "title": "Selfie", "icon": "camera", "placement": "composer", "needs": "image" },
    { "id": "reroll", "title": "New look", "icon": "dice", "placement": "settings",
      "args": { "type": "object", "properties": { "seed": { "type": "integer" } } } } ]
  ```

  - `placement` is `composer` (a button by the message box), `message` (in a reply's menu, sent with that message's
    id) or `settings` (in the agent editor).
  - `needs` hides the button when the agent has no host for that need.
  - `args` (schema subset) opens a small form first.
  - **Clicking sends `turn.start` with `action: {id, args, messageID?}`** and no user line. The app logs a small
    marker ("▸ Gen image") so the transcript shows why the turn happened. From there it is an ordinary turn: deltas,
    tool calls (the image via `image.generate`) and messages.
  - The same verb serves every front end: `conversations.action {conversation, agent, action, args}`. An agent with
    tier `act` or an outside client can press a button too.
  - **The built-in buttons go through the same seam.** `chat`'s manifest declares
    `{ "id": "compact", "title": "Compact now", "placement": "settings" }`, so the separate `turn.compact` message
    is gone. `parlor_harness.py` gives each harness `@action("compact")`-style handlers.
- **Tools still run in the app.** `tool.call` → allowlist and tier → `tool.result`, logged by the app. This remains
  the one enforcement point.
- **Scoped tokens.** A harness's token may call the gateway only for the hosts its agent names. A Python harness
  therefore can't reach around the tool allowlist to other models. `/mcp` verbs are gated by the agent's tier as
  today.
- **Process model:** one long-lived process per agent, one turn at a time (today's `TurnLock`). Stopped after
  `idleMinutes` (default 30) by the Supervisor. Crash mid-turn: keep the partial text, log an error line, apply the
  restart policy.

### Agents (`agents/<id>/agent.json` + `persona.md`)

Agents are plain JSON like every other noun. The persona is a separate Markdown file, because it is long prose that
people and models edit as text.

```json
{ "version": 1, "id": "wren", "name": "Wren", "description": "courier with a secret",
  "harness": "character",
  "hosts": { "chat": "stheno", "image": "wai-illustrious" },
  "tools": ["image.generate"], "tier": "none", "group": "Characters", "color": 3,
  "config": { "temperature": 1.17, "max_reply_tokens": 600, "images": "auto",
              "appearance": "a woman in her late twenties, ..." } }
```

- The V2 front-matter format (`agents/<id>.md`) is gone, and so are `Frontmatter.swift` and the YAML-ish parser.
  This is the fresh start, so nothing needs reading.
- `harness` names a harness id. An inline command (`"harness": {"command": "python3 /path/x.py"}`) is still accepted
  for quick experiments, with no schema form.
- The agent's folder also holds `memory.md` (was `memory/<id>.md`), so one agent is one folder you can copy.

### API design: one shape, self-describing

The builder agent (below) creates agents through this API, so the API has to be easy for a model to get right on
the first try. These rules apply to every noun: engine, host, harness, agent, tool mount and settings.

1. **One shape everywhere.** The JSON in the file is the JSON a verb takes and returns, and the JSON over `/mcp` and
   the CLI.
   - Each noun is one `Codable` Swift struct with `version`, `id` and an `extra: [String: JSON]` that keeps unknown
     keys on save.
   - Loading a folder means decoding its JSON into structs. There are no other parsers.
2. **Uniform verbs per noun:** `list`, `read`, `create`, `update` (JSON merge patch), `delete` and `validate`.
   - `validate` takes a draft and writes nothing.
   - `create` and `update` run `validate` first and refuse with the same errors.
3. **Self-describing.**
   - `schema.get {noun}` returns the JSON Schema for that noun.
   - `harnesses.read` includes its `config` schema and `actions`; `engines.read` includes its `settings` schema.
   - An agent never has to guess a shape; it asks.
   - Schemas are hand-written constants beside the structs. A test decodes every seeded file and validates it
     against its schema, so they cannot drift.
4. **Errors a model can fix:** `{"code": "invalid", "path": "config.images", "message": "must be one of auto, ask,
   off", "hint": "harnesses.read character lists the options"}`. They are returned as a list, all at once, never
   only the first.
5. **References are ids.** References are checked on `validate`:
   - `harness` must exist;
   - every required `needs` entry must have a host of that capability;
   - every tool pattern must match something.

   A missing reference is an error that includes the available ids. When the agent can just pick the obvious one,
   for example one image host, the error hint names it.
6. **Try before save.** `agents.try {draft, message}` runs one turn of a draft agent in a throwaway conversation and
   returns the reply. Nothing is written. The builder can test a character before creating it.

### The builder agent

A seeded agent, **Builder**, on the `chat` harness with tier `configure`. This is the main way new agents get made.

- **Tools:**
  - `parlor.schema.get`;
  - `parlor.harnesses.*`;
  - `parlor.hosts.list`/`read`;
  - `parlor.models.list`;
  - `parlor.agents.templates`/`validate`/`try`/`create`/`update`;
  - `parlor.tools.list`;
  - `image.generate` (to show you a character's look while designing it).
- **Its persona is a short procedure:**
  1. Ask what you want.
  2. Pick a harness from `harnesses.list`.
  3. Read its schema.
  4. Pick hosts from `hosts.list` by capability.
  5. Draft the JSON and `validate` it.
  6. `try` one line in character and show you the result.
  7. `create` it.
  8. Tell you where to find it.
- **Permissions, replacing git undo:**
  - `create` is free.
  - `update` and `delete` of an agent the caller did not create in this conversation show you a diff in the app
    and wait for a yes. That is the destructive-confirm mechanism, extended.
  - Creating a harness or engine (code that will run) always asks, with the manifest and script shown. Module
    approval works the same way.
- **New harnesses too.** For a character style no harness covers, Builder writes `harnesses/<id>/harness.json` plus
  a script on `parlor_harness.py`, following `harnesses/CLAUDE.md`. That needs your approval before first run.

### Tools

- `image.generate` is a thin tool. It calls `Gateway.image(host: agent.hosts.image, …)` and words the result for the
  model (today's text). `ImageProvider`'s format branching and worker management are deleted, because the adapter
  owns them.
  - The host comes from the *calling* agent on every call. That also fixes V2's single, global
    `ImageProvider.setModel`, under which two agents in a room drew with whichever model was set last.
- `mcp.json` (Claude Desktop format) and `modules/` stay as they are. Their processes run on the Supervisor.

### Supervisor

One `Supervisor` actor owns every child process: hosts, harnesses, MCP servers and modules.

```swift
struct ServiceSpec {
    var id: String                 // "host:stheno", "harness:wren", "mcp:fs", "module:dice"
    var kind: Kind                 // host | harness | tools
    var launch: Launch?            // argv, env, cwd; nil for in-process (MLX)
    var ready: Ready               // .http(URL) | .stdio | .none
    var lifecycle: Lifecycle       // .always | .onDemand(idleMinutes:) | .perRequest
    var restart: RestartPolicy     // backoff 1/5/30 s, failed after 3 in 10 min
}
```

- **Surface:** `ensureRunning`, `stop`, `restart`, `snapshot` (state, pid, resident memory, last used, log URL,
  restarts).
- The System dashboard and `services.*` verbs are a projection of `snapshot`, with no id-prefix switch.
- **Logs** go to `logs/<id>.log` (1 MB × 3), for the user to open. The executor never reads them (see Rules).
- In-process MLX registers as a virtual service whose start and stop are load and unload.

### Control plane

The rule from V2 stays: state is files, actions are verbs, and the UI, `/mcp` and `parlor-cli` call the same verbs.

Every JSON noun gets `list`, `read`, `create`, `update`, `delete` and `validate` from `NounStore`. The table lists
only the verbs beyond those.

| Group | Noun? | Verbs beyond CRUD + validate |
|-------|-------|------------------------------|
| `schema` | | get |
| `engines` | yes | (create and update always ask: code that runs) |
| `models` | | list, search, download, pause, delete |
| `hosts` | yes | duplicate, enable, start, stop, test, bench, stats |
| `harnesses` | yes | (create and update always ask, with manifest and script shown) |
| `agents` | yes | templates, try, duplicate |
| `conversations` | list, read, create, post, action, run, stop, prune, reset |
| `tools` | list, mount, unmount, enable |
| `services` | list, logs, restart |
| `settings` | get, set, reset |
| `memory` | read, append, replace |
| `home` | paths, reveal |

- `providers.*` and `settings.providers` are retired.
- Destructive verbs ask the user as today: delete, prune, reset, hosts.delete and models.delete.

### Screens

| Screen | Holds |
|--------|-------|
| Chats | 1:1 conversations and rooms |
| Agents | the editor: harness picker; one host picker per `needs` entry, filtered by capability and showing Catalog status; tools; tier; the config form from the schema; persona |
| Library | Models (search, download, installed, "used by"); Hosts (engine, model, status, start/stop, endpoint to copy, settings form, *auto* badge); Engines (installed backends, missing binaries with the install hint) |
| Tuning | the superadmin pane (below) |
| System | services, tool mounts and modules, tokens, gateway port, "Your data" |

Chat shows the agent's `composer` and `message` actions. The agent editor shows the `settings` actions under the
config form.

### Tuning: the superadmin pane

This is one screen for tuning the whole stack. Everything on it is a verb, so an agent or Claude Code can tune too.
The screen is laid out in three parts.

**Machine knobs (top strip):**
- memory budget;
- prompt cache on or off and slots per model;
- KV bits and quantised-KV start;
- prefill step;
- model unload timer;
- gateway port.

These are today's Performance console, moved here.

**Left: hosts, grouped Text and Image.** Each shows its status dot, resident memory and last speed.

**Right: the selected host.**
- **Engine settings**, a form rendered from the engine manifest's `settings` schema.
  - llama.cpp: context, GPU layers, threads, batch size, flash attention, extra args.
  - stable-diffusion.cpp: sampler, scheduler, steps (fast and detailed), CFG, CLIP skip, prediction, sizes per
    shape, quality-tag prefix, default negative, VAE.
  - z-image: precision, steps, sizes.
  - mlx: context cap.

  `parlor-image.json` is retired into these host settings. A save that needs a restart says so and offers
  "Restart now".
- **Request defaults:** temperature, top-p, min-p, repetition penalty and max tokens, used when an agent's config
  is silent. They are stored in the host file under `defaults`. Precedence: agent config, then host defaults, then
  engine schema defaults.
- **Live stats from the Supervisor and Gateway:**
  - text: tokens per second, time to first token, prompt-cache hit rate and resident memory;
  - image: seconds per image at each shape and resident memory.
- **Benchmark (`hosts.bench`):** a fixed short chat (three turns, so cache reuse shows) or a fixed image prompt at
  fast and detailed. Results go to `hosts/<id>.bench.json`, the last 10 runs with the settings that produced them,
  and show as a small before/after table. This is the one place a measurement is written to Home; it holds no
  prompts from real chats.
- **Duplicate host:** clone it with different settings, such as "stheno-long-ctx", to A/B two configs on the same
  model.

## Scope

**Kept and frozen.** No new work in V3; touched only where a rename forces it. Each keeps its tests:

- rooms and their schedulers (characters talking to each other suits the product);
- per-agent memory;
- tiers and tokens;
- the prompt cache.

**Cut in V3** (default; Zubin can veto). They aren't part of building character experiences, and each is a
standing source of surprise:

- **Triggers:** clock and file-change starts.
- **The autonomous scheduler.**
- **Git history:** `HomeHistory`, System → History, `home.history`/`home.restore`. It is no longer needed with no
  migration, and it is a background writer on every save.

Their verbs, UI and tests go too.

**Non-goals:**
- parallel generation or batching;
- any network beyond 127.0.0.1;
- sandboxing harnesses (they run with their workspace and token, as MCP servers do);
- sync;
- multi-user;
- bundling Python (later, with `uv`, if parlor is ever shipped to others).

## Fresh start (instead of a migration)

On first V3 launch, when `.parlor-v3` is absent, the app does this:

1. Moves `agents/`, `chats/`, `memory/`, `workspaces/`, `harnesses/`, `.git/` and `settings.json` into
   `~/parlor/old-v2-<date>/`. This is a move, not a delete, because it is cheap and Zubin can `rm` it.
2. Keeps `models/`, `media/`, `modules/`, `logs/` and `mcp.json` in place.
3. Seeds the engines, harnesses and starter agents. Mabel uses `chat` with the auto host of an installed MLX model.
   Wren uses `character` with the auto hosts of an installed GGUF and an SD checkpoint, when present.
4. Writes `.parlor-v3`.

Hand-made hosts like Stheno's sampler flags are recreated from `hosts.create` or the Library, not migrated.

## Order of work (Sonnet executes; each step ships and keeps the app usable)

1. **Supervisor.**
   - Add `Supervisor` and port the provider process, image worker, MCP stdio servers and harness processes onto it.
   - Rebuild `refreshServices` and `controlService` as projections.

   Gate:
   - all tests pass;
   - the dashboard shows the same rows;
   - a killed llama-server shows `failed`, then restarts on the next message.
2. **Formats, engines, hosts, Catalog.**
   - First revert the uncommitted `isModelAvailable` edits (`Engine.swift`, `AgentEditorView.swift`,
     `ChatView.swift`, `Tests/GGUFTests.swift`).
   - Add the format field at download, the five engine manifests, `HostRegistry` (auto and explicit) and `Catalog`.
   - Route all availability through `Catalog`.
   - Delete `registerGGUF`, `unregisterGGUF`, `syncGGUFProviders` and the library callbacks.
   - Build `Noun`/`NounStore` here, with `Host` and `Engine` as its first two nouns. Agents join in step 4.
   - Until step 4, V2 agents still say `model:`/`image_model:`. A temporary `LegacyRef.host(for:)` maps them:
     - repo → that model's auto host;
     - `provider/x` → the host made from that provider entry, in memory.

     It is deleted in step 4.

   Gate:
   - a test per format resolves to a host with the right status;
   - a test that `installedDirectory` has no callers outside `Catalog`/`ModelLibrary`;
   - the app shows no "not downloaded" for an installed GGUF or SD model.
3. **Gateway.**
   - One loopback server for `/v1/*` and `/mcp`.
   - Chat over the `inprocess-mlx` and `http-openai` adapters; `/v1/images/generations` over `command` and
     `mcp-stdio`.
   - `image.generate` calls the gateway. Add priorities, cancel on disconnect, the memory budget and scoped tokens.
   - Remove the Runtime API and its toggle; settings become `gateway {port}`.

   Gate:
   - `curl` with a token chats with an MLX host and a llama.cpp host;
   - `curl` gets an image from Z-Image and from an SD checkpoint;
   - Stop frees the host within 1 s;
   - a token scoped to host A gets 403 on host B.
4. **Harnesses only.**
   - Add `parlor_harness.py`, `chat.py` (a port of the built-in loop) and the `character.py` refactor onto the
     helper.
   - Add harness manifests (a `Noun`), protocol v2 (`hosts`, `config`, `summary`, `action`), and JSON agent
     folders (a `Noun`; delete `Frontmatter.swift` and `LegacyRef`).
   - Add the schema form, templates from manifests and harness actions. `character`'s manifest declares "Gen image"
     and "Selfie"; `chat`'s declares "Compact now".
   - Then delete `BuiltInHarness`, `Compactor`, `HarnessRouter` and `Templates.swift`.
   - Add the fresh start.

   Gate:
   - the V2 conformance cases rerun against `chat.py` over the wire: request messages match V2 for fixed
     transcripts, event order holds, and the allowlist refusal, cancel-keeps-partial and compaction-as-event all
     work;
   - Mabel and Wren chat;
   - a two-agent room keeps correct attribution;
   - Compact now works as `chat`'s action;
   - Wren's "Gen image" button draws the current scene with no user line logged, and the button vanishes when
     `hosts.image` is unset;
   - an action added to a harness manifest appears after a reload, with no Swift change;
   - editor round-trip keeps unknown keys.
5. **Drop-in proof.**
   - From the updated `engines/CLAUDE.md` and `harnesses/CLAUDE.md` alone, add an `openai-remote` engine pointed at
     LM Studio or Ollama.
   - Add a toy harness (an echo or a dice-rolling narrator).

   Gate: both appear in the Library and editor and work in chat with zero Swift changes.
6. **Verbs and screens.**
   - Regroup the verbs into the uniform CRUD-plus-`validate` shape. Add `schema.get`, `agents.try`, structured
     errors and the diff confirmation for agent-made updates. Seed Builder.
   - Build the Library tabs.
   - Build the Tuning screen: machine knobs, host list, engine settings forms, request defaults, live stats,
     `hosts.bench`, duplicate. Retire `parlor-image.json` into host settings and the Performance console into
     Tuning.
   - The agents editor gets harness and host pickers and the config form.
   - Remove the providers UI. Make the cuts (triggers, autonomous, history).

   Gate:
   - every UI action maps to a verb (list in SPEC);
   - the UI test passes;
   - Claude Code with a `configure` token runs `hosts.create`, `agents.create` and `conversations.post` end to end;
   - Builder in chat, on a real local model, creates a working character from one sentence;
   - a deliberately bad draft returns every error, with path and hint, in one `validate` call;
   - Builder's `update` of Mabel shows a diff and waits;
   - a test validates every seeded file against `schema.get`;
   - from Tuning, change Stheno's context and the SD host's steps, restart, and run `hosts.bench` on each. The
     before/after table shows both runs.
7. **Cleanup.**
   - Split `Engine.swift` into `Runplane/` (Supervisor, Adapters, Engines, Hosts, Catalog, Gateway), `Agents/`
     (Harness host, Conversations, Rooms) and `Control/` (Verbs, MCP).
   - Delete dead code.
   - Rewrite `SPEC.md` to V3 and update `README.md` and the seeded guides.
   - Mark `PLAN-V2.md` as superseded.

   Gate:
   - tests pass;
   - `parlor-cli bench` shows the same cache pattern through the gateway;
   - `scripts/package.sh` and install to `/Applications` work, leaving one Parlor installed.

## Invariants (tests that guard them)

1. Only `Catalog` decides availability.
2. Only `Supervisor` creates `Process` objects, including the `command` adapter's one-per-request runs
   (`.perRequest`).
3. Only `Gateway` calls adapters. Tools and harnesses go through it.
4. Only the app writes `log.jsonl` and `summary.md`. Harnesses send events.
5. Every tool call passes `ToolExecutor` (allowlist + tier).
6. Every UI action is a verb.
7. No file under `~/parlor` is written as a side effect of something else (no derived config).
8. Built-ins use only public seams: the shipped harnesses, engines and tools are manifest folders like a user's.
9. Every noun's file, verb input, verb output and wire JSON are one Codable type validated by one schema.
10. The UI hard-codes no harness or engine specifics. Buttons come from `actions`, forms from `settings` and
   `config` schemas.
11. Only `NounStore` reads or writes noun files. The only JSON files outside it are `mcp.json` (Claude Desktop
    format, kept for interoperability), `log.jsonl` and the model markers.

## Rules for the executor

- **Never read logs or user data**: chats, media, `logs/`, or anything under `~/parlor` beyond what a gate
  explicitly checks. **Never commit or push logs or data.** Stage only source under `parlor/`.
- Tests use a temporary `PARLOR_HOME`. The only change ever made to the real `~/parlor` is the fresh-start move,
  which the app does itself. Never `rm -rf` under it.
- After adding files, run `xcodegen generate`. Run tests with
  `xcodebuild test -scheme Parlor -destination 'platform=macOS'`, with long runs wrapped in
  `perl -e 'alarm N; exec @ARGV'`.
- Never `Process.waitUntilExit` on the main thread.
- Commit at each gate and push. If a gate fails, stop and report; don't paper over it.
