# parlor V3.1: finish the three patterns

> **Status: done (2026-11-16).** Steps 1–16 are committed; see the end-of-file notes and the final report for what was deliberately left.

V3 (steps 1–7, `PLAN-V3.md`) shipped the right shape: engines → hosts → harnesses → agents, one gateway, one
catalog, every agent a harness process. Two principal reviews (2026-11-16) found that each of the three patterns
(nouns, adapters, services) is only partly done, and the gaps are where the bugs are. V3.1 finishes them, and makes every noun first party in the app. No new features.

**The bar (Zubin, 2026-11-16):** at the end of V3.1 the app works, is solid and stable, and every primitive is right,
so the only feedback left is UI/UX, and extending it later means using a seam, not fixing one. "Done means" at the
end of this file is the acceptance checklist for that.

Written by Opus for Sonnet to execute, step by step, with gates. Fresh start still applies: there is no
migration and no back-compat. Delete old paths; do not keep shims.

## Standing rules (unchanged from V3)

- Never read logs or user data (`logs/`, chats, media, anything under `~/parlor`). Never push logs or data.
- Stage only source under `parlor/`. Tests use a temp `PARLOR_HOME`. Never `rm -rf` under `~/parlor`.
- Run `xcodegen generate` after adding or moving files. Wrap long runs in `perl -e 'alarm N; exec @ARGV'`.
- Never `Process.waitUntilExit` on the main thread.
- Each step: full unit suite (`xcodebuild -scheme Parlor -destination 'platform=macOS' test`) green, then commit
  `parlor V3.1 step N: …` and push. If a gate fails, stop and report; do not weaken a test to pass it.
- Real-model checks run with a temp `PARLOR_HOME` and `PARLOR_MODELS=$HOME/parlor/models`, binaries via
  `PARLOR_BIN_SD_CLI` / `PARLOR_BIN_PARLOR_IMAGEGEN` pointing into `/Applications/Parlor.app/Contents/Resources`.
- Keep `SPEC.md` true at every commit (`testEveryVerbInTheSpecExistsAtItsTier` parses it).

## What the audit found (evidence)

| # | Finding | Where | Effect |
|---|---------|-------|--------|
| A | Agents are not on the noun pattern. `agents.read/list` return V2 JSON (`model`, `context`, `compact_at`, `temperature`, `kind`) via `CommandCenter.agentJSON`; `agents.update` takes loose `fields`; only `create {draft}` and `validate` use the file shape | `Commands.swift` `agentJSON`, `CommandVerbs.swift` `fields`/`apply` | File ≠ verb input ≠ verb output. Builder copies the read shape into drafts |
| B | `Agent` keeps V2 shims over `config`/`hosts`: `model` (is the chat host id), `imageModel`, `temperature`, `contextTokens`, `compactAt`, `thinking`, `overrides`, `Kind` | `Agent.swift` | Two models of an agent; UI and verbs mix them |
| C | Host-id bugs from those shims: the chat chip's green dot compares `loadedModel` (a repo) with `agent.model` (a host id), so it never lights; Library "used by" compares `agent.model == m.repo`, so deleting a model never warns; Room view's mixed-model hint compares host ids as models | `ChatView.swift:102`, `ModelsView.swift:86`, `RoomView.swift:55` | Visible bugs |
| D | Sampling precedence is decided three times. `ResolvedSettings` folds machine `topP`/`maxReplyTokens` into `turn.start.settings`; `chat.py` `t.opt` always sends them; the gateway applies host `defaults` only for absent keys | `Resolved.swift`, `HarnessHost.swift` (`settings`), `HarnessPython.swift` (`opt`, request kwargs), `Gateway.swift` `chat` | A host's `top_p`/`max_tokens` defaults from Tuning never reach the `chat` harness |
| E | No `Adapter` protocol. Dispatch is a string switch in `Gateway.resolveChat`, another in `ImageProvider`, and `derivedProviders` turns hosts into `ProviderConfig` for `ProviderRouter`, which re-parses `"host/host"` strings, applies repo aliases and sends anything unknown to MLX | `Gateway.swift`, `ImageProvider.swift`, `EngineCatalog.swift`, `Adapters.swift` | The V2 "model is ambiguous" problem lives on inside |
| F | `ServiceSpec` is declared and never constructed. Idle stop and restart-with-backoff exist three times (`MCP.swift`, `ProviderProcess`, the image worker); harness processes have neither. The memory budget lives in the gateway and polls each owner for PIDs; `inprocess-mlx` skips it | `Supervisor.swift`, `MCP.swift`, `Adapters.swift`, `ImageProvider.swift`, `Gateway.admit` | One supervisor in name only |
| G | `hosts.stop` switches on engine ids (`"mlx"` unloads every model, `"z-image"` stops the image service, else `"provider:"+id`) | `CommandNouns.swift` hosts.stop | Built-ins have a back door; a user engine cannot be stopped properly |
| H | The `image.generate` tool calls `ImageProvider.generate` directly, skipping the gateway's scope, problem check and memory budget | `ImageProvider.swift` `call(tool:)` | Breaks "built-ins get no back doors" |
| I | SD sampling lives in `parlor-image.json` beside the weights, not in the host's `settings` | `SDCppImage.Config` | Two homes for "how this model runs"; Tuning can't see it |
| J | The UI mutates through engine methods (`engine.save`, `create`, `delete`, `duplicate`, `renameToMatchName`, `setServer`, `saveMCPConfig`, `createToken`, …), not verbs | `App/*.swift` | Two code paths with different validation |
| K | Agent ids follow the name on rename (`renameToMatchName`) | `EngineAgents.swift` | Noun ids should be stable |
| L | Vestigial: Swift `Harness` protocol (one impl), `TurnContext.llm` (unused by it), `Agent.Kind` | `Turn.swift`, `Agent.swift` | Suggests in-process paths that don't exist |
| M | 693 lines of Python inside Swift string literals | `Runplane/HarnessPython.swift` | Not lintable or testable as Python |
| N | Every spawn goes through `zsh -l -c`, including engines whose binary path is known | `Supervisor.Launch`, `EngineCatalog.derivedProviders` | Slow, environment-dependent, shell-quoted settings |
| P | Files dropped in by a person or a coding agent are not picked up. `filesChanged` reloads agents, `mcp.json`, modules, settings and chats, but not `engines/`, `hosts/` or `harnesses/`; they reload only when a verb runs or the app restarts | `Engine.swift` `filesChanged` | A new harness from Claude Code is invisible until restart |
| Q | A broken noun file vanishes silently: `NounStore.list()` is `ids().compactMap { read($0) }` | `Runplane/Noun.swift` | A typo in `harness.json` makes the harness disappear with no error anywhere |
| R | No UI for harnesses: Library has Models, Hosts and Engines only. There is nowhere to see every harness, where it came from, whether it works, which agents use it, or to edit, duplicate or delete one | `App/LibraryView.swift` | Harnesses are not first party in the app |
| S | Tools are not a noun. PLAN-V3 names three manifest types (engine, harness, tools), but MCP servers are entries in one `mcp.json` and modules have their own manifest and approval flow, with no `tools.*` CRUD-validate-schema verbs in the noun shape | `MCP.swift`, `Modules.swift`, `CommandVerbs.swift` tools.* | Two ways to add a tool, neither on the noun pattern |
| T | Each Library tab is hand-written per noun | `LibraryView.swift` `HostsTab`, `EnginesTab` | A new noun needs new UI code, against pattern 3 |
| O | Concurrency: Swift 5 mode, 22 `@unchecked Sendable`, 11 `NSLock`s, lock-in-async warnings in `HarnessHost` | ParlorCore | Debt for Swift 6 |

## Target shapes

```swift
// Nouns: agents join the others. One shape everywhere.
// agents.list   -> [AgentRecord JSON]              (same keys as agent.json; no persona)
// agents.read   -> AgentRecord JSON + "persona" + "memory"   (exactly what create accepts, plus memory)
// agents.create {draft} | {template, name}         (template expands server-side into a draft, same path)
// agents.update {id, patch}                        (merge patch over agent.json; "persona" key writes persona.md)
// Ids never change after create. Renaming changes `name` only.

// Nouns: one registry drives files, pickup, verbs, schema and the Library.
// NounKind(id: "harness", folder: "harnesses", layout: .folder("harness.json"), schema:, verbs:, runsCode: true)
// Registry: engine, host, harness, agent, tool (and conversation, read-mostly).
// NounStore.scan() -> [NounEntry<T>]   NounEntry = (id, value: T?, issues: [Issue], source: .seeded|.user, modified: Date)
//   A file that does not decode is an entry with value nil and its issues, never dropped.
// FolderWatcher: any change under a noun folder reloads that store (debounced), recomputes the catalog and the
//   UI updates. Seeded ("managed") files carry their marker; a user edits a copy (duplicate), never the managed file.

// Adapters: one per wire protocol. The gateway is the only caller.
protocol Adapter: Sendable {
    var id: String { get }                                   // "inprocess-mlx", "http-openai", "command", "mcp-stdio"
    func chat(_ req: LLMRequest, host: HostInfo, engine: EngineManifest) -> AsyncThrowingStream<LLMEvent, Error>
    func image(_ req: ImageRequest, host: HostInfo, engine: EngineManifest) async throws -> ImageResult
    func service(for host: HostInfo, engine: EngineManifest, catalog: Catalog) -> ServiceSpec?   // nil: in-process or remote
    func residentBytes(host: HostInfo) async -> Int64        // for the memory budget
}
// Unsupported capability: throw AdapterError.unsupported(capability) (the catalog already hides such hosts).

// Services: the Supervisor owns every process's lifecycle.
// ServiceSpec(id: "host:<id>" | "harness:<agent>" | "mcp:<name>" | "module:<name>", kind, launch, ready, lifecycle, restart)
// supervisor.ensure(spec) async throws   start if needed, wait for ready, count a user
// supervisor.release(id)                 drop a user; onDemand stops after idle
// supervisor.control(id, .start/.stop/.restart)
// supervisor.footprint() -> [id: bytes]  for the budget
// Restart: backoff [1, 5, 30] s, `failed` after 3 crashes in 10 min (one implementation, here).

// Sampling: decided once, in the gateway.
// request key (the harness sent it because the agent's config set it)
//   > host.defaults[key] > machine settings.llm fallback > engine default
// turn.start.settings carries loop knobs only: context, compact_at, keep_recent, summary_tokens,
// max_steps, tool_result_chars, tool_timeout. No sampling.
```

## Order of work

Each step is self-contained and leaves the app working. Do them in order.

### Step 8: the noun registry, live pickup, visible errors (P, Q)

1. `Runplane/NounRegistry.swift`: a `NounKind` per noun (engine, host, harness, agent; tool joins in step 13) with
   folder, layout, schema (`NounSchemas`), whether it runs code, and its reload hook. `schema.get` and the noun
   verbs read from it, so a new noun is one entry.
2. `NounStore.scan()` returns every file as an entry: the decoded value or `nil`, its `[Issue]` (decode errors with
   paths, `T.check`, schema check, cross-references via the catalog: unknown engine, missing host, missing
   `command`), `source` (`seeded` when it carries the managed marker), and modified date. `list()` stays for callers
   that want only valid values. Every `*.list` verb returns entries with `ok`, `issues` and `source`, so an agent
   sees a broken file instead of nothing.
3. `filesChanged` reloads by noun folder: `engines/`, `hosts/`, `harnesses/`, `agents/` (and later `tools/`), each
   debounced, then `reloadRunplane()` / `reloadAgents()`. Remove the agents-only special case.
4. Managed seeds (`chat`, `character`, built-in engines) are read-only to verbs: `update`/`delete` on a seeded noun
   fails with the hint "duplicate it first"; add `harnesses.duplicate` and `engines.duplicate` (copy the folder
   under a new id, drop the managed marker).
5. Folder guide and `harnesses/CLAUDE.md`: "drop a folder here and it appears in the app within a second; if it
   does not, `parlor.harnesses.list` shows why".

Tests: a harness folder written to disk appears in `harnessList` within 2 s with no verb call (same for an engine
and a host); a `harness.json` with a type error is listed with `ok: false` and an issue at its path; fixing the
file clears it; `harnesses.update` on `chat` is refused and `harnesses.duplicate chat → mychat` works and is
editable; an agent naming a broken harness shows `harness_invalid` in `agents.validate`.

Gate: suite green; real check: copy `harnesses/chat` to `harnesses/test-chat` with `cp -R` in a temp home while
`parlor-cli` waits on `harnesses.list`, and see it listed.

### Step 9: agents are a true noun (A, B, C, K)

1. `agents.list` / `agents.read` return `AgentRecord` JSON from `agentStore.raw(id)` (unknown keys included), plus
   `persona` (and `memory` on read). Delete `CommandCenter.agentJSON`.
2. `agents.update {id, patch}`: merge patch over `agent.json`; a `persona` key writes `persona.md`; validate the
   merged result with `agentIssues` before writing; keep the diff-confirm for callers that did not create it.
   Delete `fields`, `apply`, and the loose-field form of `agents.create`.
3. `agents.create {template, name}` builds a draft from the template (`harness`, `config`, `tools`, `group`,
   default hosts) and goes through the same draft path. The try-before-create guard for agent callers stays.
4. `Agent`: delete `model`, `imageModel`, `temperature`, `contextTokens`, `compactAt`, `thinking`, `overrides`,
   `AgentOverrides`, `Kind`, `kind`. Callers read `hosts["chat"]`, `hosts["image"]`, `config[...]`. Add tiny
   typed helpers only where several call sites need them (`chatHost`, `imageHost`, `config.int("context")`).
   `AgentRecord` drops `style = "kind"`; `group` is the only sidebar key.
5. Ids are stable: delete `renameToMatchName`; the editor's name field changes `name` only.
6. Fix C: the chat chip asks `catalog.status(host:)` for the agent's chat host (`running` = green); Library
   "used by" uses `catalog.hosts(usingModel:)` then agents whose `hosts` values contain those ids; Room's
   mixed-host hint compares host ids and says "hosts".
7. `Seeds` (mabel, pip, builder) and harness `templates` write only file-shape keys. Builder's persona: drafts use
   exactly the `agents.read` shape; drop the warning about loose fields.
8. Editor (`AgentEditorView`): binds `hosts` pickers and `SchemaForm` over `config`; no `Kind` picker (group
   field instead).

Tests: `agents.read(x)` validates clean and, with a new id, creates an identical agent; update by patch keeps
unknown keys; `null` removes a config key; rename keeps the id and the chat folder; the chip and "used by" logic
(unit-test the helpers). Update `SPEC.md` §2.5 and the verb table (`agents.update` arguments). UI tests updated
for the editor.

Gate: suite green; `grep -rn "agentJSON\|AgentOverrides\|\.contextTokens\|renameToMatchName" Packages App CLI`
is empty.

### Step 10: one place for sampling (D)

1. `ResolvedSettings` and `turn.start.settings`: remove `top_p`, `temperature`, `repetition_penalty`,
   `max_reply_tokens`. Keep the loop knobs listed above.
2. `parlor_harness.py`: `t.opt` is for loop knobs only. `chat.py` and `character.py` send a sampling key only
   when `t.config` has it. Bump `HarnessPython.marker` to `parlor-harnesses-v3` so seeded copies are rewritten.
3. Gateway `chat`: precedence request > `host.defaults` > machine `settings.llm` (`topP`, `repetitionPenalty`,
   `maxReplyTokens`) > engine default; only `engine.params` are passed on; report the rest in `parlor.ignored`
   (as now).
4. Delete the sampling half of `AgentOverrides` if anything survived step 9.

Tests (fake OpenAI host that records request bodies, reuse `DropInTests.fakeServer`):
- agent silent, host `defaults.top_p = 0.5` → backend receives 0.5;
- agent `config.top_p = 0.9` → 0.9;
- both silent → machine `llm.topP`;
- an unhonoured key appears in `parlor.ignored`.

Gate: suite green, plus a real check: set Stheno's host `defaults.top_p` and `max_tokens` via `hosts.update`, chat
through the `chat` harness, and confirm with `hosts.stats`/the fake recorder that the values arrived.

### Step 11: the Adapter protocol, one routing path (E, G)

1. New `Runplane/Adapter.swift` with the protocol above and `AdapterRegistry` (adapter id → instance).
2. Four adapters, each in its own file under `Runplane/Adapters/`:
   - `MLXAdapter` wraps `MLXRuntime` (route = `host.spec.model`).
   - `OpenAIAdapter` (http-openai): local hosts get a port and launch from the engine manifest (the logic now in
     `derivedProviders`); remote hosts use `url`, `keyRef`, `models`. It owns the HTTP client now in
     `OpenAIProvider` (keep `OpenAIWire`). Remote `host/model` picks the model; a bare remote host id uses
     `models.first`.
   - `CommandImageAdapter` (sd-cli, from `SDCppImage`).
   - `MCPImageAdapter` (Z-Image worker, from `ImageProvider.runWorker`).
3. Gateway: `resolveChat` returns `(HostInfo, EngineManifest, model: String?)` from the catalog only; it calls
   `registry[engine.adapter].chat(...)`. Delete `ProviderRouter`, aliases, `queueKey(route)`,
   `runtimeKnowsModel`, `derivedProviders`, `ProviderConfig`, and the repo / `mlx/` fallbacks in
   `Catalog.resolve` (agents name host ids; a repo is not a model reference). Queues are keyed by host id.
4. `engine.runtime` / `deps.llm` in-process callers (rooms' title or summary helpers, `tryTurn`, CLI) go through
   the gateway's in-process entry point (`gateway.chat(request, host:, caller:)`) so there is still one path.
5. `hosts.start/stop/test/bench` act on `"host:<id>"` through the adapter/supervisor, with no engine-id switches.
   `hosts.stop` on an MLX host unloads that model only.
6. Rename what remains of `Adapters.swift` into the adapter files; delete the file.

Tests: existing Gateway/Provider/GGUF/SDCpp/DropIn tests pass after moving to adapters; new: a scoped token and
`/v1/models` still work; a bare repo name is a 404 `model_not_found`; `hosts.stop` on one MLX host leaves another
loaded (scripted runtime with two residents); a drop-in `http-openai` engine needs no Swift.

Gate: suite green; `grep -rn "ProviderRouter\|ProviderConfig\|setAliases\|\"provider:\"" Packages App CLI Tests`
is empty; `parlor-cli bench` shows the cache pattern; `hosts.bench` on Stheno, the SD host and Z-Image succeeds.

### Step 12: the Supervisor owns lifecycles (F, N)

**As built (closed).** `Supervisor` is the one place a process is spawned (`Supervisor.spawn`/`run`) and the one place the
dashboard and `services.*` verbs read (`register(source:)`, `snapshot()`, `control`). Each owner describes its own rows,
so the Supervisor never switches on an id prefix. HTTP servers (host processes) use `ManagedService` (health check,
restart policy, idle timer). Stdio-pipe children (tool servers, the image worker, harness processes) are spawned through
the Supervisor too but are not `ManagedService`: they have no health URL and are driven by their pipe. The memory budget
stays in `Gateway.admit`. The list below is the original intent.

1. Implement `ensure/release/control/footprint` and the restart policy in `Supervisor` (an actor). `ServiceSpec`
   gains `env`, `cwd`, `startupSeconds`, and `Ready.http(URL)` polling.
2. Owners describe, the Supervisor runs:
   - `OpenAIAdapter.service(for:)` → `host:<id>` (`onDemand(idleMinutes: engine.lifecycle.idleMinutes)`).
   - `MCPImageAdapter` → `host:<id>` for the worker.
   - MCP servers (`MCP.swift`) → `mcp:<name>`; delete its own backoff and idle code.
   - Modules → `module:<name>`.
   - Harness processes → `harness:<agent>` (`always` while a turn runs, restart policy on crash, failure shown in
     System).
3. Memory budget moves from `Gateway.admit` into `supervisor.admit(id, cost:)`, using `footprint()` plus
   `MLXAdapter.residentBytes`. MLX hosts are admitted too (evicting idle MLX residents or idle services, least
   recently used first).
4. `Launch`: `program(binary, args)` executes the binary directly (no shell) when the engine names
   `requires.binary` or a manifest-relative command; `shell` stays only for user-written command lines (inline
   harness commands, `mcp.json` commands, module commands).
5. System dashboard and `services.*` verbs are a projection of `supervisor.snapshot()` only; delete the per-source
   `*Entries()` builders in `EngineServices.swift` except for in-process rows (MLX residents).

Tests: one crash-restart test per service kind (fake process that exits): restarts with backoff, then `failed`;
on-demand starts on first use and stops when idle; harness crash mid-turn ends the turn with an error and the next
turn starts a fresh process; budget evicts the least recently used idle service; `SupervisorTests`
`testOnlySupervisorCreatesProcesses` still holds.

Gate: suite green; `grep -rn "backoff" Packages/ParlorCore/Sources | grep -v Supervisor.swift` is empty; the real
host benches from step 11 still pass.

### Step 13: tools are a noun (S)

1. `tools/<id>/tool.json` is the one way to add a tool server: `{id, title, transport: "stdio"|"http",
   command, args, env, url, lifecycle, idleMinutes, tier, enabled}`. A module is a tool whose program lives in
   its own folder (`command` relative to the folder); no separate module manifest.
2. `ToolSpec: Noun` with a schema; `tools.list/read/validate/create/update/delete/duplicate/enable`, in the noun
   shape. Tools run code, so `create`/`update` of `command`/`args`/`env` always ask the user (as engines and
   harnesses do); this replaces the module approval flow and its `moduleApprovals` setting.
3. The Supervisor runs each tool as `tool:<id>` (step 12's lifecycle). `ToolHub` reads tool specs only.
4. Delete `mcp.json` handling, `ModuleManifest`, module approvals, `tools.mount/unmount`, `modules.list`. Update
   `modules/CLAUDE.md` into `tools/CLAUDE.md`.

Tests: a tool folder dropped in appears and its tools list after approval; changing its `command` asks again; a
crashing tool restarts with backoff then shows failed; an agent's `tools` pattern still gates calls.

Gate: suite green; `grep -rn "mcp.json\|ModuleManifest\|moduleApprovals" Packages App CLI` is empty except the
folder guide's one line saying it is gone.

### Step 14: images through the gateway, settings in the host (H, I)

1. `image.generate` is a thin client: it calls the gateway's in-process image entry point with the agent as caller,
   so scope, problem check, budget and the per-host queue apply. Image requests are queued per host like chat.
2. SD settings move into the `stable-diffusion.cpp` engine's `settings` schema (`prediction`, `sampler`,
   `scheduler`, `steps`, `fast_steps`, `cfg`, `clip_skip`, `portrait`/`square`/`landscape`, `prompt_prefix`,
   `negative`, `vae`, `extra_args`). V-prediction detection becomes the auto host's default (`prediction: v` when
   the file name says so). Delete `SDCppImage.Config` file IO and every mention of `parlor-image.json`.
3. Tuning shows these through the existing `SchemaForm` with no new UI code.

Tests: the tool path is refused for a scoped caller without the host; an SD host's `settings.steps` reaches the
sd-cli command line (`SDCppTests`); no file named `parlor-image.json` is read or written.

Gate: suite green; real: `hosts.update` the SD host's `steps`, `hosts.bench` it, and see the timing change.

### Step 15: the UI goes through verbs, and every noun has a Library tab (J, R, T)

1. Every mutating UI action calls `engine.commands.call(verb, args, caller: .ui)` through one helper (the
   `runVerb` helper in `LibraryView` generalised into `App/Verbs.swift`), and shows verb errors inline.
2. Engine mutators that duplicate a verb become `internal` (verbs use them; the App target cannot). Read-only
   state (`agents`, `catalog`, `services`, `room(for:)`, `send`) stays public.
3. Library becomes one generic `NounLibraryView(kind:)` for every noun in the registry: Models (kept as is, it
   manages downloads), Hosts, Engines, **Harnesses**, **Tools**, and Agents. Each tab lists `scan()` entries with
   status and issues, a detail pane (read-only summary from the schema: for a harness its needs, config schema,
   actions, templates and which agents use it), a JSON editor with Validate and Save (`update` patch), and
   Duplicate, Delete, Reveal in Finder, and noun-specific buttons declared in the registry (`hosts`: Test, Bench,
   Tuning; `harnesses`: Try, which runs `agents.try` with a minimal draft on the default chat host). Delete the
   hand-written `HostsTab` and `EnginesTab`.
4. Add `CommandTests.testTheAppOnlyMutatesThroughVerbs`: scans `App/*.swift` for calls to a list of engine
   mutators and fails on any hit (same style as `testNobodyElseCallsInstalledDirectory`).

Gate: unit and UI suites green; the test above passes.

### Step 16: hygiene (L, M, O)

1. Python harness files move to `Packages/ParlorCore/Sources/ParlorCore/Resources/harnesses/*.py` (SwiftPM
   resources), copied into the home by the existing marker logic. Add a test that runs `python3 -m py_compile` on
   each and the seeded `chat` harness end to end (already exists; keep it).
2. Delete the Swift `Harness` protocol and `TurnContext.llm`; `HarnessHost` is the concrete type. Rename it
   `HarnessHost` (the plan's name).
3. Concurrency: convert `HostQueues` users, `HarnessHost` process state and the adapter caches to actors or
   `Mutex`; zero lock-in-async warnings in a ParlorCore build; no new `@unchecked Sendable`. Staying in Swift 5 mode
   is fine; record the remaining `@unchecked` count in the commit message.
4. Update `SPEC.md` (§1 shape: Adapter, Supervisor ownership; §2.2 SD settings; §2.5 agent; §3 verbs; §4 settings
   in `turn.start`), `README.md`, the folder guide, and mark `PLAN-V3.1.md` done at the top.
5. Walk "Done means" below and report each line.

Gate (final): unit + UI suites green; `parlor-cli bench` cache pattern; real benches on Qwen3 (MLX), Stheno
(llama.cpp), the SD checkpoint and Z-Image; Builder on Qwen3-4B creates an agent through validate → try → create
(record honestly if the model still fails; it is not a gate failure if the verbs behave); `scripts/package.sh`;
install to `/Applications` leaving one Parlor.

## Done means (acceptance for "only UI feedback left")

Walk this list at the end of step 16 on the installed app. Use a temp `PARLOR_HOME` for anything that writes;
only Zubin resets his real `~/parlor`, never the agent. Each line is either already covered by a test or checked by hand and
reported.

1. Fresh `~/parlor`: the app starts, seeds Mabel, Pip and Builder, `chat` and `character` harnesses, built-in
   engines, and shows every installed model with an auto host. No errors in System.
2. Every noun (engine, host, harness, agent, tool) has a Library tab: list, status, issues, detail, JSON edit with
   validate, duplicate, delete. Seeded ones say "built in, duplicate to change".
3. A coding agent drops a harness, engine, host, agent or tool folder in `~/parlor`: it appears within a second; a
   broken one shows its issues; fixing the file clears them. No restart.
4. Chat works on an MLX host, a llama.cpp host and a remote OpenAI host; images on Z-Image and an SD checkpoint;
   Stop, compaction, rooms and the harness buttons (Gen image, Selfie, Compact now) work.
5. Tuning changes a host's settings and defaults; the next chat and `hosts.bench` reflect them.
6. Kill any child process (a host server, a harness, a tool): it restarts, or shows failed with its log; the app
   never hangs.
7. Memory: loading a second large host evicts the idle one instead of swapping.
8. An outside client with a scoped token can list and use only its hosts at `/v1` and call verbs at `/mcp`.
9. Builder (or Claude Code with a configure token) makes an agent with validate → try → create; the result reads
   back in exactly the shape it was written.
10. `SPEC.md` describes the app as it is; every verb in it exists at its tier; there is no code path for a removed
    feature.

## Expected size

Net deletion. Roughly: −500 lines (`ProviderRouter`, aliases, three lifecycle copies, `agentJSON`/`fields`/
`apply`, V2 shims), +350 lines (protocol, four thin adapters, Supervisor lifecycle), and the Python moving out of
Swift.

## Not in V3.1

New engines or harnesses, a Swift 6 language-mode switch, remote sync, and any UI redesign beyond what the steps
need.
