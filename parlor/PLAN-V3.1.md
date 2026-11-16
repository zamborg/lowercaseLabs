# parlor V3.1: finish the three patterns

V3 (steps 1–7, `PLAN-V3.md`) shipped the right shape: engines → hosts → harnesses → agents, one gateway, one
catalog, every agent a harness process. Two principal reviews (2026-11-16) found that each of the three patterns
(nouns, adapters, services) is only partly done, and the gaps are where the bugs are. V3.1 finishes them. No new
features.

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
| D | Sampling precedence is decided three times. `ResolvedSettings` folds machine `topP`/`maxReplyTokens` into `turn.start.settings`; `chat.py` `t.opt` always sends them; the gateway applies host `defaults` only for absent keys | `Resolved.swift`, `ExternalHarness.swift` (`settings`), `HarnessPython.swift` (`opt`, request kwargs), `Gateway.swift` `chat` | A host's `top_p`/`max_tokens` defaults from Tuning never reach the `chat` harness |
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
| O | Concurrency: Swift 5 mode, 22 `@unchecked Sendable`, 11 `NSLock`s, lock-in-async warnings in `ExternalHarness` | ParlorCore | Debt for Swift 6 |

## Target shapes

```swift
// Nouns: agents join the others. One shape everywhere.
// agents.list   -> [AgentRecord JSON]              (same keys as agent.json; no persona)
// agents.read   -> AgentRecord JSON + "persona" + "memory"   (exactly what create accepts, plus memory)
// agents.create {draft} | {template, name}         (template expands server-side into a draft, same path)
// agents.update {id, patch}                        (merge patch over agent.json; "persona" key writes persona.md)
// Ids never change after create. Renaming changes `name` only.

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

### Step 8: agents are a true noun (A, B, C, K)

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

### Step 9: one place for sampling (D)

1. `ResolvedSettings` and `turn.start.settings`: remove `top_p`, `temperature`, `repetition_penalty`,
   `max_reply_tokens`. Keep the loop knobs listed above.
2. `parlor_harness.py`: `t.opt` is for loop knobs only. `chat.py` and `character.py` send a sampling key only
   when `t.config` has it. Bump `HarnessPython.marker` to `parlor-harnesses-v3` so seeded copies are rewritten.
3. Gateway `chat`: precedence request > `host.defaults` > machine `settings.llm` (`topP`, `repetitionPenalty`,
   `maxReplyTokens`) > engine default; only `engine.params` are passed on; report the rest in `parlor.ignored`
   (as now).
4. Delete the sampling half of `AgentOverrides` if anything survived step 8.

Tests (fake OpenAI host that records request bodies, reuse `DropInTests.fakeServer`):
- agent silent, host `defaults.top_p = 0.5` → backend receives 0.5;
- agent `config.top_p = 0.9` → 0.9;
- both silent → machine `llm.topP`;
- an unhonoured key appears in `parlor.ignored`.

Gate: suite green, plus a real check: set Stheno's host `defaults.top_p` and `max_tokens` via `hosts.update`, chat
through the `chat` harness, and confirm with `hosts.stats`/the fake recorder that the values arrived.

### Step 10: the Adapter protocol, one routing path (E, G)

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

### Step 11: the Supervisor owns lifecycles (F, N)

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
host benches from step 10 still pass.

### Step 12: images through the gateway, settings in the host (H, I)

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

### Step 13: the UI goes through verbs (J)

1. Every mutating UI action calls `engine.commands.call(verb, args, caller: .ui)` through one helper (the
   `runVerb` helper in `LibraryView` generalised into `App/Verbs.swift`), and shows verb errors inline.
2. Engine mutators that duplicate a verb become `internal` (verbs use them; the App target cannot). Read-only
   state (`agents`, `catalog`, `services`, `room(for:)`, `send`) stays public.
3. Add `CommandTests.testTheAppOnlyMutatesThroughVerbs`: scans `App/*.swift` for calls to a list of engine
   mutators and fails on any hit (same style as `testNobodyElseCallsInstalledDirectory`).

Gate: unit and UI suites green; the test above passes.

### Step 14: hygiene (L, M, O)

1. Python harness files move to `Packages/ParlorCore/Sources/ParlorCore/Resources/harnesses/*.py` (SwiftPM
   resources), copied into the home by the existing marker logic. Add a test that runs `python3 -m py_compile` on
   each and the seeded `chat` harness end to end (already exists; keep it).
2. Delete the Swift `Harness` protocol and `TurnContext.llm`; `ExternalHarness` is the concrete type. Rename it
   `HarnessHost` (the plan's name).
3. Concurrency: convert `HostQueues` users, `ExternalHarness` process state and the adapter caches to actors or
   `Mutex`; zero lock-in-async warnings in a ParlorCore build; no new `@unchecked Sendable`. Staying in Swift 5 mode
   is fine; record the remaining `@unchecked` count in the commit message.
4. Update `SPEC.md` (§1 shape: Adapter, Supervisor ownership; §2.2 SD settings; §2.5 agent; §3 verbs; §4 settings
   in `turn.start`), `README.md`, the folder guide, and mark `PLAN-V3.1.md` done at the top.

Gate (final): unit + UI suites green; `parlor-cli bench` cache pattern; real benches on Qwen3 (MLX), Stheno
(llama.cpp), the SD checkpoint and Z-Image; Builder on Qwen3-4B creates an agent through validate → try → create
(record honestly if the model still fails; it is not a gate failure if the verbs behave); `scripts/package.sh`;
install to `/Applications` leaving one Parlor.

## Expected size

Net deletion. Roughly: −500 lines (`ProviderRouter`, aliases, three lifecycle copies, `agentJSON`/`fields`/
`apply`, V2 shims), +350 lines (protocol, four thin adapters, Supervisor lifecycle), and the Python moving out of
Swift.

## Not in V3.1

New engines or harnesses, a Swift 6 language-mode switch, remote sync, and any UI redesign beyond what the steps
need.
