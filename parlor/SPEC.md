# parlor specs (v3)

The contracts behind `PLAN-V3.md`. Code implements these; when the code and this file disagree, one of them is a bug.
`PLAN-V2.md` is superseded.

- **Words:** MUST and MUST NOT are requirements. Anything else is guidance.
- **Versioning:** every JSON file carries `"version": 1`; a missing version means 1. A reader ignores unknown keys and a
  writer keeps the keys it does not understand.

## 1. The shape

Three layers, each made of nouns (a JSON file) and services (a process).

```
Runplane   engines -> hosts           what can serve a model, and how it is run
Agents     harnesses -> agents        how an agent thinks, and who it is
Control    verbs, gateway, MCP        how the UI, other agents and outside clients ask for things
```

- **Engine**: a backend. Adapter is `inprocess-mlx`, `http-openai`, `command` or `mcp-stdio`.
- **Host**: an engine plus a model, how it runs (`settings`) and how it is asked (`defaults`).
- **Harness**: a program that runs an agent's turns over the harness protocol (section 4).
- **Agent**: a folder: its harness, the hosts it uses for each need, tools, tier and config.
- **Catalog**: the one authority for what is available (`resolve`, `status(host:)`, `hosts(capability:)`).
- **Gateway**: one loopback server for `/v1/*` and `/mcp`. Every ask of a host goes through it, so the app's own chats,
  harnesses and outside clients share the same per-host queues, priorities, memory budget and token scopes.

Source layout: `Packages/ParlorCore/Sources/ParlorCore/{Runplane,Agents,Control}` for the engine's parts, with
`Engine.swift` holding state and lifecycle.

## 2. Home: file formats

Root `~/parlor`, or `$PARLOR_HOME`.

```
settings.json               machine settings, tokens (hashes), module approvals, gateway
mcp.json                    mounted MCP servers (Claude Desktop format)
engines/<id>/engine.json    an engine
hosts/<id>.json             a host; hosts/<id>.bench.json holds its last 10 bench runs
harnesses/<id>/harness.json a harness, beside its program
agents/<id>/agent.json      an agent; persona.md is its system prompt; memory.md its memory
chats/<id>/log.jsonl        a conversation; summary.md, conversation.json (room), tools.json
modules/<name>/module.json  a small tool server you or a model wrote
media/YYYY-MM/  models/<org>/<name>/  workspaces/<agent>/  logs/
```

There is no migration: a home made by an earlier version is not read. Start with an empty folder.

### 2.1 Nouns

Engines, hosts, harnesses and agents are nouns. One Codable struct is the file, the verb input and output, and the wire JSON.
`NounStore` is the only reader and writer. Every noun has `list`, `read`, `validate`, `create`, `update` and `delete` verbs.
`update` is a JSON merge patch (RFC 7386: objects merge, `null` removes a key). `validate` writes nothing and returns every
problem as `{code, path, message, hint, severity}`, so a model can fix a draft in one pass. `schema.get {noun}` returns
the JSON Schema.

The schema subset used for harness `config`, engine `settings` and action `args`: `type` (string, number, integer, boolean,
array, object), `enum`, `default`, `minimum`, `maximum`, `format: multiline`, `description`.

### 2.2 Engine: `engines/<id>/engine.json`

`id`, `title`, `capabilities` (`chat`, `image`), `formats` (model formats it serves), `adapter`, `launch {command, args}`
(placeholders from the host's `settings`), `health`, `tool` (for `mcp-stdio`), `params` (sampling parameters honoured; the
rest are dropped and reported), `requires {binary, hint}`, `settings` (schema), `lifecycle {idleMinutes, startupSeconds}`,
`remote` (true when the host names a URL). Creating or changing an engine always asks the user.

### 2.3 Host: `hosts/<id>.json`

`id`, `engine`, `model`, `settings`, `defaults` (used when an agent's config is silent), `enabled`; for remote engines
`url`, `keyRef` (Keychain item) and `models`. An installed model whose format has exactly one engine gets an **auto host**
(`slug(shortName)`, never written); a file with the same id wins.

### 2.4 Harness: `harnesses/<id>/harness.json`

`id`, `title`, `description`, `command`, `args`, `needs` (`{"chat": "required", "image": "optional"}`), `config` (schema, shown
as a form in the agent editor), `templates` (starting agents), `actions` (buttons: `id`, `title`, `icon`, `placement`
`composer|message|settings`, `needs`, `hint`, `args`). Creating or changing a harness always asks the user.

Seeded: `chat` (the default) and `character`, with `parlor_harness.py` (the helper) beside them, managed by the marker
`parlor-harnesses-v2`; copy one to make your own.

### 2.5 Agent: `agents/<id>/agent.json`

`id` (the folder), `name`, `description`, `harness` (default `chat`), `hosts` (need to host or `host/model`), `tools`
(patterns `*`, `server.*`, `server.tool`), `tier` (`none|read|act|configure`), `group` (the sidebar section; default `Characters` for the `character` harness, else `Assistants`), `color`, and
`config` (the harness's own settings, validated against its schema; a missing key falls back to the host's `defaults`,
then to machine `settings`). Unknown keys survive an edit. The id never changes: renaming changes `name` only.
`agents.list` returns this shape; `agents.read` returns it plus `persona` and `memory`, which is exactly what
`agents.create` accepts (memory is ignored). `agents.update {id, patch}` is a JSON merge patch over it; a `persona` key
rewrites `persona.md`.

### 2.6 Log line, summary, room

`chats/<id>/log.jsonl` is one JSON object per line (ISO-8601 dates): `id`, `role`, `text`, `at`, and optionally `speaker`,
`tool`, `images`. `summary.md` is the harness's rolling summary. `conversation.json` makes a chat a room: `participants` and a
`scheduler` (`userDriven`, `roundRobin` or `mention`). `chats/<id>/tools.json` is the last tool set seen, for the "Tools updated" marker.

### 2.7 Settings, mounts, modules

`settings.json` is tolerant: unknown keys survive, bad values fall back. Hosts are never listed there; they are files. `mcp.json` is the Claude Desktop format with
optional `lifecycle` and `idleMinutes`. A module is a folder with `module.json` and a program; it runs only after the user
approves it and re-asks when the manifest changes. See `modules/CLAUDE.md`.

## 3. Control

### 3.1 Verbs

Every control the app has is a verb behind one gate: tier check, destructive confirmation, argument validation (unknown keys
are refused). The UI, the in-process `parlor` provider, the HTTP server and the CLI all call `call`. Verb names are
`parlor.<group>.<verb>`.

| Verb | Tier | UI action |
|------|------|-----------|
| `agents.list`, `agents.read`, `agents.templates`, `agents.validate` | read | sidebar, editor, New Agent menu; validate reports every problem in a draft |
| `agents.try` | act | one throwaway turn of a draft, nothing saved |
| `agents.create`, `agents.update`, `agents.duplicate` | configure | New Agent, editor save, Duplicate; edits to an agent another caller made show a diff and ask first |
| `agents.delete` | destructive | Delete agent |
| `conversations.list`, `conversations.read` | read | sidebar, chat view |
| `conversations.create`, `conversations.post`, `conversations.run`, `conversations.stop`, `conversations.compact` | act | rooms, send, Stop, Compact now |
| `conversations.prune`, `conversations.reset` | destructive | delete messages, Clear chat |
| `models.list`, `models.search` | read | Library |
| `models.download`, `models.pause`, `models.load`, `models.unload` | act | Download, Pause, Resume, load |
| `models.delete` | destructive | Delete model |
| `hosts.list`, `hosts.read`, `hosts.validate`, `hosts.test`, `hosts.stats` | read | Library, Hosts |
| `hosts.create`, `hosts.update`, `hosts.duplicate`, `hosts.enable` | configure | Hosts editor |
| `hosts.start`, `hosts.stop`, `hosts.bench` | act | Start, Stop, Tuning bench |
| `hosts.delete` | destructive | Delete host |
| `engines.list`, `engines.read`, `engines.validate` | read | Library, Engines |
| `engines.create`, `engines.update`, `engines.duplicate` | configure | always asks the user: an engine runs code. Built-in engines are read-only; duplicate to change |
| `engines.delete` | destructive | Delete engine |
| `harnesses.list`, `harnesses.read`, `harnesses.validate` | read | agent editor |
| `harnesses.create`, `harnesses.update`, `harnesses.duplicate` | configure | always asks the user: a harness runs code. Built-in harnesses are read-only; duplicate to change |
| `harnesses.delete` | destructive | Delete harness |
| `schema.get` | read | the JSON Schema of agent, host, harness or engine |
| `services.list`, `services.logs`, `modules.list` | read | System |
| `services.start`, `services.stop`, `services.restart` | act | Restart server |
| `settings.get` | read | Performance |
| `settings.set`, `settings.reset` | configure | Performance fields, Reset to defaults |
| `tools.list` | read | Tools |
| `tools.mount`, `tools.unmount`, `tools.enable` | configure | add, delete, enable or disable a server |
| `home.paths`, `home.reveal` | read | Your data (reveal does nothing over the wire) |
| `memory.read`, `memory.append`, `memory.replace` | none | an agent's own memory; another agent's needs configure |

Rules:
- Tiers are ordered `none` < `read` < `act` < `configure`. `destructive` is never a tier; it always asks.
- An agent's `tier` gates `parlor.*` tools. Mounted MCP tools are gated by `tools` patterns.
- Destructive verbs called by an agent or token show a confirmation in the app and block until answered. With no one to ask they
  are refused with "needs the user". Tokens never bypass this.
- An agent or token that edits an agent it did not create sees a diff and the user must approve it.
- Writes are validated; unknown keys and out-of-range values are refused with a reason.

### 3.2 Gateway and tokens

`http://127.0.0.1:<port>` serves `GET /v1/models`, `POST /v1/chat/completions` (streaming, tools, usage),
`POST /v1/images/generations` and `/mcp` (streamable HTTP). Errors are OpenAI-shaped. A foreign `Origin` or `Host` is refused.
A bearer token is required. Tokens are made in System, stored in the Keychain (names and hashes in `settings.json`), and have a
name, a tier, an optional expiry and an optional host list; a scoped token is forbidden on other hosts. Processes the app
launches (harnesses, modules, bench) get an ephemeral token at their manifest's or agent's tier.

### 3.3 Services

One `Supervisor` creates every process (engines, MCP servers, modules, harnesses). Each shows in System with state, log and
Start/Stop/Restart. `onDemand` starts on first use and stops when idle; a crash restarts with backoff and then shows failed.
Logs rotate at 1 MB.

## 4. Harness protocol (v2)

JSON lines over the harness's stdin and stdout; stderr goes to the log.

App to harness:
- `turn.start`: `version: 2`, `turn`, `conversation`, `agent`, `hosts` (need to `{model, baseURL}`, the gateway), `config`,
  `participants`, `transcript`, `tools`, `settings`, `memory`, `summary`, `lastPromptTokens`, `action`.
- `tool.result`: `turn`, `callID`, `text`, `isError`.
- `turn.cancel`: `turn`.

Harness to app:
- `text.delta`, `tool.call` (the app runs it, held to the agent's allowlist and tier), `message` (a finished message, such as
  an image), `status`, `stats`, `summary` (the new rolling summary), `error`, and `turn.done`.

Order: `turn.start`, then any number of `text.delta` and `tool.call` (each answered by `tool.result`), then `turn.done`. The
harness asks hosts through the gateway at the URL it was given, so queues and memory budget apply. `parlor_harness.py` hides
all of this behind a small API; see `harnesses/CLAUDE.md`.

## 5. Conformance tests

`Tests/` runs without weights: a scripted model and a fake OpenAI server stand in. They cover nouns and schema checks, the
catalog, the gateway (scopes, errors, disconnects), the harness wire format, rooms, every verb in the table above at its
stated tier (`CommandTests.testEveryVerbInTheSpecExistsAtItsTier` parses this file), drop-in engines and harnesses, and the
memory, services and module lifecycles. `parlor-cli bench` runs the same cache-pattern check through the gateway on real
models.
