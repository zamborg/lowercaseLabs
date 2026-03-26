# roundRobinMapReduce

Minimal MapReduce runner for agent wrappers.

## Contract

- A config file declares three mapper agents and one reducer.
- Each agent is a shell script with the same interface: it receives `task.md` as its first argument.
- Mappers run in parallel in isolated workspaces.
- The reducer runs after all mappers finish in the run root, where it can inspect all mapper workspaces.
- The reducer still runs if some mappers fail, as long as at least one mapper completed successfully.
- A small web page shows live tmux panes and run status.

## Quick start

Build and launch the sample run:

```bash
node src/runTask.js configs/example.json
```

That command will:

1. Build the Docker image.
2. Create a fresh run directory under `runs/`.
3. Start one container for the run.
4. Print the local viewer URL.

For a stronger end-to-end test with real code generation and reduction, run:

```bash
node src/runTask.js configs/tri-models-slugify.json
```

## Config shape

```json
{
  "imageTag": "ruffle-mapreduce",
  "port": 9310,
  "keepAliveSecondsAfterExit": 120,
  "sourceDir": "../tasks/sample-source",
  "task": "../tasks/example.md",
  "mappers": [
    { "name": "alpha", "command": "../agents/examples/mapper-echo.sh" },
    { "name": "beta", "command": "../agents/examples/mapper-echo.sh" },
    { "name": "gamma", "command": "../agents/examples/mapper-echo.sh" }
  ],
  "reducer": { "name": "reducer", "command": "../agents/examples/reducer-echo.sh" }
}
```

## Task shape

The task markdown must contain these headings:

```md
## Map Task
...

## Reduce Task
...
```

## Output conventions

- Each mapper should write its local artifacts in its own workspace.
- Each mapper should also write `RESULT.md` in its workspace.
- The reducer should write `RESULT.md` in the run root.

## Architecture

The system has two layers:

- `src/runTask.js`: host-side launcher
- `src/runtime.js`: in-container run orchestrator

The host launcher:

1. Reads a config file.
2. Reads a task file and splits it into `Map Task` and `Reduce Task`.
3. Builds a `runspec.json` for the run.
4. Creates a fresh host-side run directory under `runs/<run-id>/`.
5. Starts one Docker container for the run.
6. Mounts the project into the container read-only at `/project`.
7. Mounts the run directory into the container read-write at `/run`.

The in-container runtime:

1. Creates one workspace per mapper at `/run/workspaces/mapper-*`.
2. Copies `sourceDir` into each mapper workspace.
3. Writes a mapper-specific `task.md` into each mapper workspace.
4. Launches each mapper in its own tmux session.
5. Waits for all mappers to finish.
6. If at least one mapper succeeded, writes a reducer `task.md` at `/run/task.md`.
7. Launches the reducer in `/run`, where it can inspect all mapper workspaces.
8. Writes final outputs directly into `/run`, which persists back to the host.
9. If all mappers fail, the run ends as `failed` and the reducer is not started.

After the run reaches a terminal state (`complete` or `failed`), the container stays alive for a short grace period and then exits automatically. Because Docker runs with `--rm`, the container is removed after exit.

## Agent Interface

Every agent wrapper uses the same contract:

```bash
your-wrapper.sh task.md
```

When a wrapper starts:

- the current working directory is already set to its assigned workspace
- `task.md` is present in that directory
- the wrapper should do all work relative to the current directory

### Mapper Interface

Each mapper gets:

- an isolated workspace
- a copy of the configured `sourceDir`
- a generated `task.md` containing:
  - `# Map Task`
  - identity line
  - the original map instructions
  - a short output contract

Each mapper is expected to:

- work only inside its own workspace
- create whatever candidate artifacts it needs
- write `RESULT.md` in the workspace root
- exit nonzero on failure

### Reducer Interface

The reducer gets:

- working directory `/run`
- generated `/run/task.md`
- visibility into `/run/workspaces/mapper-*`
- access to all mapper outputs, including their `RESULT.md`

The reducer is expected to:

- inspect mapper workspaces
- ignore failed mapper runs when they do not produce usable outputs
- decide on a final output
- write final artifacts into `/run`
- write `/run/RESULT.md`

## Data Flow

Input flow:

- config -> `src/runTask.js`
- task markdown -> split into map/reduce sections
- source directory -> copied into each mapper workspace

Output flow:

- mapper outputs stay in `/run/workspaces/mapper-*`
- reducer outputs go into `/run`
- `/run` is a bind mount to `runs/<run-id>/` on the host

That means persistence does not depend on the container staying alive. The container is only needed for the live viewer and tmux inspection.

## Shutdown Policy

- Run outputs are durable as soon as they are written into `runs/<run-id>/` on the host.
- After a run finishes, the container stays up for `keepAliveSecondsAfterExit`.
- Default: `120` seconds.
- Set `keepAliveSecondsAfterExit` to `0` for immediate shutdown.
- During the grace period, `/api/status` includes `shutdownAt`.

## Built-in Provider Wrappers

The image now installs three CLIs globally:

- `claude` via `@anthropic-ai/claude-code`
- `codex` via `@openai/codex`
- `gemini` via `@google/gemini-cli`

Provider wrappers are available at:

- [agents/claude.sh](/Users/zubinaysola/Documents/personal/lowercaseLabs/roundRobinMapReduce/agents/claude.sh)
- [agents/codex.sh](/Users/zubinaysola/Documents/personal/lowercaseLabs/roundRobinMapReduce/agents/codex.sh)
- [agents/gemini.sh](/Users/zubinaysola/Documents/personal/lowercaseLabs/roundRobinMapReduce/agents/gemini.sh)

They all use the same contract: `script task.md`

Model-pinned wrappers are also available:

- [agents/claude-sonnet.sh](/Users/zubinaysola/Documents/personal/lowercaseLabs/roundRobinMapReduce/agents/claude-sonnet.sh)
- [agents/codex-gpt-5-codex.sh](/Users/zubinaysola/Documents/personal/lowercaseLabs/roundRobinMapReduce/agents/codex-gpt-5-codex.sh)
- [agents/codex-gpt-5.2.sh](/Users/zubinaysola/Documents/personal/lowercaseLabs/roundRobinMapReduce/agents/codex-gpt-5.2.sh)
- [agents/gemini-2.5-flash-lite.sh](/Users/zubinaysola/Documents/personal/lowercaseLabs/roundRobinMapReduce/agents/gemini-2.5-flash-lite.sh)

These set:

- Claude: `sonnet`
- Codex: `gpt-5-codex`
- Codex: `gpt-5.2-codex`
- Gemini: `gemini-2.5-flash-lite`

## Credentials

`src/runTask.js` forwards these host environment variables into the container when present:

- `ANTHROPIC_API_KEY`
- `CLAUDE_CODE_OAUTH_TOKEN`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_GENAI_USE_VERTEXAI`
- `RUFFLE_CLAUDE_MODEL`
- `RUFFLE_CODEX_MODEL`
- `RUFFLE_GEMINI_MODEL`

For a simple API-key setup, export:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

Then run a config that points at the provider wrappers.

## Seams To Tighten

Current weak seams in the design:

1. Mapper success is based on exit code only. The runtime does not yet verify required outputs like `RESULT.md`.
2. Reducer input is implicit. It discovers `workspaces/mapper-*` by convention instead of receiving an explicit manifest.
3. The map/reduce interface is encoded in runtime prose rather than a versioned schema.
4. Some provider runtime behavior is still selected by wrapper naming conventions.
5. Auth/config mounting is powerful but currently under-documented as an operational dependency.

Recommended next contract:

- mapper emits both `RESULT.md` and `result.json`
- runtime generates `/run/reduce-input.json`
- reducer consumes `reduce-input.json` plus mapper workspaces
- execution semantics like provider, user, and auth mode move into explicit config rather than filename heuristics

Feedback / next architecture changes can go here.
