# roundRobin

Multi-agent ensemble coding system. Spawn N Claude agents in isolated workspaces, coordinate via a shared chat channel, watch them work in real time.

## How it works

- **Control plane** runs locally always — CRUD for tasks and souls, launches Docker containers per run
- Each run spins up a fresh Docker container with its own hub, DB, and agents
- Agents get isolated copies of the task data directory inside the container
- Agents communicate via a push-based MCP chat channel (Claude's `--dangerously-load-development-channels`)
- Watch each run at its own port — no stale history, fully ephemeral

## Running

### 1. Build the image (once, or after changes)

```bash
docker build -t roundrobin .
```

### 2. Start the control plane

```bash
ANTHROPIC_API_KEY=sk-... bun src/control/server.ts
```

Open `http://localhost:8080`.

### 3. Use the UI

- **Tasks tab** — create and edit tasks (task.md + optional data/)
- **Souls tab** — create and edit agent personalities
- **Runs tab** — select a task + souls, hit Launch — get a link to the live run

Each run opens at its own port (starting at 9100). The run UI has Chat and Logs tabs.

## UI

### Control plane (`localhost:8080`)
- **Tasks** — CRUD for task bundles
- **Souls** — CRUD for agent soul files
- **Runs** — launch runs, view active containers, kill runs

### Run UI (per-container port)
- **Chat** — real-time agent conversation feed, human message input
- **Logs** — live tmux pane stream per agent (Split view) or interleaved JSONL

## Task structure

```
tasks/
  <task-name>/
    task.md       # problem description
    data/         # files cloned into each agent workspace (optional)
```

## Soul structure

```
souls/
  <name>.md       # agent personality + instructions
```

Frontmatter `name:` field sets the display name. Falls back to filename.

## Project structure

```
src/
  control/
    server.ts     # local control plane (CRUD + docker lifecycle)
    ui/
      index.html  # control plane SPA
  hub/
    server.ts     # run hub (runs inside Docker per run)
    ui/
      index.html  # run UI (chat + logs)
  channel/
    server.ts     # MCP server (one per agent), bridges hub WS ↔ Claude channel
souls/            # agent soul files
tasks/            # task bundles
prompts/
  default.md      # shared system prompt injected into every agent's CLAUDE.md
```
