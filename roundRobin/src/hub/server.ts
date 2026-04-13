import { Database } from "bun:sqlite";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "fs";
import { join, resolve, dirname } from "path";
import { $ } from "bun";
import { createRunner } from "../runners/index.ts";
import type { RuntimeAgent } from "../runners/index.ts";

interface AgentBundle {
  name: string;
  variant: string;
  model: string;
  soul: string;
  systemPrompt: string;
}

interface RunBundle {
  taskName: string;
  taskContent: string;
  dataFiles: Record<string, string>; // relPath -> base64
  agents: AgentBundle[];
}

// --- Config ---
const args = Bun.argv.slice(2);
function getArg(name: string, fallback: string): string {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : fallback;
}

const PORT = parseInt(getArg("port", "9090"));
const DB_PATH = getArg("db", "./run.db");
const PROJECT_ROOT = resolve(dirname(import.meta.dir), "..");

// --- SQLite ---
const db = new Database(DB_PATH);
db.run("PRAGMA journal_mode = WAL");
db.run(`CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id TEXT NOT NULL,
  content TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'message',
  timestamp TEXT NOT NULL,
  metadata TEXT
)`);

const insertMsg = db.prepare(
  "INSERT INTO messages (agent_id, content, type, timestamp, metadata) VALUES (?, ?, ?, ?, ?)"
);
const getMessages = db.prepare(
  "SELECT * FROM messages ORDER BY id DESC LIMIT ?"
);
const getAllMessages = db.prepare("SELECT * FROM messages ORDER BY id ASC");

// --- State ---
const agentSockets = new Map<string, any>(); // agent_id -> ws
const uiSockets = new Set<any>();
const consensusState = new Map<string, string>(); // agent_id -> summary

function storeMessage(msg: { agent_id: string; content: string; type: string; timestamp: string; metadata?: any }) {
  insertMsg.run(msg.agent_id, msg.content, msg.type, msg.timestamp, JSON.stringify(msg.metadata || null));
}

function broadcast(msg: any, excludeWs?: any) {
  const data = JSON.stringify(msg);
  for (const [, ws] of agentSockets) {
    if (ws !== excludeWs && ws.readyState === 1) ws.send(data);
  }
  for (const ws of uiSockets) {
    if (ws.readyState === 1) ws.send(data);
  }
}

function broadcastAll(msg: any) {
  const data = JSON.stringify(msg);
  for (const [, ws] of agentSockets) {
    if (ws.readyState === 1) ws.send(data);
  }
  for (const ws of uiSockets) {
    if (ws.readyState === 1) ws.send(data);
  }
}

function systemMessage(content: string) {
  const msg = { type: "system", agent_id: "system", content, timestamp: new Date().toISOString() };
  storeMessage(msg);
  broadcastAll(msg);
}

// --- UI file ---
const uiPath = join(import.meta.dir, "ui", "index.html");
let uiHtml: string;
try {
  uiHtml = readFileSync(uiPath, "utf-8");
} catch {
  uiHtml = "<html><body><h1>roundRobin</h1><p>UI file not found</p></body></html>";
}

// --- Run management ---
let currentRun: {
  status: "idle" | "running" | "complete";
  runDir?: string;
  agents: { id: string; name: string; variant: string; session: string }[];
  startedAt?: string;
} = { status: "idle", agents: [] };


async function startRun(bundle: RunBundle) {
  const { taskName, taskContent, dataFiles, agents: agentBundles } = bundle;

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const runDir = join(PROJECT_ROOT, "runs", timestamp);

  // Use pre-compiled binary in Docker (dist/channel), fall back to source for local dev
  const compiledChannel = join(PROJECT_ROOT, "dist/channel");
  const channelPath = existsSync(compiledChannel)
    ? compiledChannel
    : resolve(PROJECT_ROOT, "src/channel/server.ts");
  const isCompiled = existsSync(compiledChannel);

  // Build roster
  const agents: (RuntimeAgent & { soul: string; systemPrompt: string; variant: string })[] = agentBundles.map((b, i) => ({
    id: `agent-${i + 1}`,
    name: b.name,
    variant: b.variant,
    model: b.model,
    soul: b.soul,
    systemPrompt: b.systemPrompt,
    session: `roundrobin-agent-${i + 1}`,
  }));
  const roster = agents.map(a => `- ${a.id}: ${a.name}`).join("\n");

  // Create workspaces
  for (const agent of agents) {
    const workspace = join(runDir, "workspaces", agent.id);
    mkdirSync(workspace, { recursive: true });

    // Write data files from bundle (base64-decoded)
    for (const [relPath, b64] of Object.entries(dataFiles)) {
      const dest = join(workspace, relPath);
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, Buffer.from(b64, "base64"));
    }

    // .mcp.json
    const mcpConfig = {
      mcpServers: {
        "roundrobin-chat": {
          command: isCompiled ? channelPath : "bun",
          args: isCompiled ? [] : [channelPath],
          env: {
            ROUNDROBIN_AGENT_ID: agent.id,
            ROUNDROBIN_AGENT_NAME: agent.name,
            ROUNDROBIN_HUB_URL: `ws://localhost:${PORT}/ws`,
            ROUNDROBIN_HUB_HTTP: `http://localhost:${PORT}`,
          },
        },
      },
    };
    writeFileSync(join(workspace, ".mcp.json"), JSON.stringify(mcpConfig, null, 2));

    // Pre-accept folder trust
    const claudeSettingsDir = join(workspace, ".claude");
    mkdirSync(claudeSettingsDir, { recursive: true });
    writeFileSync(join(claudeSettingsDir, "settings.json"), JSON.stringify({
      permissions: { defaultMode: "bypassPermissions" },
    }));

    // CLAUDE.md — everything the agent needs, assembled from the bundle
    const claudeMd = `${agent.systemPrompt}\n\n## Your Identity\n${agent.soul}\n\n## The Task\n${taskContent}\n\n## Agent Roster\n${roster}\n\n## Your Workspace\nYou are ${agent.name} (${agent.id}). You are working in an isolated copy of the repository.\nOther agents have their own copies. Coordinate via the chat channel.\n`;
    writeFileSync(join(workspace, "CLAUDE.md"), claudeMd);
  }

  // Bootstrap + spawn + wait via runner (grouped by variant)
  const variantGroups = new Map<string, typeof agents>();
  for (const agent of agents) {
    const group = variantGroups.get(agent.variant) ?? [];
    group.push(agent);
    variantGroups.set(agent.variant, group);
  }

  for (const [variant, group] of variantGroups) {
    const runner = createRunner(variant, systemMessage);
    await runner.bootstrap(group, runDir);
    for (const agent of group) {
      await runner.spawn(agent, join(runDir, "workspaces", agent.id));
    }
    await Promise.all(group.map(agent => runner.waitUntilReady(agent)));
  }

  await Bun.sleep(500);

  // Prompt all agents to begin
  for (const agent of agents) {
    await $`tmux send-keys -t ${agent.session} ${"Read CLAUDE.md and begin working on the task. Check in with other agents via the chat."} Enter`.quiet();
  }

  currentRun = { status: "running", runDir, agents, startedAt: new Date().toISOString() };
  systemMessage(`Run started with ${agents.length} agent(s)`);
}

async function killRun() {
  if (currentRun.status !== "running") return;
  for (const agent of currentRun.agents) {
    try {
      const runner = createRunner(agent.variant, systemMessage);
      await runner.kill(agent);
    } catch {}
  }
  systemMessage("Run killed");
  currentRun = { status: "idle", agents: [] };
}

// --- Server ---
const server = Bun.serve({
  port: PORT,
  async fetch(req, server) {
    const url = new URL(req.url);

    // WebSocket upgrade
    if (url.pathname === "/ws" || url.pathname === "/ws/ui") {
      const agentId = url.searchParams.get("agent_id");
      const isUi = url.pathname === "/ws/ui";
      const ok = server.upgrade(req, { data: { agentId, isUi } });
      return ok ? undefined : new Response("WebSocket upgrade failed", { status: 400 });
    }

    // SSE log stream: /api/logs/<agent-id>
    const logsMatch = url.pathname.match(/^\/api\/logs\/(.+)$/);
    if (req.method === "GET" && logsMatch) {
      const agentId = logsMatch[1];
      const agent = currentRun.agents.find(a => a.id === agentId);
      if (!agent) return Response.json({ error: "Agent not found" }, { status: 404 });

      let cancelled = false;
      const stream = new ReadableStream({
        async start(controller) {
          let lastContent = "";
          const enc = new TextEncoder();
          while (!cancelled) {
            try {
              const { stdout } = await $`tmux capture-pane -t ${agent.session} -p -S -500`.quiet();
              const content = stdout.toString();
              if (content !== lastContent) {
                const entry = JSON.stringify({
                  ts: new Date().toISOString(),
                  agent_id: agentId,
                  type: "pane",
                  content,
                });
                controller.enqueue(enc.encode(`data: ${entry}\n\n`));
                lastContent = content;
              }
            } catch {
              const entry = JSON.stringify({ ts: new Date().toISOString(), agent_id: agentId, type: "error", content: "Session not available" });
              controller.enqueue(enc.encode(`data: ${entry}\n\n`));
              break;
            }
            await Bun.sleep(1000);
          }
          controller.close();
        },
        cancel() { cancelled = true; },
      });

      return new Response(stream, {
        headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" },
      });
    }

    // Snapshot log (non-streaming): /api/logs-snapshot/<agent-id>
    const snapMatch = url.pathname.match(/^\/api\/logs-snapshot\/(.+)$/);
    if (req.method === "GET" && snapMatch) {
      const agentId = snapMatch[1];
      const agent = currentRun.agents.find(a => a.id === agentId);
      if (!agent) return Response.json({ error: "Agent not found" }, { status: 404 });
      try {
        const { stdout } = await $`tmux capture-pane -t ${agent.session} -p -S -500`.quiet();
        return Response.json({ agent_id: agentId, ts: new Date().toISOString(), content: stdout.toString() });
      } catch {
        return Response.json({ error: "Session not available" }, { status: 500 });
      }
    }

    // HTTP routes
    if (req.method === "GET" && url.pathname === "/") {
      return new Response(uiHtml, { headers: { "Content-Type": "text/html" } });
    }

    if (req.method === "GET" && url.pathname === "/api/messages") {
      const limit = parseInt(url.searchParams.get("limit") || "200");
      const rows = getMessages.all(limit);
      return Response.json((rows as any[]).reverse());
    }

    if (req.method === "GET" && url.pathname === "/api/agents") {
      const agents = Array.from(agentSockets.keys()).map((id) => ({
        id,
        connected: true,
        consensus: consensusState.get(id) || null,
      }));
      return Response.json(agents);
    }

    if (req.method === "GET" && url.pathname === "/api/status") {
      const totalMessages = db.prepare("SELECT COUNT(*) as count FROM messages").get() as any;
      return Response.json({
        status: "ok",
        agents: agentSockets.size,
        uiClients: uiSockets.size,
        messages: totalMessages.count,
        consensus: Object.fromEntries(consensusState),
      });
    }

    if (req.method === "POST" && url.pathname === "/api/message") {
      const body = await req.json();
      const msg = {
        agent_id: body.agent_id || "human",
        content: body.content,
        type: "message",
        timestamp: new Date().toISOString(),
      };
      storeMessage(msg);
      broadcastAll(msg);
      return Response.json({ ok: true });
    }

    // --- Config endpoints ---

    if (req.method === "GET" && url.pathname === "/api/run") {
      return Response.json(currentRun);
    }

    if (req.method === "POST" && url.pathname === "/api/run") {
      if (currentRun.status === "running") {
        return Response.json({ error: "A run is already in progress" }, { status: 409 });
      }
      const bundle = await req.json() as RunBundle;
      if (!bundle.taskContent || !bundle.agents?.length) {
        return Response.json({ error: "Need taskContent and agents" }, { status: 400 });
      }
      startRun(bundle).catch(e => {
        systemMessage(`Run failed: ${e.message}`);
        currentRun = { status: "idle", agents: [] };
      });
      return Response.json({ ok: true, status: "starting" });
    }

    if (req.method === "POST" && url.pathname === "/api/run/kill") {
      await killRun();
      return Response.json({ ok: true });
    }

    return new Response("Not Found", { status: 404 });
  },
  websocket: {
    open(ws: any) {
      const { agentId, isUi } = ws.data;
      if (isUi) {
        uiSockets.add(ws);
      } else if (agentId) {
        agentSockets.set(agentId, ws);
        systemMessage(`${agentId} has joined the chat`);
      }
    },
    message(ws: any, raw: string | Buffer) {
      const data = typeof raw === "string" ? raw : raw.toString();
      let msg: any;
      try {
        msg = JSON.parse(data);
      } catch {
        return;
      }

      const { agentId, isUi } = ws.data;
      if (isUi) return; // UI is receive-only

      msg.agent_id = agentId;
      msg.timestamp = msg.timestamp || new Date().toISOString();
      msg.type = msg.type || "message";
      if (!msg.content) return; // ignore malformed messages

      // Handle consensus
      if (msg.type === "consensus") {
        consensusState.set(agentId, msg.content);
        storeMessage(msg);
        broadcast(msg, ws);

        // Check if all agents reached consensus
        if (consensusState.size === agentSockets.size && agentSockets.size > 0) {
          systemMessage("All agents have reached consensus");
        }
        return;
      }

      // Handle consensus retraction
      if (msg.type === "message" && consensusState.has(agentId)) {
        const lower = msg.content.toLowerCase();
        if (lower.includes("retract") && lower.includes("consensus")) {
          consensusState.delete(agentId);
          systemMessage(`${agentId} has retracted their consensus`);
        }
      }

      storeMessage(msg);
      broadcast(msg, ws);
    },
    close(ws: any) {
      const { agentId, isUi } = ws.data;
      if (isUi) {
        uiSockets.delete(ws);
      } else if (agentId) {
        agentSockets.delete(agentId);
        systemMessage(`${agentId} has left the chat`);
      }
    },
  },
});

console.log(`roundRobin hub running on http://localhost:${server.port}`);
