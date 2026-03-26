import { readFileSync, readdirSync, existsSync, writeFileSync, mkdirSync, rmSync } from "fs";
import { join, resolve, dirname } from "path";

// Recursively collect all files in a directory as { relPath -> base64 }
function collectFiles(baseDir: string, dir: string, out: Record<string, string>) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    const rel = full.slice(baseDir.length + 1);
    if (entry.isDirectory()) collectFiles(baseDir, full, out);
    else out[rel] = readFileSync(full).toString("base64");
  }
}
import { $ } from "bun";
import { parseAgentConfig, serializeAgentConfig, SUPPORTED_MODELS, DEFAULT_MODEL } from "../agentConfig.ts";

const PORT = parseInt(Bun.env.CONTROL_PORT || "8080");
const PROJECT_ROOT = resolve(dirname(import.meta.dir), "..");
const TASKS_DIR = join(PROJECT_ROOT, "tasks");
const SOULS_DIR = join(PROJECT_ROOT, "souls");
const PROMPTS_DIR = join(PROJECT_ROOT, "prompts");
const IMAGE_NAME = "roundrobin";
const BASE_RUN_PORT = 9100;

// --- UI ---
const uiPath = join(import.meta.dir, "ui", "index.html");
let uiHtml: string;
try {
  uiHtml = readFileSync(uiPath, "utf-8");
} catch {
  uiHtml = "<html><body><h1>roundRobin control</h1><p>UI not found</p></body></html>";
}

// --- Active runs: { id, task, souls, port, containerId, startedAt } ---
interface Run {
  id: string;
  task: string;
  souls: string[];
  port: number;
  containerId: string;
  startedAt: string;
  url: string;
}
const activeRuns = new Map<string, Run>();

async function findFreePort(): Promise<number> {
  const used = new Set(Array.from(activeRuns.values()).map(r => r.port));
  for (let p = BASE_RUN_PORT; p < BASE_RUN_PORT + 100; p++) {
    if (!used.has(p)) return p;
  }
  throw new Error("No free ports available");
}

async function waitForHttp(url: string, timeoutMs = 30000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {}
    await Bun.sleep(1000);
  }
  throw new Error(`Timeout waiting for ${url}`);
}

async function ensureImage(): Promise<{ built: boolean }> {
  // Always rebuild — ensures Dockerfile changes are picked up
  // Could add a hash check later if build time becomes annoying
  try {
    const { stdout } = await $`docker images -q ${IMAGE_NAME}`.quiet();
    if (stdout.toString().trim()) return { built: false };
  } catch {}
  console.log("Image not found — building (this may take a minute)...");
  await $`docker build -t ${IMAGE_NAME} ${PROJECT_ROOT}`;
  return { built: true };
}

async function startRun(taskName: string, soulFiles: string[]): Promise<Run & { imageBuilt?: boolean }> {
  const { built } = await ensureImage();
  const port = await findFreePort();

  // --- Read everything locally before launching the container ---

  // Task content
  const taskPath = join(TASKS_DIR, taskName, "task.md");
  if (!existsSync(taskPath)) throw new Error(`task.md not found in tasks/${taskName}`);
  const taskContent = readFileSync(taskPath, "utf-8");

  // Data directory (respects data_dir frontmatter override)
  const taskDir = join(TASKS_DIR, taskName);
  let dataPath = join(taskDir, "data");
  const fmMatch = taskContent.match(/^---\n([\s\S]*?)\n---/);
  if (fmMatch) {
    const dataLine = fmMatch[1].split("\n").find(l => l.startsWith("data_dir:"));
    if (dataLine) {
      const rawDir = dataLine.replace("data_dir:", "").trim();
      dataPath = rawDir.startsWith("/") ? rawDir : join(taskDir, rawDir);
    }
  }
  const dataFiles: Record<string, string> = {};
  if (existsSync(dataPath)) collectFiles(dataPath, dataPath, dataFiles);

  // Soul + system prompt content per agent
  const agents = soulFiles.map(soulFile => {
    const cfg = parseAgentConfig(join(SOULS_DIR, soulFile));
    const promptPath = join(PROJECT_ROOT, "prompts", cfg.system_prompt || "default.md");
    const systemPrompt = existsSync(promptPath) ? readFileSync(promptPath, "utf-8") : "";
    return { name: cfg.name, variant: cfg.variant, model: cfg.model, soul: cfg.soul, systemPrompt };
  });

  const oauthToken = Bun.env.CLAUDE_CODE_OAUTH_TOKEN || "";
  if (!oauthToken) console.warn("Warning: CLAUDE_CODE_OAUTH_TOKEN not set — agents will fail to authenticate");

  // Launch a blank engine container — no task/soul/prompt mounts needed
  const { stdout } = await $`docker run -d \
    -p ${port}:9090 \
    -e CLAUDE_CODE_OAUTH_TOKEN=${oauthToken} \
    -v ${join(PROJECT_ROOT, "runs")}:/app/runs \
    ${IMAGE_NAME}`.quiet();

  const containerId = stdout.toString().trim();
  const url = `http://localhost:${port}`;

  await waitForHttp(`${url}/api/status`, 30000);

  // Send the full bundle — container needs no host filesystem knowledge
  await fetch(`${url}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ taskName, taskContent, dataFiles, agents }),
  });

  const run = {
    id: containerId.slice(0, 12),
    task: taskName,
    souls: soulFiles,
    port,
    containerId,
    startedAt: new Date().toISOString(),
    url,
    imageBuilt: built,
  };
  activeRuns.set(run.id, run as Run);

  // Tail container logs to control server stdout
  $`docker logs -f ${containerId}`.then(() => {}).catch(() => {});

  return run;
}

async function killRun(id: string): Promise<void> {
  const run = activeRuns.get(id);
  if (!run) return;
  try { await $`docker rm -f ${run.containerId}`.quiet(); } catch {}
  activeRuns.delete(id);
}

// --- Task helpers ---
function listTasks() {
  if (!existsSync(TASKS_DIR)) return [];
  return readdirSync(TASKS_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => {
      const taskMd = join(TASKS_DIR, d.name, "task.md");
      const dataDir = join(TASKS_DIR, d.name, "data");
      return {
        name: d.name,
        hasTask: existsSync(taskMd),
        hasData: existsSync(dataDir),
      };
    });
}

function readTaskMd(name: string): string {
  const p = join(TASKS_DIR, name, "task.md");
  return existsSync(p) ? readFileSync(p, "utf-8") : "";
}

function writeTaskMd(name: string, content: string) {
  const dir = join(TASKS_DIR, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "task.md"), content);
}

function deleteTask(name: string) {
  const dir = join(TASKS_DIR, name);
  if (existsSync(dir)) rmSync(dir, { recursive: true, force: true });
}

// --- Agent helpers ---
function listAgents() {
  if (!existsSync(SOULS_DIR)) return [];
  return readdirSync(SOULS_DIR)
    .filter(f => f.endsWith(".md"))
    .map(f => {
      const cfg = parseAgentConfig(join(SOULS_DIR, f));
      const content = existsSync(join(SOULS_DIR, f)) ? readFileSync(join(SOULS_DIR, f), "utf-8") : "";
      return { ...cfg, content };
    });
}

function writeAgent(file: string, content: string) {
  mkdirSync(SOULS_DIR, { recursive: true });
  writeFileSync(join(SOULS_DIR, file), content);
}

function deleteAgent(file: string) {
  const p = join(SOULS_DIR, file);
  if (existsSync(p)) rmSync(p);
}

// --- Prompt helpers ---
function listPrompts() {
  if (!existsSync(PROMPTS_DIR)) return [];
  return readdirSync(PROMPTS_DIR)
    .filter(f => f.endsWith(".md"))
    .map(f => ({ file: f }));
}

function readPrompt(file: string): string {
  const p = join(PROMPTS_DIR, file);
  return existsSync(p) ? readFileSync(p, "utf-8") : "";
}

function writePrompt(file: string, content: string) {
  mkdirSync(PROMPTS_DIR, { recursive: true });
  writeFileSync(join(PROMPTS_DIR, file), content);
}

function deletePrompt(file: string) {
  const p = join(PROMPTS_DIR, file);
  if (existsSync(p)) rmSync(p);
}

// --- Server ---
Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname;

    if (req.method === "GET" && path === "/") {
      return new Response(uiHtml, { headers: { "Content-Type": "text/html" } });
    }

    // --- Tasks ---
    if (req.method === "GET" && path === "/api/tasks") {
      return Response.json(listTasks());
    }

    const taskMatch = path.match(/^\/api\/tasks\/([^/]+)$/);
    if (taskMatch) {
      const name = decodeURIComponent(taskMatch[1]);
      if (req.method === "GET") {
        return Response.json({ name, content: readTaskMd(name) });
      }
      if (req.method === "PUT") {
        const { content } = await req.json();
        writeTaskMd(name, content);
        return Response.json({ ok: true });
      }
      if (req.method === "DELETE") {
        deleteTask(name);
        return Response.json({ ok: true });
      }
    }

    if (req.method === "POST" && path === "/api/tasks") {
      const { name, content } = await req.json();
      if (!name) return Response.json({ error: "name required" }, { status: 400 });
      writeTaskMd(name, content || `# Task: ${name}\n\n## Problem\n\n## Requirements\n\n## Acceptance Criteria\n`);
      return Response.json({ ok: true, name });
    }

    // --- Agents ---
    if (req.method === "GET" && path === "/api/souls") {
      return Response.json(listAgents());
    }

    const soulMatch = path.match(/^\/api\/souls\/([^/]+)$/);
    if (soulMatch) {
      const file = decodeURIComponent(soulMatch[1]);
      if (req.method === "GET") {
        const p = join(SOULS_DIR, file);
        return Response.json({ file, content: existsSync(p) ? readFileSync(p, "utf-8") : "" });
      }
      if (req.method === "PUT") {
        const { content } = await req.json();
        writeAgent(file, content);
        return Response.json({ ok: true });
      }
      if (req.method === "DELETE") {
        deleteAgent(file);
        return Response.json({ ok: true });
      }
    }

    if (req.method === "POST" && path === "/api/souls") {
      const { name } = await req.json();
      if (!name) return Response.json({ error: "name required" }, { status: 400 });
      const file = name.toLowerCase().replace(/\s+/g, "-") + ".md";
      const content = serializeAgentConfig({ file, name, model: DEFAULT_MODEL, system_prompt: "default.md", soul: "## Personality\n\n## Focus\n\n## Communication Style\n" });
      writeAgent(file, content);
      return Response.json({ ok: true, file });
    }

    // --- Prompts ---
    if (req.method === "GET" && path === "/api/prompts") {
      return Response.json(listPrompts());
    }

    const promptMatch = path.match(/^\/api\/prompts\/([^/]+)$/);
    if (promptMatch) {
      const file = decodeURIComponent(promptMatch[1]);
      if (req.method === "GET") {
        return Response.json({ file, content: readPrompt(file) });
      }
      if (req.method === "PUT") {
        const { content } = await req.json();
        writePrompt(file, content);
        return Response.json({ ok: true });
      }
      if (req.method === "DELETE") {
        deletePrompt(file);
        return Response.json({ ok: true });
      }
    }

    if (req.method === "POST" && path === "/api/prompts") {
      const { name } = await req.json();
      if (!name) return Response.json({ error: "name required" }, { status: 400 });
      const file = name.toLowerCase().replace(/\s+/g, "-") + ".md";
      writePrompt(file, `# ${name}\n\nYou are a skilled software engineer...\n`);
      return Response.json({ ok: true, file });
    }

    // --- Runs ---
    if (req.method === "GET" && path === "/api/runs") {
      return Response.json(Array.from(activeRuns.values()));
    }

    if (req.method === "POST" && path === "/api/runs") {
      const { task, souls } = await req.json() as { task: string; souls: string[] };
      if (!task || !souls?.length) {
        return Response.json({ error: "task and souls required" }, { status: 400 });
      }
      try {
        const run = await startRun(task, souls);
        return Response.json(run);
      } catch (e: any) {
        return Response.json({ error: e.message }, { status: 500 });
      }
    }

    const runKillMatch = path.match(/^\/api\/runs\/([^/]+)\/kill$/);
    if (req.method === "POST" && runKillMatch) {
      await killRun(runKillMatch[1]);
      return Response.json({ ok: true });
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`roundRobin control plane: http://localhost:${PORT}`);
