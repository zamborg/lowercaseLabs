import { existsSync, readFileSync, writeFileSync } from "fs";
import { cp, mkdir } from "fs/promises";
import http from "node:http";
import { spawn } from "node:child_process";
import { basename, join, resolve } from "path";
import { ensureDir, shellEscape } from "./lib.js";

function runCapture(command, args, options = {}) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env || process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", rejectPromise);
    child.on("close", (code) => {
      if (code === 0) resolvePromise({ stdout, stderr });
      else rejectPromise(new Error(stderr || `${command} exited with code ${code}`));
    });
  });
}

function sleep(ms) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms));
}

const specPath = process.env.RUFFLE_RUNSPEC || process.argv[2];
if (!specPath) {
  console.error("Missing runspec path. Pass /run/runspec.json or set RUFFLE_RUNSPEC.");
  process.exit(1);
}

const spec = JSON.parse(readFileSync(specPath, "utf-8"));
const runRoot = "/run";
const projectRoot = "/project";
const sourceDir = resolve(projectRoot, spec.sourceDirRel);
const originalTaskPath = resolve(projectRoot, spec.originalTaskRel);
const keepAliveSecondsAfterExit = Number.isFinite(spec.keepAliveSecondsAfterExit)
  ? spec.keepAliveSecondsAfterExit
  : 120;
let shutdownTimer = null;

const state = {
  runId: spec.runId,
  phase: "booting",
  startedAt: new Date().toISOString(),
  finishedAt: null,
  shutdownAt: null,
  error: null,
  runRoot,
  sourceDir,
  originalTask: originalTaskPath,
  mappers: spec.mappers.map((mapper) => ({
    id: mapper.id,
    name: mapper.name,
    session: `ruffle-${mapper.id}`,
    workspace: join(runRoot, "workspaces", mapper.id),
    status: "pending",
    exitCode: null,
  })),
  reducer: null,
};

const uiHtml = readFileSync(join(import.meta.dirname, "ui", "index.html"), "utf-8");

function mapperTaskBody(mapper) {
  return [
    "# Map Task",
    "",
    `You are ${mapper.name} (${mapper.id}).`,
    "",
    spec.mapTask,
    "",
    "## Output Contract",
    "",
    "- Work only inside your local workspace.",
    "- Write a short RESULT.md in the workspace root.",
  ].join("\n");
}

function reducerTaskBody() {
  const successfulMappers = state.mappers.filter((mapper) => mapper.status === "complete");
  const failedMappers = state.mappers.filter((mapper) => mapper.status === "failed");
  return [
    "# Reduce Task",
    "",
    `You are ${spec.reducer.name} (${spec.reducer.id}).`,
    "",
    spec.reduceTask,
    "",
    "## Workspace Layout",
    "",
    "- The current directory contains `workspaces/mapper-*`.",
    "- Each mapper workspace may contain artifacts and a RESULT.md file.",
    successfulMappers.length > 0
      ? `- Successful mappers: ${successfulMappers.map((mapper) => `${mapper.id} (${mapper.name})`).join(", ")}.`
      : "- Successful mappers: none.",
    failedMappers.length > 0
      ? `- Failed mappers: ${failedMappers.map((mapper) => `${mapper.id} (${mapper.name})`).join(", ")}. Ignore these unless their existing workspace files are still useful.`
      : "- Failed mappers: none.",
    "- Write the final RESULT.md in the current directory.",
  ].join("\n");
}

async function copySourceIntoWorkspace(workspace) {
  await cp(sourceDir, workspace, { recursive: true, force: true });
}

async function prepareWorkspacePermissions() {
  await runCapture("bash", ["-lc", "chmod -R a+rwX /run"]);
}

function doneFile(workspace) {
  return join(workspace, ".ruffle-done");
}

function exitCodeFile(workspace) {
  return join(workspace, ".ruffle-exit-code");
}

function readExitCode(workspace) {
  const file = exitCodeFile(workspace);
  if (!existsSync(file)) return null;
  const value = readFileSync(file, "utf-8").trim();
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function shouldRunAsClaudeUser(commandRel) {
  return basename(commandRel).startsWith("claude");
}

async function launchSession(agent, commandRel) {
  const scriptPath = resolve(projectRoot, commandRel);
  const wrappedCommand = shouldRunAsClaudeUser(commandRel)
    ? `su -s /bin/bash claude -c 'cd ${shellEscape(agent.workspace)} && ${shellEscape(scriptPath)} task.md'`
    : `${shellEscape(scriptPath)} task.md`;
  const innerCommand = [
    wrappedCommand,
    "exit_code=$?",
    `printf '%s\\n' \"$exit_code\" > ${shellEscape(exitCodeFile(agent.workspace))}`,
    `touch ${shellEscape(doneFile(agent.workspace))}`,
    "printf '\\n[ruffle] exit=%s\\n' \"$exit_code\"",
    "exec bash",
  ].join("; ");
  await runCapture("tmux", ["new-session", "-d", "-s", agent.session, "-x", "220", "-y", "60", "-c", agent.workspace, "bash", "-lc", innerCommand]);
}

async function waitForAgent(agent) {
  while (!existsSync(doneFile(agent.workspace))) {
    await sleep(500);
  }
  agent.exitCode = readExitCode(agent.workspace);
  agent.status = agent.exitCode === 0 ? "complete" : "failed";
}

async function runMappers() {
  state.phase = "preparing";
  ensureDir(join(runRoot, "workspaces"));

  for (let index = 0; index < state.mappers.length; index += 1) {
    const mapperState = state.mappers[index];
    const mapperSpec = spec.mappers[index];
    await copySourceIntoWorkspace(mapperState.workspace);
    writeFileSync(join(mapperState.workspace, "task.md"), mapperTaskBody(mapperSpec));
    mapperState.status = "running";
    await launchSession(mapperState, mapperSpec.commandRel);
  }

  state.phase = "mapping";
  await Promise.all(state.mappers.map((mapper) => waitForAgent(mapper)));
  if (!state.mappers.some((mapper) => mapper.status === "complete")) {
    throw new Error("No mapper agents completed successfully.");
  }
}

async function runReducer() {
  state.phase = "reducing";
  const reducerWorkspace = runRoot;
  writeFileSync(join(reducerWorkspace, "task.md"), reducerTaskBody());
  state.reducer = {
    id: spec.reducer.id,
    name: spec.reducer.name,
    session: "ruffle-reducer",
    workspace: reducerWorkspace,
    status: "running",
    exitCode: null,
  };
  await launchSession(state.reducer, spec.reducer.commandRel);
  await waitForAgent(state.reducer);
  if (state.reducer.status === "failed") {
    throw new Error("Reducer agent failed.");
  }
}

async function capturePanes() {
  const panes = {};
  const sessions = [
    ...state.mappers.map((mapper) => ({
      id: mapper.id,
      title: mapper.name,
      session: mapper.session,
      status: mapper.status,
    })),
    ...(state.reducer ? [{
      id: state.reducer.id,
      title: state.reducer.name,
      session: state.reducer.session,
      status: state.reducer.status,
    }] : []),
  ];

  for (const session of sessions) {
    let content = "";
    try {
      const result = await runCapture("tmux", ["capture-pane", "-t", session.session, "-p", "-S", "-120"]);
      content = result.stdout;
    } catch {
      content = session.status === "pending" ? "Not started yet." : "Session unavailable.";
    }
    panes[session.id] = {
      title: session.title,
      content,
      status: session.status,
    };
  }

  return panes;
}

function scheduleShutdown() {
  if (shutdownTimer || keepAliveSecondsAfterExit < 0) return;

  state.shutdownAt = new Date(Date.now() + (keepAliveSecondsAfterExit * 1000)).toISOString();
  shutdownTimer = setTimeout(() => {
    server.close(() => {
      process.exit(0);
    });
    setTimeout(() => {
      process.exit(0);
    }, 1000);
  }, Math.max(0, keepAliveSecondsAfterExit * 1000));
}

async function startRuntime() {
  try {
    await mkdir(join(runRoot, "workspaces"), { recursive: true });
    await prepareWorkspacePermissions();
    writeFileSync(join(runRoot, "full-task.md"), spec.fullTask);
    writeFileSync(join(runRoot, "map-task.md"), spec.mapTask);
    writeFileSync(join(runRoot, "reduce-task.md"), spec.reduceTask);
    await runMappers();
    await runReducer();
    state.phase = "complete";
    state.finishedAt = new Date().toISOString();
  } catch (error) {
    state.phase = "failed";
    state.error = error instanceof Error ? error.message : String(error);
    state.finishedAt = new Date().toISOString();
  }
  scheduleShutdown();
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || "/", "http://localhost:3000");
  if (request.method === "GET" && url.pathname === "/") {
    response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    response.end(uiHtml);
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/status") {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify(state));
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/panes") {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify(await capturePanes()));
    return;
  }
  response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  response.end("Not found");
});

server.listen(3000);

void startRuntime();
