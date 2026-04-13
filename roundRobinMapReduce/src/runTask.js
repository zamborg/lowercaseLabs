import { dirname, join, resolve } from "path";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { spawn } from "node:child_process";
import { findFreePort, mustStayInProject, parseTaskSections, readText, resolveFrom, timestampId, waitForHttp } from "./lib.js";

const projectRoot = process.cwd();
const FORWARDED_ENV_VARS = [
  "ANTHROPIC_API_KEY",
  "CLAUDE_CODE_OAUTH_TOKEN",
  "OPENAI_API_KEY",
  "GEMINI_API_KEY",
  "GOOGLE_API_KEY",
  "GOOGLE_CLOUD_PROJECT",
  "GOOGLE_GENAI_USE_VERTEXAI",
  "RUFFLE_CLAUDE_MODEL",
  "RUFFLE_CODEX_MODEL",
  "RUFFLE_GEMINI_MODEL",
];

const SHELL_ENV_FILES = [
  ".zshenv",
  ".zprofile",
  ".zshrc",
  ".bash_profile",
  ".bashrc",
  ".profile",
];

const HOME_MOUNT_CANDIDATES = [
  { host: ".codex", container: "/root/.codex", writable: true },
  { host: ".gemini", container: "/root/.gemini", writable: true },
  { host: ".claude", container: "/home/claude/.claude", writable: true },
  { host: ".claude.json", container: "/home/claude/.claude.json", writable: true },
];

function runLogged(command, args, options = {}) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      stdio: "inherit",
      env: options.env || process.env,
    });
    child.on("error", rejectPromise);
    child.on("close", (code) => {
      if (code === 0) resolvePromise(undefined);
      else rejectPromise(new Error(`${command} exited with code ${code}`));
    });
  });
}

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

async function queryShellForEnv(variableName) {
  const shell = process.env.SHELL || "/bin/zsh";
  const probe = `source ~/.zshenv >/dev/null 2>&1 || true; source ~/.zprofile >/dev/null 2>&1 || true; source ~/.zshrc >/dev/null 2>&1 || true; value=$(printenv ${variableName} 2>/dev/null || true); if [ -n "$value" ]; then printf '%s' "$value"; fi`;

  try {
    const { stdout } = await runCapture(shell, ["-lc", probe]);
    const value = stdout.trim();
    return value || null;
  } catch {
    return null;
  }
}

function parseSimpleShellAssignments(content, variableName) {
  const patterns = [
    new RegExp(`^\\s*export\\s+${variableName}=(.*)\\s*$`, "m"),
    new RegExp(`^\\s*${variableName}=(.*)\\s*$`, "m"),
  ];

  for (const pattern of patterns) {
    const match = content.match(pattern);
    if (!match) continue;
    let value = match[1].trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (value && !value.includes("$(") && !value.includes("`")) {
      return value;
    }
  }

  return null;
}

function readEnvVarFromShellFiles(variableName) {
  const home = process.env.HOME;
  if (!home) return null;

  for (const file of SHELL_ENV_FILES) {
    const absolutePath = join(home, file);
    if (!existsSync(absolutePath)) continue;
    try {
      const content = readFileSync(absolutePath, "utf-8");
      const value = parseSimpleShellAssignments(content, variableName);
      if (value) return value;
    } catch {
      // Ignore unreadable shell files.
    }
  }

  return null;
}

async function resolveForwardedEnv() {
  const resolved = {};
  for (const name of FORWARDED_ENV_VARS) {
    const direct = process.env[name];
    if (direct) {
      resolved[name] = direct;
      continue;
    }
    const fromShell = readEnvVarFromShellFiles(name);
    if (fromShell) {
      resolved[name] = fromShell;
      continue;
    }
    const fromRuntimeShell = await queryShellForEnv(name);
    if (fromRuntimeShell) {
      resolved[name] = fromRuntimeShell;
    }
  }
  return resolved;
}

function resolveHomeMountArgs() {
  const home = process.env.HOME;
  if (!home) return [];

  return HOME_MOUNT_CANDIDATES
    .map((entry) => ({
      ...entry,
      absoluteHost: join(home, entry.host),
    }))
    .filter((entry) => existsSync(entry.absoluteHost))
    .flatMap((entry) => ["-v", `${entry.absoluteHost}:${entry.container}${entry.writable ? "" : ":ro"}`]);
}

function usage() {
  console.error("Usage: node src/runTask.js <config.json>");
  process.exit(1);
}

function requireFile(path, label) {
  if (!existsSync(path)) {
    throw new Error(`${label} not found: ${path}`);
  }
}

async function main() {
  const configArg = process.argv[2];
  if (!configArg) usage();

  const configPath = resolve(projectRoot, configArg);
  requireFile(configPath, "Config");
  const configDir = dirname(configPath);
  const config = JSON.parse(readText(configPath));

  if (!Array.isArray(config.mappers) || config.mappers.length === 0) {
    throw new Error("Config must define at least one mapper.");
  }
  if (!config.reducer) {
    throw new Error("Config must define a reducer.");
  }

  const sourceDir = resolveFrom(configDir, config.sourceDir);
  const taskPath = resolveFrom(configDir, config.task);
  requireFile(taskPath, "Task file");
  if (!existsSync(sourceDir)) {
    throw new Error(`Source directory not found: ${sourceDir}`);
  }

  const fullTask = readText(taskPath);
  const { mapTask, reduceTask } = parseTaskSections(fullTask);

  const imageTag = config.imageTag || "ruffle-mapreduce";
  const port = config.port || await findFreePort();
  const keepAliveSecondsAfterExit = Number.isFinite(config.keepAliveSecondsAfterExit)
    ? config.keepAliveSecondsAfterExit
    : 120;
  const runId = timestampId();
  const runRoot = join(projectRoot, "runs", runId);
  mkdirSync(runRoot, { recursive: true });

  const mappers = config.mappers.map((mapper, index) => {
    const commandAbs = resolveFrom(configDir, mapper.command);
    requireFile(commandAbs, `Mapper command for ${mapper.name}`);
    return {
      id: `mapper-${index + 1}`,
      name: mapper.name,
      command: mapper.command,
      commandRel: mustStayInProject(projectRoot, commandAbs),
    };
  });

  const reducerCommandAbs = resolveFrom(configDir, config.reducer.command);
  requireFile(reducerCommandAbs, "Reducer command");

  const spec = {
    runId,
    imageTag,
    port,
    keepAliveSecondsAfterExit,
    sourceDirRel: mustStayInProject(projectRoot, sourceDir),
    originalTaskRel: mustStayInProject(projectRoot, taskPath),
    fullTask,
    mapTask,
    reduceTask,
    mappers,
    reducer: {
      id: "reducer",
      name: config.reducer.name,
      command: config.reducer.command,
      commandRel: mustStayInProject(projectRoot, reducerCommandAbs),
    },
  };

  writeFileSync(join(runRoot, "runspec.json"), JSON.stringify(spec, null, 2));

  console.log(`Building Docker image ${imageTag}...`);
  await runLogged("docker", ["build", "-t", imageTag, projectRoot]);

  console.log(`Starting run ${runId} on http://localhost:${port}`);
  const forwardedEnv = await resolveForwardedEnv();
  const envArgs = Object.entries(forwardedEnv)
    .flatMap(([name, value]) => ["-e", `${name}=${value}`]);
  const homeMountArgs = resolveHomeMountArgs();

  const { stdout: containerStdout } = await runCapture("docker", [
    "run",
    "-d",
    "--rm",
    "-p",
    `${port}:3000`,
    "-v",
    `${projectRoot}:/project:ro`,
    "-v",
    `${runRoot}:/run`,
    ...homeMountArgs,
    ...envArgs,
    "-e",
    "RUFFLE_RUNSPEC=/run/runspec.json",
    imageTag,
  ]);

  const containerId = containerStdout.trim();
  writeFileSync(join(runRoot, "launch.json"), JSON.stringify({
    containerId,
    url: `http://localhost:${port}`,
    configPath,
  }, null, 2));

  await waitForHttp(`http://localhost:${port}/api/status`);

  console.log(`Viewer: http://localhost:${port}`);
  console.log(`Run dir: ${runRoot}`);
  console.log(`Container: ${containerId}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
