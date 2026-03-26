import { mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname, relative, resolve, sep } from "path";
import { createServer } from "node:net";

export function timestampId(date = new Date()) {
  const parts = [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
    "-",
    String(date.getHours()).padStart(2, "0"),
    String(date.getMinutes()).padStart(2, "0"),
    String(date.getSeconds()).padStart(2, "0"),
  ];
  return `run-${parts.join("")}-${Math.random().toString(36).slice(2, 8)}`;
}

export async function findFreePort(start = 9310, end = 9399) {
  for (let port = start; port <= end; port += 1) {
    const isOpen = await new Promise((resolvePromise) => {
      const server = createServer();
      server.once("error", () => resolvePromise(false));
      server.listen(port, "127.0.0.1", () => {
        server.close(() => resolvePromise(true));
      });
    });
    if (isOpen) return port;
  }
  throw new Error(`No free port available in ${start}-${end}`);
}

export function ensureDir(path) {
  mkdirSync(path, { recursive: true });
}

export function writeText(path, content) {
  ensureDir(dirname(path));
  writeFileSync(path, content);
}

export function readText(path) {
  return readFileSync(path, "utf-8");
}

export function mustStayInProject(projectRoot, absolutePath) {
  const rel = relative(projectRoot, absolutePath);
  if (rel.startsWith("..") || rel === "" && absolutePath !== projectRoot) {
    throw new Error(`Path must stay within project root: ${absolutePath}`);
  }
  if (rel.startsWith(`..${sep}`) || rel === "..") {
    throw new Error(`Path must stay within project root: ${absolutePath}`);
  }
  return rel || ".";
}

export function resolveFrom(baseDir, maybeRelative) {
  return resolve(baseDir, maybeRelative);
}

export function shellEscape(value) {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

export function parseTaskSections(task) {
  const match = task.match(/## Map Task\s*([\s\S]*?)## Reduce Task\s*([\s\S]*)$/m);
  if (!match) {
    throw new Error("Task file must contain `## Map Task` followed by `## Reduce Task`.");
  }
  const mapTask = match[1].trim();
  const reduceTask = match[2].trim();
  if (!mapTask || !reduceTask) {
    throw new Error("Both map and reduce task sections must be non-empty.");
  }
  return { mapTask, reduceTask };
}

export async function waitForHttp(url, timeoutMs = 30000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Ignore transient startup failures.
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 500));
  }
  throw new Error(`Timed out waiting for ${url}`);
}
