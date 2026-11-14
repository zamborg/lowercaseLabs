import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { localEnv } from "../runtime";
import { jsonLines, tail } from "./lines";
import {
  classify,
  toolActivity,
  relPath,
  type Agent,
  type AgentEvent,
  type Decision,
  type Probe,
  type TurnInput,
} from "./types";
import type { Model, PiLocalSettings } from "../contracts";
import {
  checkLocalPi,
  configureLocalPi,
  LOCAL_PI_PROVIDER,
  localPiDirectory,
  localPiSession,
} from "./pi-local";

export function findPi(): string | undefined {
  if (process.env.MOM_MADE_PI) return process.env.MOM_MADE_PI;
  for (const path of (process.env.PATH ?? "").split(":").filter(Boolean)) {
    const candidate = join(path, "pi");
    if (path.startsWith("/") && existsSync(candidate)) return candidate;
  }
  const fixed = [
    "/opt/homebrew/bin/pi",
    "/usr/local/bin/pi",
    join(homedir(), ".local/bin/pi"),
    join(homedir(), ".npm-global/bin/pi"),
    join(homedir(), ".bun/bin/pi"),
  ].find(existsSync);
  if (fixed) return fixed;
  const nvm = join(homedir(), ".nvm/versions/node");
  for (const v of existsSync(nvm) ? readdirSync(nvm).sort().reverse() : []) {
    const p = join(nvm, v, "bin/pi");
    if (existsSync(p)) return p;
  }
}
const LEVELS = ["off", "low", "medium", "high"];
const DEFAULT_MODEL = "anthropic/claude-sonnet-4.5";
function piEnv(bin: string) {
  const env = localEnv();
  return { ...env, PATH: [dirname(bin), env.PATH].filter(Boolean).join(":") };
}

type PiConnection =
  | { getKey: () => string | undefined }
  | { local: () => PiLocalSettings; root: () => string };

/** Pi (pi.dev) over JSONL RPC. Cloud and local inference keep separate sessions. */
export class PiAgent implements Agent {
  readonly id: "pi" | "pi-local";
  readonly label: string;
  readonly approvals = false;
  readonly help: Agent["help"];
  private child?: ChildProcess;
  private interrupted = false;
  private send?: (c: Record<string, unknown>) => void;
  constructor(
    private connection: PiConnection,
    private request: typeof fetch = fetch,
  ) {
    const local = "local" in connection;
    this.id = local ? "pi-local" : "pi";
    this.label = local ? "Pi + Local model" : "Pi + OpenRouter";
    this.help = {
      install: [
        "npm install -g --ignore-scripts @earendil-works/pi-coding-agent",
      ],
      needsKey: !local,
      loginHint: local
        ? "Connect to a model server running on this Mac."
        : "Paste an OpenRouter API key (openrouter.ai/keys).",
      note: local
        ? "Local Pi uses text chat and page tools. For visual markup, switch to an assistant with image support."
        : "Pi works on its own and won't stop to ask you. You pay OpenRouter per use.",
    };
  }
  private localEnvironment(settings: PiLocalSettings, root: string) {
    const directory = localPiDirectory(root, settings);
    configureLocalPi(directory, settings);
    return {
      PI_CODING_AGENT_DIR: directory,
      PI_CODING_AGENT_SESSION_DIR: join(directory, "sessions"),
      PI_OFFLINE: "1",
      PI_TELEMETRY: "0",
    };
  }
  async probe(): Promise<Probe> {
    const bin = findPi();
    if (!bin) return { installed: false, ready: false, models: [] };
    const unavailable = (detail: string): Probe => ({
      installed: true,
      ready: false,
      detail,
      models: [],
    });
    if ("local" in this.connection) {
      const settings = this.connection.local();
      const problem = await checkLocalPi(settings, this.request);
      if (problem) return unavailable(problem);
      const env = this.localEnvironment(settings, this.connection.root());
      const models = await this.catalog(bin, env, LOCAL_PI_PROVIDER);
      return models.some((model) => model.model === settings.model)
        ? {
            installed: true,
            ready: true,
            account: "Local model connected",
            models,
          }
        : unavailable(
            "Pi couldn't load the local model. Update Pi and refresh.",
          );
    }
    const key = this.connection.getKey();
    if (!key) return unavailable("Add your OpenRouter key.");
    try {
      const result = await this.request("https://openrouter.ai/api/v1/key", {
        headers: { Authorization: `Bearer ${key}` },
        signal: AbortSignal.timeout(8000),
        redirect: "error",
      });
      if (result.status === 401 || result.status === 403)
        return unavailable(
          "OpenRouter rejected this key. Replace it and try again.",
        );
      if (!result.ok)
        return unavailable(
          "OpenRouter could not verify this key. Try refreshing the connection.",
        );
    } catch {
      return unavailable(
        "Couldn't reach OpenRouter. Check your connection and refresh.",
      );
    }
    const models = await this.catalog(
      bin,
      { OPENROUTER_API_KEY: key },
      "openrouter",
    );
    return models.length
      ? {
          installed: true,
          ready: true,
          account: "OpenRouter key verified",
          models,
        }
      : unavailable(
          "Pi couldn't load OpenRouter models. Check your Pi installation and refresh.",
        );
  }
  private catalog(bin: string, env: Record<string, string>, provider: string) {
    return new Promise<Model[]>((resolve) => {
      const c = spawn(
        bin,
        ["--mode", "rpc", "--no-session", "--no-extensions", "--no-approve"],
        {
          env: { ...piEnv(bin), ...env },
          stdio: ["pipe", "pipe", "ignore"],
        },
      );
      const finish = (models: Model[]) => {
        clearTimeout(timer);
        resolve(models);
      };
      const timer = setTimeout(() => {
        c.kill();
        finish([]);
      }, 12000);
      jsonLines(c, (r) => {
        if (
          r.type === "response" &&
          r.id === "m" &&
          r.command === "get_available_models"
        ) {
          const list = ((r.data?.models ?? []) as any[])
            .filter((m) => m.provider === provider)
            .map((m) => ({
              model: String(m.id),
              displayName: String(m.name ?? m.id),
              isDefault:
                provider === LOCAL_PI_PROVIDER || m.id === DEFAULT_MODEL,
              defaultReasoningEffort:
                provider === "openrouter" && m.reasoning ? "medium" : "",
              supportedReasoningEfforts:
                provider === "openrouter" && m.reasoning
                  ? LEVELS.map((l) => ({ reasoningEffort: l, description: l }))
                  : [],
            }));
          c.stdin!.end();
          finish(list);
        }
      });
      c.on("error", () => finish([]));
      c.on("close", () => finish([]));
      c.stdin!.on("error", () => finish([]));
      c.stdin!.write(
        JSON.stringify({ id: "m", type: "get_available_models" }) + "\n",
      );
    });
  }
  async run(t: TurnInput, emit: (e: AgentEvent) => void) {
    const bin = findPi();
    if (!bin)
      return void emit({
        type: "done",
        status: "failed",
        error: "Pi is not installed.",
      });
    const local =
      "local" in this.connection ? this.connection.local() : undefined;
    const key =
      "getKey" in this.connection ? this.connection.getKey() : undefined;
    if (local && t.images.length)
      return void emit({
        type: "done",
        status: "failed",
        error:
          "Local Pi accepts text and page tools. Switch assistants to send visual markup.",
      });
    if (!local && !key)
      return void emit({
        type: "done",
        status: "failed",
        error: "Add your OpenRouter key first.",
      });
    const images = t.images.map((p) => ({
      type: "image",
      mimeType: "image/png",
      data: readFileSync(p).toString("base64"),
    }));
    this.interrupted = false;
    const provider = local ? LOCAL_PI_PROVIDER : "openrouter";
    const args = [
      "--mode",
      "rpc",
      "--provider",
      provider,
      "--no-approve",
      "--append-system-prompt",
      t.instructions,
    ];
    if (t.mcp) args.push("-e", join(t.mcp.args[0], "..", "pi-mcp.mjs"));
    else args.push("--no-tools", "--no-mcp");
    args.push("--model", local?.model ?? (t.model || DEFAULT_MODEL));
    let env = key ? { OPENROUTER_API_KEY: key } : {};
    let session = t.session;
    if (local && "local" in this.connection) {
      const localEnv = this.localEnvironment(local, this.connection.root());
      env = { ...env, ...localEnv };
      session = localPiSession(localEnv.PI_CODING_AGENT_DIR, session);
      if (t.mcp) args.push("-e", "builtin:mcp");
    } else if (t.effort && LEVELS.includes(t.effort))
      args.push("--thinking", t.effort);
    if (local || !t.mcp) args.push("--no-extensions");
    if (session) args.push("--session", session);
    const child = spawn(bin, args, {
      cwd: t.cwd,
      env: {
        ...piEnv(bin),
        ...env,
        ...(local ? { MOM_PI_TEXT_ONLY: "1" } : {}),
        ...(t.mcp ? {
          MOM_MCP_COMMAND: t.mcp.command,
          MOM_MCP_ARGS: JSON.stringify(t.mcp.args),
          MOM_MCP_ENV: JSON.stringify(t.mcp.env),
        } : {}),
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    let err = "";
    let outcome: AgentEvent | undefined;
    child.stderr!.on("data", (d) => (err = tail(err + d, 1200)));
    const finish = (e: AgentEvent) => {
      outcome ??=
        key && e.type === "done" && e.error
          ? { ...e, error: e.error.replaceAll(key, "[redacted]") }
          : e;
    };
    const write = (c: Record<string, unknown>) =>
      child.stdin!.write(JSON.stringify(c) + "\n");
    this.send = write;
    let lastError: string | undefined;
    jsonLines(child, (r) => {
      if (r.type === "response") {
        if (r.command === "prompt" && r.id === "p") {
          if (r.success === false) {
            finish({
              type: "done",
              status: "failed",
              error: String(r.error ?? "Pi would not start."),
            });
            child.stdin!.end();
          } else if (r.data?.disposition === "handled") {
            finish({ type: "done", status: "completed" });
            child.stdin!.end();
          }
        }
        if (r.command === "get_state" && r.id === "state") {
          if (
            local &&
            (r.data?.model?.provider !== provider ||
              r.data?.model?.id !== local.model)
          ) {
            finish({
              type: "done",
              status: "failed",
              error:
                "Pi did not select the local model. Refresh its connection in Settings.",
            });
            child.stdin!.end();
            return;
          }
          if (this.interrupted) {
            finish({ type: "done", status: "interrupted" });
            child.stdin!.end();
            return;
          }
          if (r.data?.sessionFile)
            emit({ type: "session", id: String(r.data.sessionFile) });
          write({
            id: "p",
            type: "prompt",
            message: t.text,
            ...(images.length ? { images } : {}),
          });
        }
        return;
      }
      if (r.type === "message_start" && r.message?.role === "assistant") {
        lastError = undefined;
        emit({ type: "textBreak" });
      }
      if (r.type === "message_update") {
        const u = r.assistantMessageEvent ?? {};
        if (u.type === "text_delta") emit({ type: "text", delta: u.delta });
        if (u.type === "error")
          lastError = String(u.error?.errorMessage ?? u.reason ?? "");
      }
      if (r.type === "message_end" && r.message?.role === "assistant") {
        if (r.message.errorMessage) lastError = String(r.message.errorMessage);
        else if (
          local &&
          r.message.stopReason === "length" &&
          !r.message.content?.some((c: { type: string }) => c.type === "text")
        )
          lastError =
            "The local model ran out of response tokens. Start a fresh chat or increase its context.";
      }
      if (r.type === "tool_execution_start") {
        const name = String(r.toolName);
        const a = r.args ?? {};
        const m = toolActivity(name);
        if (m) emit(m);
        else if (/^(edit|write)$/i.test(name))
          emit({
            type: "activity",
            kind: "edit",
            text: "Edited 1 file",
            detail: relPath(String(a.path ?? a.file_path ?? ""), t.cwd),
          });
        else if (name === "bash") {
          const c = classify(String(a.command ?? ""));
          emit({
            type: "activity",
            kind: c.kind,
            text: c.text,
            detail: String(a.command ?? ""),
          });
        } else if (/^(read|grep|find|ls)$/i.test(name))
          emit({
            type: "activity",
            kind: "look",
            text: "Looked through the files",
          });
      }
      if (r.type === "agent_settled") {
        finish(
          r.aborted || this.interrupted
            ? { type: "done", status: "interrupted" }
            : lastError
              ? { type: "done", status: "failed", error: lastError }
              : { type: "done", status: "completed" },
        );
        child.stdin!.end();
      }
    });
    await new Promise<void>((resolve) => {
      child.on("error", (e) =>
        finish({ type: "done", status: "failed", error: e.message }),
      );
      child.stdin!.on("error", (e) =>
        finish({ type: "done", status: "failed", error: e.message }),
      );
      child.on("close", (code) => {
        finish(
          this.interrupted
            ? { type: "done", status: "interrupted" }
            : {
                type: "done",
                status: "failed",
                error: `Pi stopped unexpectedly (${code}). ${err}`.trim(),
              },
        );
        resolve();
      });
      write({ id: "state", type: "get_state" });
    });
    this.child = undefined;
    this.send = undefined;
    if (outcome) emit(outcome);
  }
  async interrupt() {
    const child = this.child;
    this.interrupted = true;
    if (child?.stdin?.writable) {
      this.send?.({ type: "abort" });
      child.stdin.end();
    }
    setTimeout(() => child?.kill("SIGTERM"), 4000).unref();
  }
  answer(_id: string | number, _d: Decision) {
    throw Error("Pi does not ask for approval.");
  }
  close() {
    this.child?.kill("SIGTERM");
  }
}
