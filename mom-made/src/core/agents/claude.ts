import { spawn, execFile, type ChildProcess } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
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

const CANDIDATES = [
  join(homedir(), ".local/bin/claude"),
  "/opt/homebrew/bin/claude",
  "/usr/local/bin/claude",
  join(homedir(), ".claude/local/claude"),
];
export function findClaude(): string | undefined {
  if (process.env.MOM_MADE_CLAUDE) return process.env.MOM_MADE_CLAUDE;
  return CANDIDATES.find(existsSync);
}
const SAFE_BASH = [
  "ls",
  "cat",
  "rg",
  "grep",
  "find",
  "head",
  "tail",
  "wc",
  "pwd",
  "mkdir",
  "mv",
  "cp",
  "touch",
  "npm install",
  "npm i",
  "npm run build",
  "npm uninstall",
  "git status",
  "git diff",
  "git log",
  "sed -n",
];
const MODELS = [
  { model: "sonnet", displayName: "Claude Sonnet", isDefault: true },
  { model: "opus", displayName: "Claude Opus", isDefault: false },
  { model: "haiku", displayName: "Claude Haiku", isDefault: false },
].map((m) => ({
  ...m,
  defaultReasoningEffort: "",
  supportedReasoningEfforts: [],
}));

/** Claude Code through the official, unmodified CLI and its own sign-in. */
export class ClaudeAgent implements Agent {
  readonly id = "claude" as const;
  readonly label = "Claude Code";
  readonly approvals = false;
  readonly help = {
    install: [
      "curl -fsSL https://claude.ai/install.sh | bash",
      "npm install -g @anthropic-ai/claude-code",
    ],
    loginHint: "Sign in with your Claude account.",
    note: "Claude Code uses its own permission settings and does not show approval prompts in mom-made.",
  };
  private child?: ChildProcess;
  private interrupted = false;
  async probe(): Promise<Probe> {
    const bin = findClaude();
    if (!bin) return { installed: false, ready: false, models: [] };
    const out = await new Promise<string>((resolve) =>
      execFile(
        bin,
        ["auth", "status"],
        { env: localEnv(), timeout: 10000 },
        (_e, so) => resolve(String(so ?? "")),
      ),
    );
    try {
      const s = JSON.parse(out.slice(out.indexOf("{")));
      return {
        installed: true,
        ready: !!s.loggedIn,
        account: s.loggedIn
          ? (s.email ?? s.authMethod ?? "Signed in")
          : undefined,
        models: MODELS,
      };
    } catch {
      return { installed: true, ready: false, models: MODELS };
    }
  }
  async run(t: TurnInput, emit: (e: AgentEvent) => void) {
    const bin = findClaude();
    if (!bin) {
      emit({
        type: "done",
        status: "failed",
        error: "Claude Code is not installed.",
      });
      return;
    }
    this.interrupted = false;
    const mcp = JSON.stringify({
      mcpServers: t.mcp ? { "mom-made": t.mcp } : {},
    });
    const args = [
      "-p",
      "--input-format",
      "stream-json",
      "--output-format",
      "stream-json",
      "--verbose",
      "--include-partial-messages",
      "--mcp-config",
      mcp,
      "--strict-mcp-config",
      "--append-system-prompt",
      t.instructions,
    ];
    if (t.mcp) {
      args.push(
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        [
          "Read",
          "Edit",
          "Write",
          "MultiEdit",
          "Glob",
          "Grep",
          "mcp__mom-made",
          ...SAFE_BASH.map((b) => `Bash(${b}:*)`),
        ].join(","),
      );
    } else {
      args.push("--tools", "");
    }
    if (t.model) args.push("--model", t.model);
    if (t.session) args.push("--resume", t.session);
    const content: unknown[] = [{ type: "text", text: t.text }];
    for (const img of t.images)
      content.push({
        type: "image",
        source: {
          type: "base64",
          media_type: "image/png",
          data: readFileSync(img).toString("base64"),
        },
      });
    const child = spawn(bin, args, {
      cwd: t.cwd,
      env: { ...localEnv(), CLAUDE_CODE_DISABLE_TELEMETRY: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    let err = "";
    let outcome: AgentEvent | undefined;
    let sawText = false;
    child.stderr!.on("data", (d) => (err = tail(err + d, 1200)));
    const finish = (e: AgentEvent) => {
      outcome ??= e;
    };
    jsonLines(child, (r) => {
      if (r.type === "system" && r.subtype === "init" && r.session_id)
        emit({ type: "session", id: r.session_id });
      else if (r.type === "stream_event") {
        const ev = r.event ?? {};
        if (ev.type === "message_start") emit({ type: "textBreak" });
        if (
          ev.type === "content_block_delta" &&
          ev.delta?.type === "text_delta"
        ) {
          sawText = true;
          emit({ type: "text", delta: ev.delta.text });
        }
      } else if (r.type === "assistant") {
        for (const b of r.message?.content ?? []) {
          if (b.type !== "tool_use") continue;
          const name = String(b.name);
          const input = b.input ?? {};
          const mcpEv = toolActivity(name);
          if (mcpEv) emit(mcpEv);
          else if (/^(Edit|Write|MultiEdit|NotebookEdit)$/.test(name))
            emit({
              type: "activity",
              kind: "edit",
              text: "Edited 1 file",
              detail: relPath(String(input.file_path ?? ""), t.cwd),
            });
          else if (name === "Bash") {
            const c = classify(String(input.command ?? ""));
            emit({
              type: "activity",
              kind: c.kind,
              text: c.text,
              detail: String(input.command ?? ""),
            });
          } else if (/^(Read|Glob|Grep|LS)$/.test(name))
            emit({
              type: "activity",
              kind: "look",
              text: "Looked through the files",
            });
        }
      } else if (r.type === "result") {
        if (r.is_error)
          finish({
            type: "done",
            status: "failed",
            error: String(r.result ?? "Claude Code stopped."),
          });
        else {
          if (!sawText && r.result)
            emit({ type: "text", delta: String(r.result) });
          finish({ type: "done", status: "completed" });
        }
      }
    });
    await new Promise<void>((resolve) => {
      child.on("error", (e) => {
        finish({ type: "done", status: "failed", error: e.message });
      });
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
                error:
                  `Claude Code stopped unexpectedly (${code}). ${err}`.trim(),
              },
        );
        resolve();
      });
      child.stdin!.end(
        JSON.stringify({ type: "user", message: { role: "user", content } }) +
          "\n",
      );
    });
    this.child = undefined;
    if (outcome) emit(outcome);
  }
  async interrupt() {
    const child = this.child;
    this.interrupted = true;
    child?.kill("SIGINT");
    setTimeout(() => child?.kill("SIGKILL"), 4000).unref();
  }
  answer(_id: string | number, _d: Decision) {
    throw Error("Claude Code does not ask for approval.");
  }
  async login() {
    const bin = findClaude();
    if (!bin) throw Error("Claude Code is not installed.");
    await new Promise<void>((resolve, reject) => {
      const child = execFile(
        bin,
        ["auth", "login"],
        { env: localEnv(), timeout: 5 * 60_000 },
        (error) => {
          if (error)
            reject(
              Error(
                "Claude Code sign-in did not finish. Run claude auth login in Terminal, then refresh connections.",
              ),
            );
          else resolve();
        },
      );
      child.stdin?.end();
    });
  }
  close() {
    this.child?.kill("SIGTERM");
  }
}
