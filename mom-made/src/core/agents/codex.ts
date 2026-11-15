import { Codex, findCodex } from "../codex";
import type { Model } from "../contracts";
import {
  classify,
  type Agent,
  type AgentEvent,
  type Decision,
  type Probe,
  type TurnInput,
} from "./types";
import { toolActivity, relPath } from "./types";

export class CodexAgent implements Agent {
  readonly id = "codex" as const;
  readonly label = "Codex (ChatGPT)";
  readonly approvals = true;
  readonly help = {
    install: ["brew install codex", "npm install -g @openai/codex"],
    loginHint: "Sign in with your ChatGPT account.",
  };
  private codex: Codex;
  private connected = false;
  private emit?: (e: AgentEvent) => void;
  private current?: {
    thread: string;
    turn?: string;
    finish: (e: AgentEvent) => void;
    cwd: string;
  };
  private approvalId?: string | number;
  private onAccountChange?: () => void;
  constructor(
    private openUrl: (u: string) => Promise<void>,
    codex = new Codex(),
  ) {
    this.codex = codex;
    codex.on("failure", (e: Error) => {
      this.connected = false;
      this.current?.finish({
        type: "done",
        status: "failed",
        error: e.message + codex.stderrTail(),
      });
    });
    codex.on("request", (m: any) => {
      if (!this.current) return void codex.reject(m.id);
      if (
        m.method === "item/commandExecution/requestApproval" ||
        m.method === "item/fileChange/requestApproval"
      ) {
        this.approvalId = m.id;
        this.emit?.({
          type: "approval",
          id: m.id,
          what: m.method.includes("fileChange") ? "files" : "command",
          reason: m.params?.reason,
          command: m.params?.command,
        });
      } else if (m.method === "mcpServer/elicitation/request") {
        codex.answer(m.id, {
          action: m.params?.serverName === "mom-made" ? "accept" : "decline",
          content: null,
        });
      } else {
        codex.reject(m.id);
        this.emit?.({
          type: "activity",
          kind: "note",
          text: "The agent asked for something mom-made can't do yet.",
          detail: m.method,
        });
      }
    });
    codex.on("notification", (m: any) => this.notify(m));
  }
  onAccount(cb: () => void) {
    this.onAccountChange = cb;
  }
  private notify(m: any) {
    if (m.method === "account/login/completed") return this.onAccountChange?.();
    const cur = this.current;
    if (!cur) return;
    const p = m.params ?? {};
    if (p.threadId && p.threadId !== cur.thread) return;
    const emit = this.emit!;
    if (m.method === "turn/started") cur.turn = p.turn.id;
    if (m.method === "item/agentMessage/delta")
      emit({ type: "text", delta: p.delta });
    if (m.method === "item/started") {
      const it = p.item ?? {};
      if (it.type === "agentMessage") emit({ type: "textBreak" });
      else if (it.type === "commandExecution") {
        const c = classify(String(it.command ?? ""));
        emit({
          type: "activity",
          kind: c.kind,
          text: c.text,
          detail: String(it.command ?? ""),
        });
      } else if (it.type === "mcpToolCall" && it.server === "mom-made") {
        const a = toolActivity(it.tool);
        emit(a ?? { type: "activity", kind: "look", text: "Used a tool" });
      } else if (it.type === "webSearch")
        emit({ type: "activity", kind: "look", text: "Searched the web" });
    }
    if (m.method === "item/completed" && p.item?.type === "fileChange") {
      const changes = (p.item.changes ?? []) as { path: string }[];
      if (changes.length)
        emit({
          type: "activity",
          kind: "edit",
          text:
            changes.length === 1
              ? "Edited 1 file"
              : `Edited ${changes.length} files`,
          detail: changes.map((c) => relPath(c.path, cur.cwd)).join("\n"),
        });
    }
    if (m.method === "error" && !p.willRetry)
      emit({
        type: "activity",
        kind: "note",
        text: p.error?.message ?? "The agent ran into a problem.",
      });
    if (m.method === "turn/completed") {
      const s = String(p.turn.status);
      cur.finish({
        type: "done",
        status:
          s === "completed"
            ? "completed"
            : s === "interrupted"
              ? "interrupted"
              : "failed",
        error: p.turn.error?.message,
      });
    }
  }
  private async connect() {
    await this.codex.open();
    this.connected = true;
  }
  async probe(): Promise<Probe> {
    if (!findCodex()) return { installed: false, ready: false, models: [] };
    try {
      await this.connect();
      const [a, m] = await Promise.all([
        this.codex.request("account/read", {}),
        this.codex.request("model/list", {}),
      ]);
      return {
        installed: true,
        ready: !!a.account,
        account: a.account
          ? `${a.account.type}${a.account.email ? " · " + a.account.email : ""}`
          : undefined,
        models: m.data.filter((x: any) => !x.hidden) as Model[],
      };
    } catch (e) {
      return { installed: true, ready: false, detail: String(e), models: [] };
    }
  }
  async run(t: TurnInput, emit: (e: AgentEvent) => void) {
    this.emit = emit;
    if (!this.connected) await this.connect();
    const params = {
      cwd: t.cwd,
      approvalPolicy: t.mcp ? "on-request" : "never",
      sandbox: t.mcp ? "workspace-write" : "read-only",
      model: t.model || undefined,
      developerInstructions: t.instructions,
      config: t.mcp ? { mcp_servers: { "mom-made": t.mcp } } : undefined,
    };
    let thread = t.session;
    if (thread)
      await this.codex.request("thread/resume", {
        ...params,
        threadId: thread,
      });
    else
      thread = (await this.codex.request("thread/start", params)).thread
        .id as string;
    emit({ type: "session", id: thread });
    const done = new Promise<void>((resolve) => {
      this.current = {
        thread: thread!,
        cwd: t.cwd,
        finish: (e) => {
          if (!this.current) return;
          this.current = undefined;
          this.approvalId = undefined;
          emit(e);
          resolve();
        },
      };
    });
    try {
      const input: unknown[] = [{ type: "text", text: t.text }];
      for (const img of t.images) input.push({ type: "localImage", path: img });
      const r = await this.codex.request("turn/start", {
        threadId: thread,
        input,
        cwd: t.cwd,
        model: t.model || undefined,
        effort: t.effort || undefined,
      });
      if (this.current) this.current.turn = r.turn.id;
    } catch (e) {
      this.current?.finish({
        type: "done",
        status: "failed",
        error: e instanceof Error ? e.message : String(e),
      });
    }
    await done;
  }
  async interrupt() {
    const c = this.current;
    if (!c) return;
    if (c.turn)
      await this.codex.request("turn/interrupt", {
        threadId: c.thread,
        turnId: c.turn,
      });
    else {
      this.codex.close();
      c.finish({ type: "done", status: "interrupted" });
    }
  }
  answer(id: string | number, d: Decision) {
    if (this.approvalId !== id) throw Error("Approval is no longer pending");
    this.approvalId = undefined;
    this.codex.answer(id, {
      decision:
        d === "decline"
          ? "decline"
          : d === "always"
            ? "acceptForSession"
            : "accept",
    });
  }
  async login() {
    await this.codex.open();
    const r = await this.codex.request("account/login/start", {
      type: "chatgpt",
    });
    if (
      typeof r.authUrl === "string" &&
      new URL(r.authUrl).protocol === "https:"
    )
      await this.openUrl(r.authUrl);
  }
  async compact(cwd: string, session: string) {
    await this.codex.open();
    await this.codex.request("thread/resume", { threadId: session, cwd });
    await this.codex.request("thread/compact/start", { threadId: session });
  }
  close() {
    this.connected = false;
    this.codex.close();
  }
}
