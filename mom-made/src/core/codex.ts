import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { localEnv } from "./runtime";
export class Codex extends EventEmitter {
  child?: ChildProcessWithoutNullStreams;
  private seq = 0;
  private pending = new Map<
    number,
    {
      resolve: (v: any) => void;
      reject: (e: Error) => void;
      timer: NodeJS.Timeout;
    }
  >();
  private buffer = "";
  private opening?: Promise<void>;
  constructor(
    readonly binary = process.env.MOM_MADE_CODEX ||
      [
        "/opt/homebrew/bin/codex",
        "/usr/local/bin/codex",
        join(homedir(), ".local/bin/codex"),
      ].find(existsSync) ||
      "codex",
  ) {
    super();
  }
  async open() {
    if (this.opening) return this.opening;
    this.opening = this.launch().catch((e) => {
      this.opening = undefined;
      throw e;
    });
    return this.opening;
  }
  private async launch() {
    const child = spawn(this.binary, ["app-server"], {
      stdio: "pipe",
      env: localEnv(),
    });
    this.child = child;
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (s: string) => {
      if (this.child !== child) return;
      this.buffer += s;
      if (this.buffer.length > 8_000_000) {
        this.close();
        this.emit(
          "failure",
          new Error("Codex message exceeds transport limit"),
        );
        return;
      }
      let n;
      while ((n = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, n);
        this.buffer = this.buffer.slice(n + 1);
        if (!line.trim()) continue;
        try {
          this.receive(JSON.parse(line));
        } catch (e) {
          this.emit(
            "failure",
            new Error("Invalid Codex protocol record: " + String(e)),
          );
        }
      }
    });
    child.stderr.on("data", () => {});
    const fail = (e: Error) => {
      if (this.child !== child) return;
      for (const p of this.pending.values()) {
        clearTimeout(p.timer);
        p.reject(e);
      }
      this.pending.clear();
      this.child = undefined;
      this.opening = undefined;
      this.emit("failure", e);
    };
    child.on("error", fail);
    child.on("exit", () => fail(new Error("Codex disconnected")));
    await this.request("initialize", {
      clientInfo: { name: "mom_made", title: "mom-made", version: "0.1.0" },
    });
    this.write({ method: "initialized" });
  }
  private receive(m: any) {
    if (m.id !== undefined && !m.method) {
      const p = this.pending.get(m.id);
      if (p) {
        clearTimeout(p.timer);
        this.pending.delete(m.id);
        m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result);
      }
    } else if (m.method) {
      this.emit(m.id !== undefined ? "request" : "notification", m);
    }
  }
  private write(value: unknown) {
    if (!this.child) throw Error("Codex is not connected");
    this.child.stdin.write(JSON.stringify(value) + "\n");
  }
  request(method: string, params: unknown): Promise<any> {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(Error(method + " timed out"));
      }, 30000);
      this.pending.set(id, { resolve, reject, timer });
      try {
        this.write({ id, method, params });
      } catch (e) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(e);
      }
    });
  }
  answer(id: string | number, result: unknown) {
    this.write({ id, result });
  }
  reject(id: string | number) {
    this.write({
      id,
      error: {
        code: -32601,
        message: "This client does not support that request.",
      },
    });
  }
  close() {
    const child = this.child;
    this.child = undefined;
    this.buffer = "";
    this.opening = undefined;
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(new Error("Codex connection closed"));
    }
    this.pending.clear();
    child?.stdin.end();
    child?.kill();
  }
}
