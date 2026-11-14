import { spawn, type ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import { join } from "node:path";
import type { AppRecord, Runtime } from "./contracts";
export function localEnv() {
  return Object.fromEntries(
    Object.entries(process.env).filter(([k]) =>
      [
        "HOME",
        "PATH",
        "TMPDIR",
        "TEMP",
        "LANG",
        "USER",
        "SHELL",
        "SYSTEMROOT",
      ].includes(k),
    ),
  ) as NodeJS.ProcessEnv;
}
export class Supervisor {
  states: Record<string, Runtime> = {};
  children = new Map<string, ChildProcess>();
  pending = new Map<string, Promise<void>>();
  constructor(
    readonly executable: string,
    readonly base: string,
    readonly changed: () => void,
  ) {}
  start(app: AppRecord): Promise<void> {
    const p = this.pending.get(app.id);
    if (p) return p;
    if (this.states[app.id]?.status === "ready") return Promise.resolve();
    const task = (async () => {
      if (this.children.has(app.id)) await this.stop(app.id);
      await this.launch(app);
    })().finally(() => this.pending.delete(app.id));
    this.pending.set(app.id, task);
    return task;
  }
  private async launch(app: AppRecord) {
    const state: Runtime = { status: "starting", logs: "" };
    this.states[app.id] = state;
    this.changed();
    try {
      const port = await new Promise<number>((resolve, reject) => {
        const s = createServer();
        s.on("error", reject);
        s.listen(app.port ?? 0, "127.0.0.1", () => {
          const p = (s.address() as { port: number }).port;
          s.close(() => resolve(p));
        });
      }).catch(
        () =>
          new Promise<number>((resolve, reject) => {
            const s = createServer();
            s.on("error", reject);
            s.listen(0, "127.0.0.1", () => {
              const p = (s.address() as { port: number }).port;
              s.close(() => resolve(p));
            });
          }),
      );
      app.port = port;
      const child = spawn(
        this.executable,
        [
          join(this.base, "dist/watchdog.cjs"),
          app.path,
          this.executable,
          join(this.base, "node_modules/vite/bin/vite.js"),
          "--host",
          "127.0.0.1",
          "--port",
          String(port),
          "--strictPort",
        ],
        {
          env: {
            ...localEnv(),
            ELECTRON_RUN_AS_NODE: "1",
            WRANGLER_SEND_METRICS: "false",
          },
          stdio: ["pipe", "pipe", "pipe"],
        },
      );
      this.children.set(app.id, child);
      const log = (data: Buffer) => {
        state.logs = (
          state.logs + data.toString().replace(/\x1b\[[0-9;]*m/g, "")
        ).slice(-24000);
        this.changed();
      };
      child.stdout!.on("data", log);
      child.stderr!.on("data", log);
      child.on("error", (e) => {
        state.status = "failed";
        state.error = e.message;
        this.changed();
      });
      child.on("exit", () => {
        this.children.delete(app.id);
        if (state.status !== "stopping" && state.status !== "stopped") {
          state.status = "failed";
          state.error = "Local server exited. View logs and retry.";
        }
        this.changed();
      });
      for (let i = 0; i < 240; i++) {
        if (state.status !== "starting")
          throw new Error(state.error ?? "Startup cancelled");
        try {
          const r = await fetch(`http://127.0.0.1:${port}/api/health`, {
            signal: AbortSignal.timeout(700),
          });
          if (r.ok) {
            state.status = "ready";
            state.url = `http://127.0.0.1:${port}`;
            this.changed();
            return;
          }
        } catch {}
        await new Promise((r) => setTimeout(r, 250));
      }
      throw new Error("Server did not become ready within 60 seconds.");
    } catch (e) {
      if (state.status !== "stopping" && state.status !== "stopped") {
        state.status = "failed";
        state.error = String(e);
      }
      const child = this.children.get(app.id);
      child?.stdin?.end();
      this.changed();
    }
  }
  async stop(id: string) {
    const state = this.states[id];
    if (state) state.status = "stopping";
    this.changed();
    const child = this.children.get(id);
    if (child) {
      await new Promise<void>((resolve) => {
        child.once("exit", () => resolve());
        child.stdin?.end();
        setTimeout(resolve, 3000).unref();
      });
    }
    if (state) state.status = "stopped";
    this.changed();
  }
  async stopAll() {
    await Promise.all(
      [...new Set([...this.children.keys(), ...this.pending.keys()])].map(
        (id) => this.stop(id),
      ),
    );
    await Promise.all(this.pending.values());
  }
}
