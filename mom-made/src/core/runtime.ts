import { spawn, type ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import type { AppRecord, Runtime } from "./contracts";
let shimDir: string | undefined;
/** Gives assistants a `node` command (the app's bundled runtime) when the Mac has none. */
export function installNodeShim(dir: string, runtime: string) {
  mkdirSync(dir, { recursive: true });
  const file = join(dir, "node");
  writeFileSync(
    file,
    `#!/bin/sh\nELECTRON_RUN_AS_NODE=1 exec ${JSON.stringify(runtime)} "$@"\n`,
  );
  chmodSync(file, 0o755);
  shimDir = dir;
}
export function localEnv() {
  const env = Object.fromEntries(
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
  // The shim goes last so a real Node install always wins.
  const extra = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    ...(shimDir ? [shimDir] : []),
  ];
  const have = (env.PATH ?? "").split(":");
  env.PATH = [...have, ...extra.filter((p) => !have.includes(p))]
    .filter(Boolean)
    .join(":");
  return env;
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
      const state = this.states[app.id];
      // Parallel starts can choose the same inspector port before either binds it.
      if (state.status === "failed" && state.logs.includes("EADDRINUSE"))
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
      await this.migrate(app, state);
      if (state.status !== "starting") return;
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
  /** Applies the app's checked-in D1 migrations to its local database. */
  private async migrate(app: AppRecord, state: Runtime) {
    const dir = join(app.path, "migrations");
    if (!existsSync(dir) || !readdirSync(dir).some((f) => f.endsWith(".sql")))
      return;
    const out = await new Promise<{ code: number | null; text: string }>(
      (resolve) => {
        const c = spawn(
          this.executable,
          [
            "-r",
            join(this.base, "dist/wrangler.cjs"),
            join(this.base, "node_modules/wrangler/wrangler-dist/cli.js"),
            "d1",
            "migrations",
            "apply",
            "DB",
            "--local",
          ],
          {
            cwd: app.path,
            env: {
              ...localEnv(),
              ELECTRON_RUN_AS_NODE: "1",
              WRANGLER_SEND_METRICS: "false",
              CI: "1",
            },
            stdio: ["ignore", "pipe", "pipe"],
          },
        );
        let text = "";
        const grab = (d: Buffer) => (text = (text + d.toString()).slice(-8000));
        c.stdout.on("data", grab);
        c.stderr.on("data", grab);
        c.on("error", (e) => resolve({ code: 1, text: e.message }));
        c.on("exit", (code) => resolve({ code, text }));
        setTimeout(() => c.kill(), 60000).unref();
      },
    );
    state.logs += out.text.replace(/\x1b\[[0-9;]*m/g, "");
    if (out.code !== 0)
      throw new Error(
        "Your app's database could not be updated.\n" + out.text.slice(-600),
      );
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
