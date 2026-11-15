import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import {
  cp,
  mkdir,
  readFile,
  readdir,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import type { AppRecord, Cloudflare } from "./contracts";
import { localEnv } from "./runtime";

type Run = { code: number | null; out: string };
export type PublishResult = { url?: string; hash: string; dryRun: boolean };

/** Publishes a snapshot of an app with the user's own Wrangler login. */
export class Publisher {
  constructor(
    readonly executable: string,
    readonly base: string,
    readonly root: string,
  ) {}
  private wrangler(args: string[], cwd: string, onData?: (s: string) => void) {
    return new Promise<Run>((resolve) => {
      const c = spawn(
        this.executable,
        [
          "-r",
          join(this.base, "dist/wrangler.cjs"),
          join(this.base, "node_modules/wrangler/wrangler-dist/cli.js"),
          ...args,
        ],
        {
          cwd,
          env: {
            ...localEnv(),
            ELECTRON_RUN_AS_NODE: "1",
            WRANGLER_SEND_METRICS: "false",
            ...(args[0] === "login" ? {} : { CI: "1" }),
            NO_COLOR: "1",
          },
          stdio: ["ignore", "pipe", "pipe"],
        },
      );
      let out = "";
      const grab = (d: Buffer) => {
        const s = d.toString().replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
        out = (out + s).slice(-20000);
        onData?.(s);
      };
      c.stdout.on("data", grab);
      c.stderr.on("data", grab);
      const timer = setTimeout(
        () => c.kill(),
        args[0] === "whoami" ? 15000 : 10 * 60_000,
      );
      c.on("error", (e) => {
        clearTimeout(timer);
        resolve({ code: 1, out: e.message });
      });
      c.on("close", (code) => {
        clearTimeout(timer);
        resolve({ code, out });
      });
    });
  }
  async whoami(): Promise<Cloudflare> {
    const r = await this.wrangler(["whoami", "--json"], this.root);
    try {
      const who = JSON.parse(r.out);
      if (who.loggedIn === false) return { state: "login-required" };
      if (r.code === 0 && who.loggedIn === true)
        return {
          state: "ready",
          account: who.email ?? who.accounts?.[0]?.name,
        };
    } catch {}
    return { state: "unknown" };
  }
  /** Opens the browser for Cloudflare's own login page; resolves when it ends. */
  async login(): Promise<Cloudflare> {
    const result = await this.wrangler(["login"], this.root);
    const status = await this.whoami();
    if (result.code !== 0 && status.state !== "ready")
      throw Error("Cloudflare sign-in did not finish. Try signing in again.");
    return status;
  }
  private async snapshot(app: AppRecord) {
    const dest = join(this.root, "staging", app.id, String(Date.now()));
    await mkdir(dest, { recursive: true });
    const skip = new Set(["node_modules", ".wrangler", "dist", ".mom-made"]);
    await cp(app.path, dest, {
      recursive: true,
      filter: (src) => !skip.has(src.slice(app.path.length + 1).split("/")[0]),
    });
    await symlink(
      join(this.base, "node_modules"),
      join(dest, "node_modules"),
      "dir",
    );
    const hash = createHash("sha256");
    const walk = async (d: string, rel = ""): Promise<void> => {
      for (const e of (await readdir(d, { withFileTypes: true })).sort((a, b) =>
        a.name.localeCompare(b.name),
      )) {
        if (rel === "" && skip.has(e.name)) continue;
        const p = join(d, e.name);
        if (e.isDirectory()) await walk(p, rel + e.name + "/");
        else if (e.isFile())
          hash.update(rel + e.name).update(await readFile(p));
      }
    };
    await walk(dest);
    return { dir: dest, hash: hash.digest("hex") };
  }
  private async prune(app: AppRecord) {
    const dir = join(this.root, "staging", app.id);
    const names = (await readdir(dir).catch(() => [])).sort();
    for (const n of names.slice(0, -3))
      await rm(join(dir, n), { recursive: true, force: true });
  }
  async publish(
    app: AppRecord,
    step: (s: string, log?: string) => void,
    opts: { dryRun?: boolean } = {},
  ): Promise<PublishResult> {
    const dry = !!opts.dryRun;
    step("Checking your Cloudflare account");
    if (!dry) {
      const who = await this.whoami();
      if (who.state !== "ready")
        throw new Error("Sign in to Cloudflare first, then publish again.");
    }
    step("Saving a snapshot of your app");
    const snap = await this.snapshot(app);
    const manifestPath = join(app.path, "mom-made.json");
    const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
    const cfg = join(snap.dir, "wrangler.jsonc");
    const wr = JSON.parse(await readFile(cfg, "utf8"));
    if (!dry && wr.d1_databases?.length) {
      step("Preparing your app's database");
      const name = wr.d1_databases[0].database_name as string;
      // Cloudflare's own list is the source of truth; a saved id is never trusted blindly.
      const find = async () => {
        const list = await this.wrangler(["d1", "list", "--json"], snap.dir);
        try {
          return (
            JSON.parse(list.out.slice(list.out.indexOf("["))) as {
              name: string;
              uuid: string;
            }[]
          ).find((d) => d.name === name)?.uuid;
        } catch {
          return undefined;
        }
      };
      let id = await find();
      if (!id) {
        const r = await this.wrangler(["d1", "create", name], snap.dir);
        id =
          r.out.match(/database_id["\s:=]+"?([0-9a-f-]{36})/i)?.[1] ??
          (await find());
        if (!id)
          throw new Error(
            "Cloudflare would not create the database.\n" + r.out.slice(-500),
          );
      }
      manifest.cloudflare = {
        ...manifest.cloudflare,
        d1: { ...manifest.cloudflare?.d1, [name]: id },
      };
      await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
      wr.d1_databases[0].database_id = id;
      await writeFile(cfg, JSON.stringify(wr, null, 2));
    }
    step("Building");
    const build = await new Promise<Run>((resolve) => {
      const c = spawn(
        this.executable,
        [join(this.base, "node_modules/vite/bin/vite.js"), "build"],
        {
          cwd: snap.dir,
          env: { ...localEnv(), ELECTRON_RUN_AS_NODE: "1", NO_COLOR: "1" },
          stdio: ["ignore", "pipe", "pipe"],
        },
      );
      let out = "";
      const g = (d: Buffer) => (out = (out + d.toString()).slice(-8000));
      c.stdout.on("data", g);
      c.stderr.on("data", g);
      c.on("exit", (code) => resolve({ code, out }));
      c.on("error", (e) => resolve({ code: 1, out: e.message }));
    });
    if (build.code !== 0)
      throw new Error("The app did not build.\n" + build.out.slice(-800));
    if (!dry && existsSync(join(snap.dir, "migrations"))) {
      step("Updating the live database");
      const m = await this.wrangler(
        ["d1", "migrations", "apply", "DB", "--remote"],
        snap.dir,
      );
      if (m.code !== 0)
        throw new Error(
          "The live database could not be updated.\n" + m.out.slice(-600),
        );
    }
    step(dry ? "Checking the deployment (dry run)" : "Going live");
    const d = await this.wrangler(
      dry
        ? ["deploy", "--dry-run", "--outdir", join(snap.dir, ".out")]
        : ["deploy"],
      snap.dir,
    );
    if (d.code !== 0)
      throw new Error(
        "Cloudflare would not publish this app.\n" + d.out.slice(-800),
      );
    const url = d.out.match(/https:\/\/[\w.-]+\.workers\.dev/)?.[0];
    if (!dry && url) {
      step("Checking that it works");
      let ok = false;
      for (let i = 0; i < 20 && !ok; i++) {
        try {
          const [h, p] = await Promise.all([
            fetch(url + "/api/health", { signal: AbortSignal.timeout(5000) }),
            fetch(url, { signal: AbortSignal.timeout(5000) }),
          ]);
          ok = h.ok && p.ok;
        } catch {}
        if (!ok) await new Promise((r) => setTimeout(r, 1500));
      }
      if (!ok)
        throw new Error(
          `Published to ${url}, but it did not answer yet. Try the link again in a minute.`,
        );
    }
    await this.prune(app).catch(() => {});
    return { url, hash: snap.hash, dryRun: dry };
  }
}
