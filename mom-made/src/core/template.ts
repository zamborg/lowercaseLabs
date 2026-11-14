import { cp, mkdir, writeFile, symlink, readFile } from "node:fs/promises";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import type { AppRecord } from "./contracts";
export async function createProject(
  root: string,
  base: string,
  name: string,
): Promise<AppRecord> {
  const id = randomUUID(),
    path = join(root, "workspaces", "default", "apps", id);
  await mkdir(path, { recursive: true });
  await cp(join(base, "template"), path, { recursive: true });
  const app: AppRecord = {
    schemaVersion: 1,
    id,
    name,
    template: "react-worker-v1",
    createdAt: new Date().toISOString(),
    path,
    open: true,
  };
  const pkg = JSON.parse(await readFile(join(base, "package.json"), "utf8"));
  await writeFile(
    join(path, "package.json"),
    JSON.stringify(
      {
        name: "mom-made-" + id,
        version: "0.0.1",
        type: "module",
        private: true,
        scripts: { dev: "vite", build: "vite build" },
        dependencies: Object.fromEntries(
          [
            "react",
            "react-dom",
            "vite",
            "@cloudflare/vite-plugin",
            "hono",
            "wrangler",
          ].map((k) => [k, pkg.dependencies[k]]),
        ),
      },
      null,
      2,
    ),
  );
  await symlink(join(base, "node_modules"), join(path, "node_modules"), "dir");
  await writeFile(
    join(path, "mom-made.json"),
    JSON.stringify(
      {
        schemaVersion: 1,
        id,
        name,
        template: app.template,
        createdAt: app.createdAt,
      },
      null,
      2,
    ),
  );
  await writeFile(
    join(path, "wrangler.jsonc"),
    JSON.stringify(
      {
        name: "mm-" + id,
        main: "worker/index.ts",
        compatibility_date: "2026-11-13",
        assets: {
          directory: "./dist/client",
          not_found_handling: "single-page-application",
          run_worker_first: ["/api/*"],
        },
        d1_databases: [
          { binding: "DB", database_name: "mm-" + id, database_id: id },
        ],
      },
      null,
      2,
    ),
  );
  await writeFile(
    join(path, "AGENTS.md"),
    `# ${name}\nThis is a mom-made app. Edit this app directory only. React frontend in src/, Hono Worker in worker/. The desktop owns the local dev server; do not start or stop servers. Use D1 binding DB for durable data. Do not deploy cloud resources. Keep secrets out of frontend code. No subagents.\n`,
  );
  return app;
}
