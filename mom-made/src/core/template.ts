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
          {
            binding: "DB",
            database_name: "mm-" + id,
            database_id: id,
            migrations_dir: "migrations",
          },
        ],
      },
      null,
      2,
    ),
  );
  await writeFile(join(path, "AGENTS.md"), agentInstructions(name));
  return app;
}

export function agentInstructions(name: string) {
  return `# ${name}

This is a small web app made inside mom-made, for someone who is not a programmer. Make it work, make it look cared-for, and keep your replies short, warm and in plain language (no jargon, no file paths unless asked).

## Rules
- Edit only this app directory. React frontend in src/, Hono Worker API in worker/ (same-origin /api/*).
- mom-made runs the dev server and reloads on save. Never start or stop servers. Never deploy; the Publish button does that.
- Durable data goes in the D1 binding DB. Change the schema only by ADDING a new numbered file in migrations/ (0002_*.sql, ...). Never edit an applied migration and never create tables inside request handlers.
- Never put secrets in frontend code or commit them. If an API key is needed, say so and ask the person to supply it.
- Keep dependencies to what is already installed unless truly necessary.
- If the person wants the app private or wants sign-in, follow the "Sign-in and privacy" guidance you were given. Cloudflare Access is the easiest way to limit who can open an app, and the person turns it on in Cloudflare.
- No subagents.

## Look at your work
You have a "mom-made" tool set. After a change, inspect the running page with app_page_text and app_console. Use app_screenshot when your assistant supports images. Use app_logs if the page is blank or erroring.
`;
}
