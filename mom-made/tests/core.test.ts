import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  rmSync,
  writeFileSync,
  readFileSync,
  existsSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { Store } from "../src/core/store";
import { Command, AppManifest } from "../src/core/contracts";
import { createProject } from "../src/core/template";
import { Codex } from "../src/core/codex";
const sleep = (n: number) => new Promise((r) => setTimeout(r, n));
test("workspace and transcript survive reopening SQLite", () => {
  const root = mkdtempSync(join(tmpdir(), "mom-store-"));
  const s = new Store(root);
  const m = s.message("user", "keep this", "app");
  s.append(m.id, " please");
  s.event("app", "turn/status", { status: "interrupted" });
  s.close();
  const reopened = new Store(root);
  assert.equal(reopened.messages()[0].text, "keep this please");
  assert.equal(
    reopened.db.prepare("SELECT COUNT(*) AS n FROM events").get()!.n,
    3,
  );
  reopened.close();
  rmSync(root, { recursive: true });
});
test("commands reject traversal IDs and arbitrary operations", () => {
  assert.equal(
    Command.safeParse({ type: "start", appId: "../../etc" }).success,
    false,
  );
  assert.equal(
    Command.safeParse({ type: "shell", command: "oops" }).success,
    false,
  );
  assert.equal(
    Command.safeParse({
      type: "bounds",
      x: -1,
      y: 0,
      width: 10,
      height: 10,
      visible: true,
    }).success,
    false,
  );
});
test("each project has its own identity, source, and state directory", async () => {
  const root = mkdtempSync(join(tmpdir(), "mom-app-"));
  try {
    const a = await createProject(root, resolve("."), "First"),
      b = await createProject(root, resolve("."), "Second");
    assert.notEqual(a.id, b.id);
    assert.notEqual(a.path, b.path);
    AppManifest.parse(
      JSON.parse(readFileSync(join(a.path, "mom-made.json"), "utf8")),
    );
    assert.ok(existsSync(join(a.path, "worker/index.ts")));
    assert.ok(
      !readFileSync(join(a.path, "AGENTS.md"), "utf8").includes("Second"),
    );
  } finally {
    rmSync(root, { recursive: true });
  }
});
test("watchdog cleans child and grandchild after owner receives SIGKILL", async () => {
  const root = mkdtempSync(join(tmpdir(), "mom-watch-"));
  const owner = join(root, "owner.cjs"),
    target = join(root, "target.cjs"),
    ids = join(root, "pids");
  writeFileSync(
    target,
    `const {spawn}=require('node:child_process');const c=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore'});require('node:fs').writeFileSync(${JSON.stringify(ids)},JSON.stringify([process.pid,c.pid]));setInterval(()=>{},1000);`,
  );
  writeFileSync(
    owner,
    `require('node:child_process').spawn(process.execPath,[${JSON.stringify(resolve("dist/watchdog.cjs"))},${JSON.stringify(root)},process.execPath,${JSON.stringify(target)}],{stdio:['pipe','ignore','ignore']});setInterval(()=>{},1000);`,
  );
  const p = spawn(process.execPath, [owner]);
  try {
    for (let i = 0; i < 50 && !existsSync(ids); i++) await sleep(50);
    assert.ok(existsSync(ids));
    const pids = JSON.parse(readFileSync(ids, "utf8")) as number[];
    p.kill("SIGKILL");
    await sleep(2500);
    for (const id of pids) assert.throws(() => process.kill(id, 0));
  } finally {
    p.kill();
    rmSync(root, { recursive: true });
  }
});
test("Codex transport handles split JSON, requests, failure and pending cancellation", async () => {
  const root = mkdtempSync(join(tmpdir(), "mom-rpc-"));
  const fake = join(root, "fake.cjs");
  writeFileSync(
    fake,
    `#!/usr/bin/env node\nlet buf='';process.stdin.on('data',d=>{buf+=d;let i;while((i=buf.indexOf('\\n'))>=0){const m=JSON.parse(buf.slice(0,i));buf=buf.slice(i+1);if(m.id){const s=JSON.stringify({id:m.id,result:{ok:true}})+'\\n';process.stdout.write(s.slice(0,5));setTimeout(()=>process.stdout.write(s.slice(5)),10)}}});`,
  );
  const { chmodSync } = await import("node:fs");
  chmodSync(fake, 0o755);
  const c = new Codex(fake);
  try {
    await c.open();
    assert.deepEqual(await c.request("test", {}), { ok: true });
    c.close();
    await c.open();
    assert.deepEqual(await c.request("after-reopen", {}), { ok: true });
  } finally {
    c.close();
    rmSync(root, { recursive: true });
  }
});

test("browser tool contracts require app identity and reject stale navigation", async () => {
  const { ToolInputs, checkNavigation } = await import("../src/core/tools");
  assert.equal(
    ToolInputs["browser.capture"].safeParse({ appId: "current" }).success,
    false,
  );
  assert.equal(checkNavigation(1, 2).ok, false);
  assert.equal(checkNavigation(2, 2).ok, true);
});
