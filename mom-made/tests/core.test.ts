import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  mkdirSync,
  rmSync,
  writeFileSync,
  readFileSync,
  existsSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Store } from "../src/core/store";
import { Command, AppManifest, WORKSPACE_CHAT_ID } from "../src/core/contracts";
import { createProject } from "../src/core/template";
import { Codex } from "../src/core/codex";
import { ChatMarkdown } from "../src/ui/chat-markdown";
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
    Command.safeParse({ type: "openApp", appId: "../../etc" }).success,
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
test("workspace chat stays separate from app history and app commands", () => {
  const root = mkdtempSync(join(tmpdir(), "mom-chat-"));
  const store = new Store(root);
  try {
    store.message("user", "App conversation", "fixture-app");
    store.message("user", "Workspace question", WORKSPACE_CHAT_ID);
    assert.deepEqual(store.messages(1, "fixture-app").map((m) => m.text), ["App conversation"]);
    assert.deepEqual(store.messages(1, WORKSPACE_CHAT_ID).map((m) => m.text), ["Workspace question"]);
    assert.equal(Command.safeParse({ type: "send", appId: WORKSPACE_CHAT_ID, text: "Hi", model: "", effort: "" }).success, true);
    assert.equal(Command.safeParse({ type: "openApp", appId: WORKSPACE_CHAT_ID }).success, false);
    assert.equal(Command.safeParse({ type: "openChatLink", url: "file:///etc/passwd" }).success, false);
  } finally {
    store.close();
    rmSync(root, { recursive: true });
  }
});
test("assistant Markdown formats text without loading images or unsafe links", () => {
  const text = [
    "## Heading",
    "**Bold** and `code`",
    "| Day | Plan |\n| --- | --- |\n| Saturday | Hike |",
    "[Docs](https://example.com) [Unsafe](file:///etc/passwd) ![Illustration](https://example.com/remote.png)",
    "<iframe src=https://example.com></iframe>",
  ].join("\n\n");
  const html = renderToStaticMarkup(createElement(ChatMarkdown, { id: "fixture", text, openLink: () => {} }));
  assert.match(html, /<h2>Heading<\/h2>/);
  assert.match(html, /<strong>Bold<\/strong>/);
  assert.match(html, /<code>code<\/code>/);
  assert.match(html, /<table>/);
  assert.match(html, /href="https:\/\/example\.com"/);
  assert.match(html, /Unsafe Illustration/);
  assert.doesNotMatch(html, /<(iframe|img)\b|file:\/\/|remote\.png/);
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

test("agent tools reject stale navigation and the bridge rejects bad tokens", async () => {
  const { staleNavigation, TOOLS } = await import("../src/core/tools");
  const { Bridge } = await import("../src/core/bridge");
  const { createConnection } = await import("node:net");
  assert.equal(staleNavigation(1, 2), true);
  assert.equal(staleNavigation(2, 2), false);
  assert.equal(staleNavigation(undefined, 2), false);
  assert.ok(TOOLS.some((t) => t.name === "app_screenshot"));
  const calls: string[] = [];
  const b = new Bridge(async (app, tool) => {
    calls.push(app + tool);
    return { content: [{ type: "text", text: "ok" }] };
  });
  await b.listen();
  const ask = (token: string, tool: string) =>
    new Promise<any>((resolve) => {
      const c = createConnection(b.path);
      let buf = "";
      c.on("data", (d) => (buf += d));
      c.on("end", () => resolve(JSON.parse(buf)));
      c.write(JSON.stringify({ token, appId: "a", tool, args: {} }) + "\n");
    });
  try {
    assert.equal((await ask("wrong", "app_logs")).isError, true);
    assert.equal((await ask(b.token, "rm_rf")).isError, true);
    assert.equal((await ask(b.token, "app_logs")).content[0].text, "ok");
    assert.deepEqual(calls, ["aapp_logs"]);
  } finally {
    b.close();
  }
});
test("MCP shim speaks stdio MCP and forwards calls over the bridge", async () => {
  const { Bridge } = await import("../src/core/bridge");
  const b = new Bridge(async (app, tool) => ({
    content: [{ type: "text", text: app + ":" + tool }],
  }));
  await b.listen();
  const p = spawn(process.execPath, [resolve("dist/mcp.cjs")], {
    env: { ...process.env, MOM_SOCKET: b.path, MOM_TOKEN: b.token, MOM_APP: "app1" },
  });
  const lines: any[] = [];
  let buf = "";
  p.stdout.on("data", (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      lines.push(JSON.parse(buf.slice(0, i)));
      buf = buf.slice(i + 1);
    }
  });
  try {
    p.stdin.write('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n');
    p.stdin.write('{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n');
    p.stdin.write('{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"app_logs","arguments":{}}}\n');
    for (let i = 0; i < 60 && lines.length < 3; i++) await sleep(50);
    assert.equal(lines.find((l) => l.id === 1).result.serverInfo.name, "mom-made");
    assert.ok(lines.find((l) => l.id === 2).result.tools.length >= 5);
    assert.equal(lines.find((l) => l.id === 3).result.content[0].text, "app1:app_logs");
  } finally {
    p.kill();
    b.close();
  }
});
test("feedback, deployments and message migration persist", () => {
  const root = mkdtempSync(join(tmpdir(), "mom-fb-"));
  const s = new Store(root);
  s.saveFeedback({ id: "f", appId: "a", comment: "c", image: "/x.png", status: "open", createdAt: "2026-01-01", marks: [], viewport: { width: 1, height: 1 } });
  s.deployment("a", true, "https://x.workers.dev", "Published");
  s.message("activity", "Edited 2 files", "a", "t", { kind: "edit", detail: "a\nb" });
  s.close();
  const r = new Store(root);
  assert.equal(r.feedback()[0].status, "open");
  assert.equal(r.deployments("a")[0].url, "https://x.workers.dev");
  assert.equal(r.messages()[0].kind, "edit");
  r.close();
  rmSync(root, { recursive: true });
});
async function runAgent(agent: any, input: any) {
  const events: any[] = [];
  await agent.run(input, (e: any) => events.push(e));
  return events;
}
const fakeInput = (root: string) => ({
  cwd: root,
  text: "hello",
  images: [],
  model: "",
  effort: "",
  instructions: "be nice",
  mcp: { command: process.execPath, args: [join(root, "dist", "mcp.cjs")], env: {} },
});
test("Claude Code adapter maps stream-json into product events and handles failure", async () => {
  const { chmodSync } = await import("node:fs");
  const { ClaudeAgent } = await import("../src/core/agents/claude");
  const root = mkdtempSync(join(tmpdir(), "mom-cc-"));
  const fake = join(root, "claude");
  writeFileSync(
    fake,
    `#!/usr/bin/env node
const out = (o) => process.stdout.write(JSON.stringify(o) + "\\n");
let buf = ""; process.stdin.on("data", d => buf += d);
process.stdin.on("end", () => {
  const msg = JSON.parse(buf.split("\\n")[0]);
  if (require("fs").existsSync(__dirname + "/fail")) { out({type:"result",is_error:true,result:"Failed to authenticate"}); return; }
  out({type:"system",subtype:"init",session_id:"sess-1"});
  out({type:"stream_event",event:{type:"message_start"}});
  out({type:"stream_event",event:{type:"content_block_delta",delta:{type:"text_delta",text:"Hi " + msg.message.content[0].text}}});
  out({type:"assistant",message:{content:[{type:"tool_use",name:"Edit",input:{file_path:process.cwd()+"/src/a.tsx"}},{type:"tool_use",name:"mcp__mom-made__app_screenshot",input:{}}]}});
  out({type:"result",subtype:"success",is_error:false,result:"done"});
});`,
  );
  chmodSync(fake, 0o755);
  process.env.MOM_MADE_CLAUDE = fake;
  try {
    const ev = await runAgent(new ClaudeAgent(), fakeInput(root));
    assert.deepEqual(ev.map((e) => e.type), ["session", "textBreak", "text", "activity", "activity", "done"]);
    assert.equal(ev[2].delta, "Hi hello");
    assert.equal(ev[3].text, "Edited 1 file");
    assert.equal(ev[3].detail, "src/a.tsx");
    assert.equal(ev[4].text, "Looked at the app");
    assert.equal(ev[5].status, "completed");
    writeFileSync(join(root, "fail"), "1");
    const bad = await runAgent(new ClaudeAgent(), fakeInput(root));
    assert.equal(bad.at(-1).status, "failed");
    assert.match(bad.at(-1).error, /authenticate/);
  } finally {
    delete process.env.MOM_MADE_CLAUDE;
    rmSync(root, { recursive: true });
  }
});
test("Pi adapter speaks RPC, streams text and tools, and settles", async () => {
  const { chmodSync } = await import("node:fs");
  const { PiAgent } = await import("../src/core/agents/pi");
  const root = mkdtempSync(join(tmpdir(), "mom-pi-"));
  const fake = join(root, "pi");
  writeFileSync(
    fake,
    `#!/usr/bin/env node
const out = (o) => process.stdout.write(JSON.stringify(o) + "\\n");
if (!process.env.OPENROUTER_API_KEY) process.exit(3);
let buf = "";
process.stdin.on("data", d => { buf += d; let i; while ((i = buf.indexOf("\\n")) >= 0) { const m = JSON.parse(buf.slice(0, i)); buf = buf.slice(i + 1);
  if (m.type === "get_state") out({id:m.id,type:"response",command:"get_state",success:true,data:{sessionFile:"/s/1"}});
  if (m.type === "get_available_models") out({id:m.id,type:"response",command:"get_available_models",success:true,data:{models:[{id:"anthropic/claude-sonnet-4.5",name:"Sonnet",provider:"openrouter",reasoning:true},{id:"x",provider:"openai"}]}});
  if (m.type === "prompt") { out({id:m.id,type:"response",command:"prompt",success:true,data:{disposition:"started"}});
    out({type:"message_start",message:{role:"assistant"}});
    out({type:"message_update",assistantMessageEvent:{type:"text_delta",delta:"ok " + m.message}});
    out({type:"tool_execution_start",toolName:"bash",args:{command:"ls -la"}});
    out({type:"tool_execution_start",toolName:"edit",args:{path:"src/b.tsx"}});
    out({type:"agent_settled",aborted:false}); } } });
process.stdin.on("end", () => process.exit(0));`,
  );
  chmodSync(fake, 0o755);
  process.env.MOM_MADE_PI = fake;
  try {
    const none = new PiAgent({ getKey: () => undefined });
    assert.equal((await none.probe()).ready, false);
    const pi = new PiAgent({ getKey: () => "fixture-key" }, async (url, options) => {
      assert.equal(url, "https://openrouter.ai/api/v1/key");
      assert.deepEqual(options?.headers, { Authorization: "Bearer fixture-key" });
      return new Response("{}");
    });
    const probe = await pi.probe();
    assert.equal(probe.ready, true);
    assert.deepEqual(probe.models.map((m) => m.model), ["anthropic/claude-sonnet-4.5"]);
    const rejected = new PiAgent({ getKey: () => "rejected-key" }, async () => new Response(null, { status: 401 }));
    assert.match((await rejected.probe()).detail ?? "", /rejected this key/);
    const ev = await runAgent(pi, fakeInput(root));
    const types = ev.map((e) => e.type);
    assert.ok(types.includes("session") && types.includes("text"));
    assert.equal(ev.find((e) => e.text === "Looked through the files")?.kind, "look");
    assert.equal(ev.find((e) => e.text === "Edited 1 file")?.detail, "src/b.tsx");
    assert.equal(ev.at(-1).status, "completed");
    const noKey = await runAgent(none, fakeInput(root));
    assert.equal(noKey.at(-1).status, "failed");
  } finally {
    delete process.env.MOM_MADE_PI;
    rmSync(root, { recursive: true });
  }
});

test("model routing respects the assistant catalog and saved defaults", async () => {
  const { resolveModel } = await import("../src/core/model-routing");
  const models = [
    { model: "balanced", displayName: "Balanced", isDefault: true, defaultReasoningEffort: "medium",
      supportedReasoningEfforts: [{ reasoningEffort: "medium", description: "" }, { reasoningEffort: "high", description: "" }] },
    { model: "fast", displayName: "Fast", isDefault: false, defaultReasoningEffort: "", supportedReasoningEfforts: [] },
  ];
  const saved = { model: "fast", effort: "high" };
  assert.deepEqual(resolveModel(models, saved), { model: "fast", effort: "" });
  assert.deepEqual(resolveModel(models, saved, { model: "balanced", effort: "high" }), { model: "balanced", effort: "high" });
  assert.deepEqual(resolveModel(models, saved, { model: "another-provider", effort: "high" }), { model: "fast", effort: "" });
  assert.deepEqual(resolveModel(models, { model: "retired", effort: "high" }), { model: "balanced", effort: "medium" });
  assert.deepEqual(resolveModel([], saved), { model: "", effort: "" });
});

test("local Pi confines inference to loopback and keeps sessions separate", async () => {
  const { defaultPiLocalSettings } = await import("../src/core/contracts");
  const { checkLocalPi, configureLocalPi, LOCAL_PI_PROVIDER, localPiDirectory, localPiSession, loopbackBaseUrl } =
    await import("../src/core/agents/pi-local");
  const root = mkdtempSync(join(tmpdir(), "mom-local-pi-"));
  try {
    const settings = defaultPiLocalSettings;
    assert.equal(loopbackBaseUrl("http://localhost:8080/v1/"), settings.baseUrl);
    assert.throws(() => loopbackBaseUrl("https://example.com/v1"), /local server URL/);
    assert.throws(() => loopbackBaseUrl("http://localhost@example.com/v1"), /local server URL/);
    const directory = localPiDirectory(root, settings);
    const session = join(directory, "sessions", "chat.jsonl");
    mkdirSync(join(directory, "sessions"), { recursive: true });
    writeFileSync(session, "");
    assert.match(localPiSession(directory, session) ?? "", /\/sessions\/chat\.jsonl$/);
    assert.equal(localPiSession(directory, join(root, "cloud.jsonl")), undefined);
    assert.equal(localPiSession(localPiDirectory(root, { ...settings, model: "another-model" }), session), undefined);
    configureLocalPi(directory, settings);
    const config = JSON.parse(readFileSync(join(directory, "models.json"), "utf8"));
    assert.equal(config.providers[LOCAL_PI_PROVIDER].baseUrl, settings.baseUrl);
    assert.deepEqual(config.providers[LOCAL_PI_PROVIDER].models[0].input, ["text"]);
    const ok = await checkLocalPi(settings, async (url, options) => {
      assert.equal(url, settings.baseUrl + "/models");
      assert.equal(options?.redirect, "error");
      return Response.json({ data: [{ id: "flint", meta: { n_ctx: 65536 } }] });
    });
    assert.equal(ok, undefined);
    const wrongModel = await checkLocalPi(settings, async () => Response.json({ data: [{ id: "other" }] }));
    assert.match(wrongModel ?? "", /isn't listed/);
    const wrongContext = await checkLocalPi(settings, async () => Response.json({ data: [{ id: "flint", meta: { n_ctx: 8192 } }] }));
    assert.match(wrongContext ?? "", /Context tokens/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
