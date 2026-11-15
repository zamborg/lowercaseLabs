import {
  app,
  BrowserWindow,
  WebContentsView,
  ipcMain,
  Menu,
  Tray,
  nativeImage,
  shell,
  safeStorage,
} from "electron";
import { join, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { Bridge } from "../core/bridge";
import { Publisher } from "../core/publish";
import { staleNavigation, type ToolReply } from "../core/tools";
import type { Publish } from "../core/contracts";
import { Store } from "../core/store";
import { Supervisor, installNodeShim } from "../core/runtime";
import { createProject } from "../core/template";
import { resolveModel } from "../core/model-routing";
import { CodexAgent } from "../core/agents/codex";
import { ClaudeAgent } from "../core/agents/claude";
import { PiAgent } from "../core/agents/pi";
import { loopbackBaseUrl } from "../core/agents/pi-local";
import type { Agent, AgentEvent, Probe } from "../core/agents/types";
import {
  Command,
  CHAT_PANE,
  WORKSPACE_CHAT_ID,
  defaultPiLocalSettings,
  type State,
  type AppRecord,
  type Approval,
  type AgentId,
  type AgentInfo,
  type ModelDefaults,
  type PiLocalSettings,
} from "../core/contracts";
import { runSpikes } from "./spikes";
const base = resolve(__dirname, "..");
let win: BrowserWindow,
  store: Store,
  supervisor: Supervisor,
  tray: Tray,
  bridge: Bridge,
  publisher: Publisher;
let finalExitCode = 0;
let quitting = false,
  closed = false;
let selected: string | undefined;
let chatWidth = CHAT_PANE.initial;
let chatSessions: State["chatSessions"] = {};
let piLocal = defaultPiLocalSettings;
let infos: AgentInfo[] = [],
  agentId: AgentId = "codex",
  busyAgent: AgentId | undefined,
  modelDefaults: ModelDefaults = {},
  error: string | undefined;
let busy: State["busy"];
let cloudflare: State["cloudflare"] = { state: "unknown" };
let cloudflareRequest: Promise<State["cloudflare"]> | undefined;
const publishes = new Map<string, Publish>();
const consoleLog = new Map<string, string[]>();
let busyFeedback: string | undefined;
let approval: Approval | undefined;
let bounds = { x: 370, y: 130, width: 700, height: 600, visible: true };
const views = new Map<string, WebContentsView>();
let viewGeneration = new Map<string, number>();
function snapshot(): State {
  return {
    apps: store.apps(),
    selected,
    chatWidth,
    chatSessions,
    messages: store.messages(400, selected ?? WORKSPACE_CHAT_ID),
    runtimes: supervisor.states,
    busy,
    approval,
    account: infos.find((i) => i.id === agentId)?.account ?? "",
    agent: agentId,
    agents: infos,
    modelDefaults,
    openRouterKeySaved: !!store.get("openrouterKey", null),
    piLocal,
    error,
    feedback: store.feedback(),
    publish: Object.fromEntries(
      store.apps().map((a) => [
        a.id,
        publishes.get(a.id) ?? {
          appId: a.id,
          status: "idle",
          log: "",
          history: store.deployments(a.id),
        },
      ]),
    ),
    cloudflare,
    version: app.getVersion(),
  };
}
function emit() {
  if (quitting) return;
  if (win && !win.isDestroyed()) win.webContents.send("state", snapshot());
}
function checkCloudflare(login = false) {
  if (cloudflareRequest) {
    if (!login) return cloudflareRequest;
    throw Error(
      "A Cloudflare connection check is already running. Try again in a moment.",
    );
  }
  cloudflareRequest = (login ? publisher.login() : publisher.whoami())
    .then((status) => {
      cloudflare = status;
      emit();
      return status;
    })
    .finally(() => {
      cloudflareRequest = undefined;
    });
  return cloudflareRequest;
}
function requireApp(id: string) {
  const a = store.apps().find((a) => a.id === id);
  if (!a) throw Error("Unknown app");
  return a;
}
function renderView() {
  if (!win || win.isDestroyed()) return;
  for (const [id, v] of views) {
    const visible =
      id === selected &&
      bounds.visible &&
      !closed &&
      supervisor.states[id]?.status === "ready";
    if (id === selected)
      v.setBounds({
        x: bounds.x,
        y: bounds.y,
        width: Math.max(
          0,
          Math.min(bounds.width, win.getContentBounds().width - bounds.x),
        ),
        height: Math.max(
          0,
          Math.min(bounds.height, win.getContentBounds().height - bounds.y),
        ),
      });
    v.setVisible(visible);
  }
}
async function preview(a: AppRecord) {
  const url = supervisor.states[a.id]?.url;
  if (!url) return;
  let v = views.get(a.id);
  if (!v) {
    v = new WebContentsView({
      webPreferences: {
        partition: "persist:app-" + a.id,
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
      },
    });
    views.set(a.id, v);
    v.setVisible(false);
    win.contentView.addChildView(v);
    const wc = v.webContents;
    wc.session.setPermissionRequestHandler((_, __, cb) => cb(false));
    wc.session.setPermissionCheckHandler(() => false);
    wc.setWindowOpenHandler(() => ({ action: "deny" }));
    wc.on("will-navigate", (e, target) => {
      if (
        new URL(target).origin !==
        new URL(supervisor.states[a.id]?.url ?? url).origin
      )
        e.preventDefault();
    });
    wc.on("did-start-navigation", () =>
      viewGeneration.set(a.id, (viewGeneration.get(a.id) ?? 0) + 1),
    );
    wc.session.on("will-download", (e) => e.preventDefault());
    wc.on("console-message", (e) => {
      if (e.level === "error" || e.level === "warning") {
        const l = consoleLog.get(a.id) ?? [];
        l.push(`${e.level}: ${e.message}`.slice(0, 600));
        consoleLog.set(a.id, l.slice(-60));
      }
    });
    wc.on("did-fail-load", (_e, code, desc, url, main) => {
      if (main && code !== -3) {
        const l = consoleLog.get(a.id) ?? [];
        l.push(`page failed to load (${desc}): ${url}`);
        consoleLog.set(a.id, l.slice(-60));
      }
    });
    wc.on("did-start-navigation", (_e, _u, _ip, main) => {
      if (main) consoleLog.set(a.id, []);
    });
  }
  if (
    new URL(v.webContents.getURL() || url).origin !== new URL(url).origin ||
    !v.webContents.getURL()
  )
    await v.webContents.loadURL(url);
  renderView();
}
const appStarts = new Map<string, Promise<void>>();
const appClosures = new Map<string, Promise<void>>();
function start(a: AppRecord): Promise<void> {
  const pending = appStarts.get(a.id);
  if (pending) return pending;
  const task = (async () => {
    try {
      await supervisor.start(a);
      const current = requireApp(a.id);
      if (a.port !== current.port) store.saveApp({ ...current, port: a.port });
      if (
        current.open &&
        !closed &&
        !quitting &&
        supervisor.states[a.id]?.status === "ready"
      )
        await preview(current);
    } catch (e) {
      const runtime = supervisor.states[a.id];
      if (runtime) {
        runtime.status = "failed";
        runtime.error = String(e);
      }
    }
    emit();
  })().finally(() => appStarts.delete(a.id));
  appStarts.set(a.id, task);
  return task;
}

const reply = (text: string, isError = false): ToolReply => ({
  content: [{ type: "text", text }],
  ...(isError ? { isError } : {}),
});
async function grab(appId: string) {
  const v = views.get(appId);
  if (!v) throw Error("no view");
  let img = await v.webContents
    .capturePage(undefined, { stayHidden: true })
    .catch(() => undefined);
  if ((!img || img.isEmpty()) && bounds.visible) {
    // A hidden native view cannot be captured; show it briefly, then restore.
    v.setVisible(true);
    v.setBounds({
      x: bounds.x,
      y: bounds.y,
      width: Math.max(bounds.width, 400),
      height: Math.max(bounds.height, 300),
    });
    await new Promise((r) => setTimeout(r, 200));
    img = await v.webContents.capturePage().catch(() => undefined);
    renderView();
  }
  if (!img || img.isEmpty()) throw Error("The page could not be captured.");
  return img;
}
const PAGE_SCRIPT = `(() => {
  const vis = (e) => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e); return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none"; };
  const label = (e) => (e.getAttribute("aria-label") || e.innerText || e.value || e.placeholder || e.name || "").trim().slice(0, 80);
  const els = [...document.querySelectorAll("button,a[href],input,textarea,select,[role=button],summary")].filter(vis).slice(0, 60).map((e) => {
    const id = e.id ? "#" + e.id : "";
    return e.tagName.toLowerCase() + id + (e.type ? "[" + e.type + "]" : "") + ': "' + label(e) + '"' + (e.disabled ? " (disabled)" : "");
  });
  return JSON.stringify({ title: document.title, url: location.href, text: document.body.innerText.slice(0, 5000), elements: els });
})()`;
async function tool(
  appId: string,
  name: string,
  args: Record<string, unknown>,
): Promise<ToolReply> {
  const rt = supervisor.states[appId];
  if (!store.apps().some((a) => a.id === appId))
    return reply("Unknown app.", true);
  if (name === "app_logs")
    return reply((rt?.logs || "No server output yet.").slice(-6000));
  const v = views.get(appId);
  if (!v || rt?.status !== "ready")
    return reply(
      "The app is not running right now. Its status is " +
        (rt?.status ?? "stopped") +
        ". Check app_logs.",
      true,
    );
  const wc = v.webContents;
  const gen = viewGeneration.get(appId) ?? 0;
  if (staleNavigation(args.navigationId, gen))
    return reply(
      `The page changed (navigationId is now ${gen}). Call app_page_text again before acting.`,
      true,
    );
  const js = (code: string) => wc.executeJavaScript(code, true);
  switch (name) {
    case "app_screenshot": {
      const img = await grab(appId);
      const size = img.getSize();
      const scaled = size.width > 1400 ? img.resize({ width: 1400 }) : img;
      return {
        content: [
          {
            type: "image",
            data: scaled.toPNG().toString("base64"),
            mimeType: "image/png",
          },
          { type: "text", text: `navigationId ${gen}` },
        ],
      };
    }
    case "app_page_text": {
      const r = JSON.parse(await js(PAGE_SCRIPT));
      return reply(
        `navigationId ${gen}\n${r.title} — ${r.url}\n\nInteractive:\n${r.elements.join("\n")}\n\nText:\n${r.text}`,
      );
    }
    case "app_console": {
      const l = consoleLog.get(appId) ?? [];
      return reply(l.length ? l.join("\n") : "No console errors or warnings.");
    }
    case "app_reload":
      wc.reload();
      return reply("Reloading.");
    case "app_click": {
      const r = await js(`((text, selector) => {
        const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
        let el;
        if (selector) el = document.querySelector(selector);
        else if (text) {
          const t = text.trim().toLowerCase();
          const all = [...document.querySelectorAll("button,a,[role=button],input[type=submit],summary,label")].filter(vis);
          el = all.find((e) => (e.innerText || e.value || "").trim().toLowerCase() === t) || all.find((e) => (e.innerText || e.value || "").toLowerCase().includes(t));
        }
        if (!el) return "NOT_FOUND";
        el.scrollIntoView({ block: "center" });
        el.click();
        return "OK";
      })(${JSON.stringify(args.text ?? "")}, ${JSON.stringify(args.selector ?? "")})`);
      await new Promise((res) => setTimeout(res, 400));
      return reply(
        r === "OK"
          ? "Clicked. Take a screenshot to see the result."
          : "Nothing matching that was found on the page.",
        r !== "OK",
      );
    }
    case "app_type": {
      const r = await js(`((selector, label, text) => {
        let el = selector ? document.querySelector(selector) : null;
        if (!el && label) {
          const l = [...document.querySelectorAll("label")].find((x) => x.innerText.toLowerCase().includes(label.toLowerCase()));
          el = l ? (l.control || document.getElementById(l.htmlFor)) : [...document.querySelectorAll("input,textarea")].find((x) => (x.placeholder || x.getAttribute("aria-label") || "").toLowerCase().includes(label.toLowerCase()));
        }
        if (!el) return "NOT_FOUND";
        el.focus();
        const proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        Object.getOwnPropertyDescriptor(proto, "value").set.call(el, text);
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return "OK";
      })(${JSON.stringify(args.selector ?? "")}, ${JSON.stringify(args.label ?? "")}, ${JSON.stringify(String(args.text ?? ""))})`);
      return reply(
        r === "OK" ? "Typed." : "No matching field was found.",
        r !== "OK",
      );
    }
  }
  return reply("Unknown tool.", true);
}
const agents: Record<AgentId, Agent> = {
  codex: new CodexAgent((u) => shell.openExternal(u)),
  claude: new ClaudeAgent(),
  pi: new PiAgent({ getKey }),
  "pi-local": new PiAgent({ local: () => piLocal, root: () => store.root }),
};
(agents.codex as CodexAgent).onAccount(() => void checkAgent("codex"));
function getKey(): string | undefined {
  const v = store?.get<string | null>("openrouterKey", null);
  if (!v) return undefined;
  try {
    return safeStorage.decryptString(Buffer.from(v, "base64"));
  } catch {
    return undefined;
  }
}
function setKey(key: string) {
  if (!key) return store.set("openrouterKey", null);
  if (!safeStorage.isEncryptionAvailable())
    throw Error("This Mac can't store the key safely.");
  store.set("openrouterKey", safeStorage.encryptString(key).toString("base64"));
}
const agentIds = Object.keys(agents) as AgentId[];
const agentChecks = new Map<AgentId, Promise<AgentInfo>>();
const agentLogins = new Set<AgentId>();
let changingPiKey = false;
let changingLocalModel = false;
function checkAgent(id: AgentId): Promise<AgentInfo> {
  const pending = agentChecks.get(id);
  if (pending) return pending;
  const a = agents[id];
  const task = a
    .probe()
    .catch((e): Probe => ({
      installed: true,
      ready: false,
      detail: String(e),
      models: [],
    }))
    .then((p) => {
      const info: AgentInfo = {
        id: a.id,
        label: a.label,
        installed: p.installed,
        ready: p.ready,
        account: p.account,
        detail: p.detail,
        models: p.models,
        approvals: a.approvals,
        compact: !!a.compact,
        help: a.help,
      };
      infos = [...infos.filter((item) => item.id !== id), info].sort(
        (a, b) => agentIds.indexOf(a.id) - agentIds.indexOf(b.id),
      );
      emit();
      return info;
    })
    .finally(() => agentChecks.delete(id));
  agentChecks.set(id, task);
  return task;
}
async function connect() {
  await Promise.all(agentIds.map(checkAgent));
}
function finish(status: string) {
  if (busy) {
    const ok = status === "completed";
    if (!ok)
      store.message("activity", turnLabel(status), busy.appId, busy.turnId, {
        kind: "note",
      });
    store.event(busy.appId, "turn/status", { status, turnId: busy.turnId });
    if (busyFeedback) {
      const f = store.feedback().find((x) => x.id === busyFeedback);
      if (f && ok && f.status === "open") {
        f.status = "addressed";
        store.saveFeedback(f);
      }
    }
  }
  store.set("activeTurn", null);
  busy = undefined;
  busyAgent = undefined;
  busyFeedback = undefined;
  approval = undefined;
  emit();
}
function turnLabel(status: string) {
  if (status === "interrupted") return "Stopped.";
  if (status.startsWith("failed")) {
    const why = status.slice(7).trim();
    if (/authenticate|expired|log ?in|sign ?in|unauthor|401|api key/i.test(why))
      return "I couldn't sign in to the assistant. Open Settings to reconnect.";
    return "That didn't work" + (why ? ": " + why.slice(0, 300) : ".");
  }
  return status;
}
const INSTRUCTIONS =
  "You are building an app inside mom-made for someone who is not a programmer. Work only in the specified app directory and follow AGENTS.md there. The desktop supervises the dev server. Do not start servers or deploy. Use React, Vite, Hono and Cloudflare bindings. Look at your work with the mom-made tools before saying you are done. Reply briefly in plain, friendly language. No subagents.";
const WEB_DESIGN = readFileSync(
  join(base, "skills/web-design/SKILL.md"),
  "utf8",
).trim();
const AUTH = readFileSync(join(base, "skills/auth/SKILL.md"), "utf8").trim();
function workspaceChatDirectory() {
  const cwd = join(store.root, "workspaces", "default", "chat");
  mkdirSync(cwd, { recursive: true });
  return cwd;
}
function sessionOf(a: AppRecord | undefined, id: AgentId) {
  if (!a) return chatSessions[id];
  return a.sessions?.[id] ?? (id === "codex" ? a.threadId : undefined);
}
function handler(appId: string, id: AgentId, turnId: string) {
  let assistantId: string | undefined;
  return (e: AgentEvent) => {
    if (busy?.turnId !== turnId) return;
    switch (e.type) {
      case "text":
        if (!assistantId)
          assistantId = store.message("assistant", "", appId, busy.turnId).id;
        store.append(assistantId, e.delta);
        break;
      case "textBreak":
        assistantId = undefined;
        break;
      case "activity":
        store.message("activity", e.text, appId, busy.turnId, {
          kind: e.kind,
          detail: e.detail,
        });
        break;
      case "approval":
        approval = {
          id: e.id,
          method:
            e.what === "files"
              ? "item/fileChange/requestApproval"
              : "item/commandExecution/requestApproval",
          params: { reason: e.reason, command: e.command },
        };
        break;
      case "session":
        if (appId === WORKSPACE_CHAT_ID) {
          chatSessions = { ...chatSessions, [id]: e.id };
          store.set("chatSessions", chatSessions);
        } else {
          const current = requireApp(appId);
          current.sessions = { ...current.sessions, [id]: e.id };
          store.saveApp(current);
        }
        break;
      case "done":
        finish(e.status === "failed" ? "failed: " + (e.error ?? "") : e.status);
        return;
    }
    emit();
  };
}
type SendOpts = {
  images?: string[];
  feedbackId?: string;
  display?: string;
  expectedAgent?: AgentId;
};
function turnAgent(expected?: AgentId) {
  if (expected && expected !== agentId)
    throw Error(
      "The active assistant changed. Review the new assistant and send again.",
    );
  return agentId;
}
async function send(
  a: AppRecord | undefined,
  text: string,
  model: string,
  effort: string,
  opts: SendOpts = {},
) {
  if (busy) throw Error("Finish or stop the current turn first.");
  const id = turnAgent(opts.expectedAgent);
  if (id === "pi" && changingPiKey)
    throw Error("Wait for OpenRouter to finish checking the new key.");
  if (id === "pi-local" && changingLocalModel)
    throw Error("Wait for Pi to finish checking the local model.");
  const info = infos.find((i) => i.id === id);
  if (!info?.installed)
    throw Error(
      `${agents[id].label} isn't installed yet. Open Settings for help.`,
    );
  if (!info.ready)
    throw Error(`Connect ${agents[id].label} in Settings first.`);
  const route = resolveModel(info.models, modelDefaults[id], { model, effort });
  const appId = a?.id ?? WORKSPACE_CHAT_ID;
  const cwd = a?.path ?? workspaceChatDirectory();
  const turnId = randomUUID();
  busy = { appId, turnId };
  busyAgent = id;
  busyFeedback = opts.feedbackId;
  error = undefined;
  store.message("user", opts.display ?? text, appId, turnId, {
    image: opts.images?.[0],
  });
  store.set("activeTurn", { appId, turnId });
  if (opts.feedbackId) {
    const f = store.feedback().find((x) => x.id === opts.feedbackId);
    if (f) store.saveFeedback({ ...f, turnId: busy.turnId });
  }
  store.event(appId, "turn/submitted", { agent: id, text, ...route });
  emit();
  const agent = agents[id];
  void agent
    .run(
      {
        cwd,
        text,
        images: opts.images ?? [],
        ...route,
        session: sessionOf(a, id),
        instructions: a
          ? [
              INSTRUCTIONS,
              id === "pi-local"
                ? "This is a text model. Verify the preview with app_page_text and app_console."
                : "Use app_screenshot for visual verification.",
              WEB_DESIGN,
              AUTH,
            ].join("\n\n")
          : "You are chatting with someone in mom-made. Help with their questions, ideas, and planning. No app is selected. If they want you to build or edit an app, tell them they can choose New App or open an app from the toolbar. Reply directly without using tools or modifying files.",
        mcp: a
          ? {
              command: process.execPath,
              args: [join(base, "dist/mcp.cjs")],
              env: {
                ELECTRON_RUN_AS_NODE: "1",
                MOM_SOCKET: bridge.path,
                MOM_TOKEN: bridge.token,
                MOM_APP: a.id,
              },
            }
          : undefined,
      },
      handler(appId, id, turnId),
    )
    .catch((e) => {
      if (busy?.turnId === turnId)
        finish("failed: " + (e instanceof Error ? e.message : String(e)));
    })
    .finally(() => {
      if (busy?.turnId === turnId)
        finish("failed: The assistant stopped unexpectedly.");
    });
}
async function interrupt() {
  if (busy && busyAgent) await agents[busyAgent].interrupt();
}

let stopping: Promise<void> | undefined;
function stopLocal(): Promise<void> {
  if (stopping) return stopping;
  stopping = (async () => {
    await interrupt().catch(() => {});
    if (busy) finish("interrupted");
    for (const a of Object.values(agents)) a.close();
    await supervisor.stopAll();
    renderView();
  })().finally(() => {
    stopping = undefined;
  });
  return stopping;
}
app.setName("mom-made");
app.setPath(
  "userData",
  process.env.MOM_MADE_DATA || join(app.getPath("appData"), "mom-made"),
);
if (!app.requestSingleInstanceLock()) app.exit(0);
app.whenReady().then(async () => {
  if (process.argv.includes("--spike")) {
    await runSpikes(base);
    app.quit();
    return;
  }
  store = new Store(process.env.MOM_MADE_DATA || app.getPath("userData"));
  installNodeShim(join(store.root, "bin"), process.execPath);
  bridge = new Bridge(tool);
  await bridge.listen();
  publisher = new Publisher(process.execPath, base, store.root);
  const storedSelection = store.get<string | null>("selected", null);
  selected = store.apps().find((a) => a.id === storedSelection && a.open)?.id;
  chatWidth = store.get("chatWidth", CHAT_PANE.initial);
  chatSessions = store.get("chatSessions", {});
  agentId = store.get<AgentId>("agent", "codex");
  modelDefaults = store.get<ModelDefaults>("modelDefaults", {});
  piLocal = store.get<PiLocalSettings>("piLocal", defaultPiLocalSettings);
  const interrupted = store.get<{ appId: string } | null>("activeTurn", null);
  if (interrupted) {
    store.message(
      "activity",
      "Previous turn was interrupted by desktop shutdown.",
      interrupted.appId,
    );
    store.set("activeTurn", null);
  }
  supervisor = new Supervisor(process.execPath, base, () => {
    emit();
    renderView();
  });
  win = new BrowserWindow({
    width: 1240,
    height: 830,
    minWidth: 850,
    minHeight: 600,
    title: "mom-made",
    titleBarStyle: "hiddenInset",
    backgroundColor: "#f7f5f1",
    webPreferences: {
      preload: join(base, "dist/preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });
  await win.loadFile(join(base, "dist/index.html"));
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (e) => e.preventDefault());
  const openWindow = async () => {
    await stopping;
    closed = false;
    win.show();
    for (const a of store.apps().filter((a) => a.open)) void start(a);
    emit();
  };
  app.on("activate", () => void openWindow());
  app.on("second-instance", () => void openWindow());
  win.on("close", (e) => {
    if (!quitting) {
      e.preventDefault();
      closed = true;
      win.hide();
      void stopLocal();
    }
  });
  win.on("resize", renderView);
  const icon = nativeImage.createFromBuffer(
    Buffer.from(
      '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22"><text x="1" y="17" font-size="18" font-family="Georgia">m</text></svg>',
    ),
  );
  tray = new Tray(icon);
  tray.setTitle("m");
  tray.setToolTip("mom-made");
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: "Open mom-made", click: () => void openWindow() },
      { label: "Stop all local apps", click: () => void stopLocal() },
      { type: "separator" },
      { label: "Quit mom-made", click: () => app.quit() },
    ]),
  );
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      {
        label: "mom-made",
        submenu: [
          { role: "about" },
          { type: "separator" },
          {
            label: "Settings…",
            accelerator: "CmdOrCtrl+,",
            click: async () => {
              if (closed) await openWindow();
              win.webContents.send("show-settings");
            },
          },
          {
            label: "Reveal workspace",
            click: () => void shell.openPath(store.root),
          },
          { type: "separator" },
          { role: "quit" },
        ],
      },
      { role: "editMenu" },
      { role: "viewMenu" },
      { role: "windowMenu" },
    ]),
  );
  function trusted(e: Electron.IpcMainInvokeEvent) {
    if (
      e.sender !== win.webContents ||
      e.senderFrame !== win.webContents.mainFrame
    )
      throw Error("Untrusted IPC sender");
  }
  ipcMain.handle("state", (e) => {
    trusted(e);
    return snapshot();
  });
  ipcMain.handle("command", async (e, raw) => {
    trusted(e);
    const c = Command.parse(raw);
    if (c.type === "bounds" || c.type === "previewVisibility") {
      bounds = c.type === "bounds" ? c : { ...bounds, visible: false };
      renderView();
      return;
    }
    error = undefined;
    try {
      switch (c.type) {
        case "create": {
          const a = await createProject(store.root, base, c.name);
          store.saveApp(a);
          selected = a.id;
          store.set("selected", selected);
          emit();
          void start(a);
          return a.id;
        }
        case "select":
        case "openApp": {
          if (c.type === "openApp") await appClosures.get(c.appId);
          const a = requireApp(c.appId);
          const status = supervisor.states[a.id]?.status;
          const wake =
            c.type === "openApp" &&
            (!a.open || !status || status === "stopped");
          if (wake && !a.open) {
            a.open = true;
            store.saveApp(a);
          }
          selected = a.id;
          store.set("selected", selected);
          renderView();
          if (wake) void start(a);
          break;
        }
        case "start": {
          await appClosures.get(c.appId);
          const a = requireApp(c.appId);
          a.open = true;
          store.saveApp(a);
          void start(a);
          break;
        }
        case "stop":
          await supervisor.stop(c.appId);
          renderView();
          break;
        case "close": {
          const closing = appClosures.get(c.appId);
          if (closing) {
            await closing;
            break;
          }
          const a = requireApp(c.appId);
          a.open = false;
          store.saveApp(a);
          if (selected === a.id) {
            selected = store.apps().find((a) => a.open)?.id;
            store.set("selected", selected ?? null);
          }
          renderView();
          const task = (async () => {
            await supervisor.stop(a.id);
            await appStarts.get(a.id);
            views.get(a.id)?.webContents.close();
            views.delete(a.id);
          })().finally(() => appClosures.delete(a.id));
          appClosures.set(a.id, task);
          await task;
          break;
        }
        case "reveal":
          shell.showItemInFolder(requireApp(c.appId).path);
          break;
        case "reload":
          views.get(c.appId)?.webContents.reload();
          break;
        case "capture": {
          const v = views.get(c.appId);
          if (!v) throw Error("Preview is not ready");
          const img = await v.webContents.capturePage();
          const path = join(store.root, "captures");
          await mkdir(path, { recursive: true });
          const file = join(path, `${c.appId}-${Date.now()}.png`);
          await writeFile(file, img.toPNG());
          return {
            path: file,
            data: img.toDataURL(),
            generation: viewGeneration.get(c.appId) ?? 0,
          };
        }
        case "login": {
          const { id } = c;
          const ag = agents[id];
          if (!ag.login) throw Error("Nothing to sign in to here.");
          if (agentLogins.has(id)) break;
          agentLogins.add(id);
          try {
            await ag.login();
            if (id === "claude") await checkAgent(id);
          } finally {
            agentLogins.delete(id);
          }
          break;
        }
        case "agent":
          if (busy) throw Error("Wait for the current work to finish.");
          if (!infos.find((a) => a.id === c.id)?.ready)
            throw Error("Connect this assistant before using it.");
          agentId = c.id;
          store.set("agent", agentId);
          break;
        case "modelDefault": {
          if (!c.model) delete modelDefaults[c.id];
          else {
            const catalog = infos.find((a) => a.id === c.id)?.models ?? [];
            if (!catalog.some((m) => m.model === c.model))
              throw Error(
                "This model is unavailable. Refresh connections and try again.",
              );
            modelDefaults[c.id] = resolveModel(catalog, undefined, c);
          }
          store.set("modelDefaults", modelDefaults);
          break;
        }
        case "agentKey": {
          if (busyAgent === "pi")
            throw Error("Wait for Pi to finish before changing its key.");
          if (changingPiKey)
            throw Error("An OpenRouter key change is already in progress.");
          changingPiKey = true;
          try {
            await agentChecks.get("pi");
            if (quitting) return;
            setKey(c.key);
            await checkAgent("pi");
          } finally {
            changingPiKey = false;
          }
          break;
        }
        case "piLocalSettings": {
          if (busyAgent === "pi-local" || changingLocalModel)
            throw Error(
              "Wait for local Pi to finish before changing its settings.",
            );
          const settings = {
            ...c.settings,
            baseUrl: loopbackBaseUrl(c.settings.baseUrl),
          };
          changingLocalModel = true;
          try {
            await agentChecks.get("pi-local");
            if (quitting) return;
            store.set("piLocal", settings);
            piLocal = settings;
            await checkAgent("pi-local");
          } finally {
            changingLocalModel = false;
          }
          break;
        }
        case "send":
          await send(
            c.appId === WORKSPACE_CHAT_ID ? undefined : requireApp(c.appId),
            c.text,
            c.model,
            c.effort,
            { expectedAgent: c.agent },
          );
          break;
        case "interrupt":
          await interrupt();
          break;
        case "compact": {
          if (busy) throw Error("Wait for the active turn");
          const a = selected ? requireApp(selected) : undefined;
          const ag = agents[agentId];
          const sess = sessionOf(a, agentId);
          if (sess && ag.compact) {
            await ag.compact(a?.path ?? workspaceChatDirectory(), sess);
            store.message(
              "activity",
              "Tidied up the chat",
              a?.id ?? WORKSPACE_CHAT_ID,
              undefined,
              { kind: "note" },
            );
          }
          break;
        }
        case "answer":
          if (!busyAgent || !approval || approval.id !== c.id)
            throw Error("Approval is no longer pending");
          agents[busyAgent].answer(
            c.id,
            c.accept ? (c.always ? "always" : "accept") : "decline",
          );
          approval = undefined;
          break;
        case "recheck":
          await connect();
          break;
        case "revealWorkspace": {
          const problem = await shell.openPath(store.root);
          if (problem) throw Error(problem);
          break;
        }
        case "showChat":
          selected = undefined;
          store.set("selected", null);
          renderView();
          break;
        case "chatWidth":
          chatWidth = c.width;
          store.set("chatWidth", chatWidth);
          break;
        case "openChatLink": {
          const url = new URL(c.url);
          if (url.username || url.password)
            throw Error("This link includes sign-in information.");
          await shell.openExternal(url.href);
          break;
        }
        case "open": {
          const u = new URL(c.url);
          if (u.protocol !== "https:" || !u.hostname.endsWith(".workers.dev"))
            throw Error("That link can't be opened from here.");
          await shell.openExternal(u.href);
          break;
        }
        case "newConversation": {
          if (busy) throw Error("Wait for the current work to finish.");
          const a = selected ? requireApp(selected) : undefined;
          if (a) {
            if (agentId === "codex") a.threadId = undefined;
            if (a.sessions) delete a.sessions[agentId];
            store.saveApp(a);
          } else {
            delete chatSessions[agentId];
            store.set("chatSessions", chatSessions);
          }
          store.message(
            "activity",
            "New conversation",
            a?.id ?? WORKSPACE_CHAT_ID,
            undefined,
            { kind: "divider" },
          );
          break;
        }
        case "rename": {
          const a = requireApp(c.appId);
          a.name = c.name;
          store.saveApp(a);
          break;
        }
        case "feedback": {
          const f = store.feedback().find((x) => x.id === c.id);
          if (f) store.saveFeedback({ ...f, status: c.status });
          break;
        }
        case "markup": {
          if (turnAgent(c.agent) === "pi-local")
            throw Error(
              "Switch assistants to send visual markup. You can describe the change in local Pi chat.",
            );
          const a = requireApp(c.appId);
          const m = /^data:image\/png;base64,([A-Za-z0-9+/=]+)$/.exec(c.image);
          if (!m) throw Error("That drawing could not be read.");
          const dir = join(store.root, "captures");
          await mkdir(dir, { recursive: true });
          const id = randomUUID();
          const file = join(dir, `feedback-${id}.png`);
          await writeFile(file, Buffer.from(m[1], "base64"));
          store.saveFeedback({
            id,
            appId: a.id,
            comment: c.comment,
            image: file,
            status: "open",
            createdAt: new Date().toISOString(),
            marks: c.marks,
            viewport: c.viewport,
            url: supervisor.states[a.id]?.url,
          });
          const pins = c.marks
            .filter((k) => k.type === "pin" && k.points[0])
            .map(
              (k, i) =>
                `pin ${i + 1} at ${Math.round((k.points[0][0] / Math.max(1, c.viewport.width)) * 100)}% across, ${Math.round((k.points[0][1] / Math.max(1, c.viewport.height)) * 100)}% down`,
            );
          const text = `The person marked up a screenshot of "${a.name}" and wrote: "${c.comment}"\n\nThe attached image is that screenshot with their drawing on top (circles, boxes, pins and scribbles show what they mean). Viewport ${c.viewport.width}x${c.viewport.height}.${pins.length ? " " + pins.join("; ") + "." : ""}\n\nMake the change. Then look at the running app with app_screenshot and tell them briefly what you changed.`;
          await send(a, text, c.model, c.effort, {
            images: [file],
            feedbackId: id,
            display: c.comment,
            expectedAgent: c.agent,
          });
          break;
        }
        case "fix": {
          const a = requireApp(c.appId);
          const rt = supervisor.states[a.id];
          const lines = (consoleLog.get(a.id) ?? []).join("\n");
          const text = `The app isn't working. Please find out why and fix it.\n\nStatus: ${rt?.status ?? "unknown"}${rt?.error ? "\nError: " + rt.error : ""}\n\nServer output:\n${(rt?.logs ?? "").slice(-3000)}\n\nBrowser errors:\n${lines || "none"}`;
          await send(a, text, c.model, c.effort, {
            display: "Something's wrong with this app. Can you fix it?",
            expectedAgent: c.agent,
          });
          break;
        }
        case "cloudflareCheck":
          await checkCloudflare();
          break;
        case "cloudflareLogin":
          await checkCloudflare(true);
          break;
        case "publish": {
          const a = requireApp(c.appId);
          if (publishes.get(a.id)?.status === "working")
            throw Error("Already publishing.");
          const state: Publish = {
            appId: a.id,
            status: "working",
            step: "Starting",
            log: "",
            history: store.deployments(a.id),
          };
          publishes.set(a.id, state);
          void publisher
            .publish(
              a,
              (step) => {
                state.step = step;
                emit();
              },
              { dryRun: process.env.MOM_MADE_DRY_RUN === "1" },
            )
            .then((r) => {
              state.status = "done";
              state.url = r.url;
              state.step = undefined;
              store.deployment(
                a.id,
                true,
                r.url,
                r.dryRun ? "Dry run" : "Published",
              );
            })
            .catch((e) => {
              state.status = "failed";
              state.error = String(e instanceof Error ? e.message : e);
              store.deployment(
                a.id,
                false,
                undefined,
                state.error.slice(0, 300),
              );
            })
            .finally(() => {
              state.history = store.deployments(a.id);
              emit();
            });
          break;
        }
      }
      emit();
    } catch (e) {
      error = String(e);
      emit();
      throw e;
    }
  });
  for (const a of store.apps().filter((a) => a.open)) void start(a);
  void connect().catch((e) => {
    error = String(e);
    emit();
  });
  void checkCloudflare();
  emit();
  if (process.argv.includes("--smoke")) {
    void (async () => {
      const a = await createProject(store.root, base, "First app"),
        b = await createProject(store.root, base, "Second app");
      store.saveApp(a);
      store.saveApp(b);
      selected = a.id;
      await start(a);
      await start(b);
      await connect();
      await send(
        a,
        "Edit src/main.tsx only: change the h1 to the exact text Made with mom. Preserve the rest of the application. Then call the mom-made app_screenshot tool to confirm the result. Do not use subagents or start servers.",
        "",
        "",
      );
      selected = b.id;
      store.set("selected", selected);
      renderView();
      emit();
      for (let i = 0; i < 240 && busy; i++)
        await new Promise((r) => setTimeout(r, 500));
      if (busy) throw Error("Agent edit timed out");
      await new Promise((r) => setTimeout(r, 1500));
      const first = await views
        .get(a.id)!
        .webContents.executeJavaScript(
          'document.querySelector("h1").textContent',
        );
      const second = await views
        .get(b.id)!
        .webContents.executeJavaScript(
          'document.querySelector("h1").textContent',
        );
      if (!first.includes("Made with mom") || second.includes("Made with mom"))
        throw Error("Target isolation failed: " + first + " / " + second);
      await fetch(supervisor.states[a.id].url! + "/api/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "desktop-reopen" }),
      });
      selected = a.id;
      renderView();
      emit();
      await new Promise((r) => setTimeout(r, 300));
      await views
        .get(a.id)!
        .webContents.capturePage()
        .then((i) => writeFile(join(base, ".spike/first-app.png"), i.toPNG()));
      selected = b.id;
      renderView();
      emit();
      const dry = await publisher.publish(a, () => {}, { dryRun: true });
      const result = {
        dryRunPublish: dry.hash.length === 64,
        activities: store
          .messages()
          .filter((m) => m.role === "activity")
          .map((m) => m.text),
        targetIsolation: true,
        hotReload: true,
        appCount: store.apps().length,
        firstHeading: first,
        secondHeading: second,
        messages: store.messages().length,
      };
      await writeFile(
        join(base, ".spike/ui.json"),
        JSON.stringify(result, null, 2),
      );
      console.log(JSON.stringify(result));
      app.quit();
    })().catch((e) => {
      console.error(e);
      finalExitCode = 1;
      app.quit();
    });
  }
  if (process.argv.includes("--smoke-reopen")) {
    void (async () => {
      const apps = store.apps();
      for (const a of apps) await start(a);
      const first = apps.find((a) => a.name === "First app")!;
      const notes = (await (
        await fetch(supervisor.states[first.id].url! + "/api/notes")
      ).json()) as { text: string }[];
      const firstHeading = await views
        .get(first.id)!
        .webContents.executeJavaScript(
          'document.querySelector("h1").textContent',
        );
      const second = apps.find((a) => a.name === "Second app")!;
      const secondHeading = await views
        .get(second.id)!
        .webContents.executeJavaScript(
          'document.querySelector("h1").textContent',
        );
      if (
        !firstHeading.includes("Made with mom") ||
        secondHeading.includes("Made with mom")
      )
        throw Error("Target isolation did not persist");
      const result = {
        targetIsolation: true,
        firstHeading,
        secondHeading,
        restoredApps: apps.filter((a) => a.open).length,
        restoredMessages: store.messages().length,
        restoredSelection:
          selected === apps.find((a) => a.name === "Second app")?.id,
        d1Row: notes.some((n) => n.text === "desktop-reopen"),
      };
      if (
        !result.d1Row ||
        !result.restoredSelection ||
        result.restoredMessages < 2
      )
        throw Error("Restore failed");
      await writeFile(
        join(base, ".spike/reopen.json"),
        JSON.stringify(result, null, 2),
      );
      console.log(JSON.stringify(result));
      app.quit();
    })().catch((e) => {
      console.error(e);
      finalExitCode = 1;
      app.quit();
    });
  }
});
app.on("before-quit", (e) => {
  if (process.argv.includes("--spike")) return;
  if (!quitting) {
    e.preventDefault();
    quitting = true;
    void stopLocal().finally(() => {
      bridge?.close();
      store?.close();
      app.exit(finalExitCode);
    });
  }
});
