import {
  app,
  BrowserWindow,
  WebContentsView,
  ipcMain,
  Menu,
  Tray,
  nativeImage,
  shell,
  dialog,
} from "electron";
import { join, resolve } from "node:path";
import { mkdir, writeFile } from "node:fs/promises";
import { Store } from "../core/store";
import { Supervisor } from "../core/runtime";
import { createProject } from "../core/template";
import { Codex } from "../core/codex";
import {
  Command,
  type State,
  type AppRecord,
  type Approval,
} from "../core/contracts";
import { runSpikes } from "./spikes";
const base = resolve(__dirname, "..");
let win: BrowserWindow, store: Store, supervisor: Supervisor, tray: Tray;
const codex = new Codex();
let finalExitCode = 0;
let quitting = false,
  closed = false,
  connected = false;
let selected: string | undefined;
let account = "Not connected",
  models: State["models"] = [],
  error: string | undefined;
let busy: State["busy"];
let approval: Approval | undefined;
let assistantId: string | undefined;
let bounds = { x: 370, y: 130, width: 700, height: 600, visible: true };
const views = new Map<string, WebContentsView>();
let viewGeneration = new Map<string, number>();
function snapshot(): State {
  return {
    apps: store.apps(),
    selected,
    messages: store.messages(),
    runtimes: supervisor.states,
    busy,
    approval,
    account,
    models,
    error,
  };
}
function emit() {
  if (quitting) return;
  if (win && !win.isDestroyed()) win.webContents.send("state", snapshot());
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
    v.setVisible(visible);
    if (visible)
      v.setBounds({
        x: bounds.x,
        y: bounds.y,
        width: Math.min(bounds.width, win.getContentBounds().width - bounds.x),
        height: Math.min(
          bounds.height,
          win.getContentBounds().height - bounds.y,
        ),
      });
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
  }
  if (
    new URL(v.webContents.getURL() || url).origin !== new URL(url).origin ||
    !v.webContents.getURL()
  )
    await v.webContents.loadURL(url);
  renderView();
}
const appStarts = new Map<string, Promise<void>>();
function start(a: AppRecord): Promise<void> {
  const pending = appStarts.get(a.id);
  if (pending) return pending;
  const task = (async () => {
    try {
      await supervisor.start(a);
      store.saveApp(a);
      if (!closed && !quitting && supervisor.states[a.id]?.status === "ready")
        await preview(a);
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
async function connect() {
  await codex.open();
  const [a, m] = await Promise.all([
    codex.request("account/read", {}),
    codex.request("model/list", {}),
  ]);
  account = a.account
    ? `${a.account.type}${a.account.email ? " · " + a.account.email : ""}`
    : "Login required";
  models = m.data.filter((m: any) => !m.hidden);
  connected = true;
  emit();
}
function finish(status: string) {
  if (busy) {
    store.message("activity", status, busy.appId, busy.turnId);
    store.event(busy.appId, "turn/status", { status, turnId: busy.turnId });
  }
  store.set("activeTurn", null);
  busy = undefined;
  approval = undefined;
  assistantId = undefined;
  emit();
}
codex.on("failure", (e: Error) => {
  connected = false;
  if (busy) finish("Interrupted: " + e.message);
});
codex.on("request", (m: any) => {
  if (!busy) {
    codex.reject(m.id);
    return;
  }
  if (
    m.method === "item/commandExecution/requestApproval" ||
    m.method === "item/fileChange/requestApproval"
  ) {
    approval = { id: m.id, method: m.method, params: m.params };
    emit();
  } else {
    codex.reject(m.id);
    store.message(
      "activity",
      "Unsupported agent request: " + m.method,
      busy.appId,
    );
    emit();
  }
});
codex.on("notification", (m: any) => {
  if (m.method === "account/login/completed") {
    void connect().catch((e) => {
      error = String(e);
      emit();
    });
    return;
  }
  if (!busy) return;
  const p = m.params ?? {};
  const a = requireApp(busy.appId);
  if (p.threadId && a.threadId && p.threadId !== a.threadId) return;
  if (m.method === "turn/started") busy.turnId = p.turn.id;
  if (m.method === "item/agentMessage/delta") {
    if (!assistantId)
      assistantId = store.message("assistant", "", a.id, busy.turnId).id;
    store.append(assistantId, p.delta);
  }
  if (m.method === "item/started" && p.item?.type !== "agentMessage") {
    store.message(
      "activity",
      p.item?.type === "commandExecution"
        ? p.item.command
        : (p.item?.type ?? "Working"),
      a.id,
      busy.turnId,
    );
  }
  if (m.method === "error")
    store.message(
      "error",
      p.error?.message ?? "Agent error",
      a.id,
      busy.turnId,
    );
  if (m.method === "turn/completed") {
    finish(
      p.turn.status +
        (p.turn.error?.message ? ": " + p.turn.error.message : ""),
    );
    return;
  }
  emit();
});
async function send(a: AppRecord, text: string, model: string, effort: string) {
  if (busy) throw Error("Finish or stop the current turn first.");
  busy = { appId: a.id };
  error = undefined;
  assistantId = undefined;
  store.message("user", text, a.id);
  emit();
  try {
    if (!connected) await connect();
    const params = {
      cwd: a.path,
      approvalPolicy: "on-request",
      sandbox: "workspace-write",
      model: model || undefined,
      developerInstructions:
        "You are building an app inside mom-made. Work only in the specified app directory. The desktop supervises the dev server. Do not start servers or deploy. Use React, Vite, Hono and Cloudflare bindings. No subagents.",
    };
    if (a.threadId)
      await codex.request("thread/resume", { ...params, threadId: a.threadId });
    else {
      const r = await codex.request("thread/start", params);
      a.threadId = r.thread.id;
      store.saveApp(a);
    }
    const r = await codex.request("turn/start", {
      threadId: a.threadId,
      input: [{ type: "text", text }],
      cwd: a.path,
      model: model || undefined,
      effort: effort || undefined,
    });
    if (busy) busy.turnId = r.turn.id;
    if (busy) store.set("activeTurn", { appId: a.id, turnId: busy.turnId });
    store.event(a.id, "turn/submitted", {
      threadId: a.threadId,
      text,
      model,
      effort,
    });
    emit();
  } catch (e) {
    finish("Failed: " + String(e));
    throw e;
  }
}
async function interrupt() {
  if (busy?.turnId) {
    const a = requireApp(busy.appId);
    await codex.request("turn/interrupt", {
      threadId: a.threadId,
      turnId: busy.turnId,
    });
  } else if (busy) {
    codex.close();
    finish("interrupted");
  }
}
let stopping: Promise<void> | undefined;
function stopLocal(): Promise<void> {
  if (stopping) return stopping;
  stopping = (async () => {
    await interrupt().catch(() => {});
    if (busy) finish("interrupted");
    codex.close();
    connected = false;
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
  selected = store.get("selected", undefined);
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
          selected = requireApp(c.appId).id;
          store.set("selected", selected);
          renderView();
          break;
        case "start": {
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
          const a = requireApp(c.appId);
          a.open = false;
          store.saveApp(a);
          await supervisor.stop(a.id);
          views.get(a.id)?.webContents.close();
          views.delete(a.id);
          if (selected === a.id)
            selected = store.apps().find((a) => a.open)?.id;
          store.set("selected", selected ?? null);
          renderView();
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
        case "bounds":
          bounds = c;
          renderView();
          break;
        case "connect":
          await connect();
          break;
        case "login":
          await codex.open();
          {
            const r = await codex.request("account/login/start", {
              type: "chatgpt",
            });
            if (
              typeof r.authUrl === "string" &&
              new URL(r.authUrl).protocol === "https:"
            )
              await shell.openExternal(r.authUrl);
          }
          break;
        case "send":
          await send(requireApp(c.appId), c.text, c.model, c.effort);
          break;
        case "interrupt":
          await interrupt();
          break;
        case "compact": {
          if (busy) throw Error("Wait for the active turn");
          const a = selected && requireApp(selected);
          if (a && a.threadId) {
            await codex.open();
            await codex.request("thread/resume", {
              threadId: a.threadId,
              cwd: a.path,
            });
            await codex.request("thread/compact/start", {
              threadId: a.threadId,
            });
            store.message("activity", "Compaction requested", a.id);
          }
          break;
        }
        case "answer":
          if (!approval || approval.id !== c.id)
            throw Error("Approval is no longer pending");
          codex.answer(c.id, { decision: c.accept ? "accept" : "decline" });
          approval = undefined;
          break;
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
      const model = models.find((m) => m.isDefault) ?? models[0];
      await send(
        a,
        "Edit src/main.tsx only: change the h1 to the exact text Made with mom. Preserve the rest of the application. Do not use subagents or start servers.",
        model.model,
        model.defaultReasoningEffort,
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
      const result = {
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
      store?.close();
      app.exit(finalExitCode);
    });
  }
});
