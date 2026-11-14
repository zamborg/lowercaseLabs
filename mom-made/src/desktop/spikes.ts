import { app, BrowserWindow, WebContentsView } from "electron";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { createProject } from "../core/template";
import { Supervisor } from "../core/runtime";
export async function runSpikes(base: string) {
  const root = join(base, ".spike");
  await mkdir(root, { recursive: true });
  const results: Record<string, unknown> = {
    electron: process.versions.electron,
    node: process.versions.node,
  };
  const sup = new Supervisor(process.execPath, base, () => {});
  let win: BrowserWindow | undefined;
  try {
    const a = await createProject(root, base, "Integration spike");
    await sup.start(a);
    const r = sup.states[a.id];
    if (r.status !== "ready") throw Error(r.error + "\n" + r.logs);
    results.workerReady = true;
    const url = r.url!;
    await fetch(url + "/api/notes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "durable-spike" }),
    });
    await sup.stop(a.id);
    await sup.start(a);
    const notes = (await (
      await fetch(sup.states[a.id].url! + "/api/notes")
    ).json()) as { text: string }[];
    if (!notes.some((n) => n.text === "durable-spike"))
      throw Error("D1 lost data");
    results.d1Restart = true;
    win = new BrowserWindow({ show: true, width: 1000, height: 750 });
    const view = new WebContentsView({
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        partition: "persist:spike",
      },
    });
    win.contentView.addChildView(view);
    view.setBounds({ x: 0, y: 0, width: 1000, height: 750 });
    await view.webContents.loadURL(sup.states[a.id].url!);
    await new Promise((r) => setTimeout(r, 1000));
    const inspect = await view.webContents.executeJavaScript(
      '({heading:document.querySelector("h1").textContent,node:typeof require})',
    );
    if (inspect.node !== "undefined") throw Error("Node leaked into preview");
    results.inspection = inspect;
    await view.webContents.executeJavaScript(
      'document.querySelector("input").value="same-session"',
    );
    const capture = await view.webContents.capturePage();
    await writeFile(join(root, "preview.png"), capture.toPNG());
    view.setVisible(false);
    await win.loadURL(
      'data:text/html,<canvas id="mark" width="1000" height="750"></canvas>',
    );
    await win.webContents.executeJavaScript(
      `(()=>{const c=document.querySelector('canvas');const x=c.getContext('2d');x.strokeStyle='red';x.lineWidth=4;x.strokeRect(20,20,150,70);return c.toDataURL()})()`,
    );
    results.annotationOverlay = true;
    view.setVisible(true);
    const value = await view.webContents.executeJavaScript(
      'document.querySelector("input").value',
    );
    if (value !== "same-session") throw Error("View lost state");
    results.captureAndRestore = true;
    view.webContents.focus();
    await view.webContents.executeJavaScript(
      'document.querySelector("input").focus()',
    );
    await view.webContents.insertText(" typed");
    results.realInput = await view.webContents.executeJavaScript(
      'document.querySelector("input").value',
    );
    results.browserSecurity = true;
    results.ok = true;
  } catch (e) {
    results.error = String(e);
    results.ok = false;
  } finally {
    await sup.stopAll();
    win?.destroy();
    await writeFile(
      join(root, "results.json"),
      JSON.stringify(results, null, 2),
    );
    console.log(JSON.stringify(results, null, 2));
    if (!results.ok) process.exitCode = 1;
  }
}
