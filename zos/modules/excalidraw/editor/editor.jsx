import React from "react";
import {createRoot} from "react-dom/client";
import {Excalidraw, MainMenu} from "@excalidraw/excalidraw";

const editor = document.querySelector("#editor");
const status = document.querySelector("#status");
const reload = document.querySelector("#reload");
reload.onclick = () => location.reload();
const api = editor.dataset.api;
const response = await fetch(api);
if (!response.ok) throw new Error("Could not open drawing");
const drawing = await response.json();
let version = drawing.version;
let saved = JSON.stringify(drawing.scene);
let pending = saved;
let busy = false;
let paused = false;
let timer;
let first = true;
const retry = document.querySelector("#retry");
retry.onclick = () => { retry.hidden = true; save(); };
window.addEventListener("beforeunload", event => {
  if (pending !== saved) { event.preventDefault(); event.returnValue = ""; }
});

async function save() {
  if (busy || paused || pending === saved) return;
  busy = true;
  const snapshot = pending;
  status.textContent = "Saving…";
  try {
    const res = await fetch(api, {
      method: "POST", headers: {"Content-Type":"application/json", "X-CSRF-Token":editor.dataset.csrf},
      body: JSON.stringify({if_version:version, scene:JSON.parse(snapshot)})
    });
    const result = await res.json();
    if (res.status === 409) {
      paused = true;
      status.textContent = "Another editor saved version " + result.current_version + ". Your edits are still on this canvas.";
      reload.hidden = false;
      return;
    }
    if (!res.ok) throw new Error(result.error || "Save failed");
    retry.hidden = true;
    saved = snapshot;
    version = result.version;
    status.textContent = "Saved";
  } catch (err) {
    status.textContent = err.message;
    retry.hidden = false;
    return;
  } finally { busy = false; }
  if (pending !== saved) void save();
}

function changed(elements, state, files) {
  const appState = {};
  for (const key of ["viewBackgroundColor", "gridSize", "gridStep", "gridModeEnabled"]) appState[key] = state[key];
  const next = JSON.stringify({elements, appState, files});
  if (first) { first = false; pending = saved = next; return; }
  if (pending === next) return;
  pending = next;
  if (!paused) {
    clearTimeout(timer);
    if (!busy && pending === saved) {
      status.textContent = "Saved";
      retry.hidden = true;
      return;
    }
    status.textContent = "Unsaved changes";
    timer = setTimeout(save, 450);
  }
}
createRoot(editor).render(
  <Excalidraw initialData={{...drawing.scene, scrollToContent:true}} onChange={changed}
    validateEmbeddable={false} autoFocus UIOptions={{canvasActions:{loadScene:false, saveToActiveFile:false,
    export:false, saveAsImage:false}, tools:{image:true}}}>
    <MainMenu><MainMenu.DefaultItems.ClearCanvas/></MainMenu>
  </Excalidraw>
);
status.textContent = "Saved";
