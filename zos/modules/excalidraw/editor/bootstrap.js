const assets = new URL("./", import.meta.url);
window.EXCALIDRAW_ASSET_PATH = assets.href;
import(new URL("editor.js", assets).href).catch(() => {
  document.querySelector("#status").textContent = "Could not load the editor. Reload to try again.";
});
