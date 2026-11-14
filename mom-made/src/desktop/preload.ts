import { contextBridge, ipcRenderer } from "electron";
contextBridge.exposeInMainWorld("mom", {
  command: (c: unknown) => ipcRenderer.invoke("command", c),
  state: () => ipcRenderer.invoke("state"),
  subscribe: (cb: (s: unknown) => void) => {
    const handler = (_: unknown, s: unknown) => cb(s);
    ipcRenderer.on("state", handler);
    return () => ipcRenderer.removeListener("state", handler);
  },
});
