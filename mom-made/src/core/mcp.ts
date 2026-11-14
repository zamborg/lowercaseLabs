// Tiny stdio MCP server. Codex launches it per thread; it forwards every
// tool call to the desktop over a private unix socket using a per-launch token.
import { createConnection } from "node:net";
import { TOOLS, type BridgeRequest, type ToolReply } from "./tools";
const socket = process.env.MOM_SOCKET ?? "",
  token = process.env.MOM_TOKEN ?? "",
  appId = process.env.MOM_APP ?? "";
function call(tool: string, args: Record<string, unknown>) {
  return new Promise<ToolReply>((resolve) => {
    const fail = (m: string) =>
      resolve({ content: [{ type: "text", text: m }], isError: true });
    const c = createConnection(socket);
    let buf = "";
    c.setTimeout(45000, () => {
      c.destroy();
      fail("mom-made did not answer in time.");
    });
    c.on("error", () => fail("mom-made is not reachable right now."));
    c.on("data", (d) => {
      buf += d;
      const i = buf.indexOf("\n");
      if (i >= 0) {
        try {
          resolve(JSON.parse(buf.slice(0, i)));
        } catch {
          fail("Unreadable reply from mom-made.");
        }
        c.end();
      }
    });
    const req: BridgeRequest = { token, appId, tool, args };
    c.write(JSON.stringify(req) + "\n");
  });
}
function send(m: unknown) {
  process.stdout.write(JSON.stringify(m) + "\n");
}
async function handle(m: any) {
  if (m.method === "initialize")
    return send({
      jsonrpc: "2.0",
      id: m.id,
      result: {
        protocolVersion: m.params?.protocolVersion ?? "2025-03-26",
        capabilities: { tools: {} },
        serverInfo: { name: "mom-made", version: "0.1.0" },
      },
    });
  if (m.method === "tools/list")
    return send({ jsonrpc: "2.0", id: m.id, result: { tools: TOOLS } });
  if (m.method === "tools/call") {
    const r = await call(m.params?.name, m.params?.arguments ?? {});
    return send({ jsonrpc: "2.0", id: m.id, result: r });
  }
  if (m.method === "ping")
    return send({ jsonrpc: "2.0", id: m.id, result: {} });
  if (m.id !== undefined && m.method)
    send({
      jsonrpc: "2.0",
      id: m.id,
      error: { code: -32601, message: "Method not found" },
    });
}
let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d: string) => {
  buf += d;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i);
    buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    try {
      void handle(JSON.parse(line));
    } catch {}
  }
});
process.stdin.on("end", () => process.exit(0));
