import { createServer, type Server } from "node:net";
import { mkdtempSync, rmSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomBytes, timingSafeEqual } from "node:crypto";
import { TOOLS, type BridgeRequest, type ToolReply } from "./tools";
export type ToolHandler = (
  appId: string,
  tool: string,
  args: Record<string, unknown>,
) => Promise<ToolReply>;
export class Bridge {
  readonly token = randomBytes(24).toString("hex");
  readonly dir = mkdtempSync(join(tmpdir(), "mm-"));
  readonly path = join(this.dir, "b.sock");
  private server: Server;
  constructor(handler: ToolHandler) {
    chmodSync(this.dir, 0o700);
    this.server = createServer((c) => {
      let buf = "";
      c.setEncoding("utf8");
      c.on("error", () => {});
      c.on("data", async (d: string) => {
        buf += d;
        if (buf.length > 64_000) return c.destroy();
        const i = buf.indexOf("\n");
        if (i < 0) return;
        const line = buf.slice(0, i);
        buf = "";
        const reply = (r: ToolReply) => c.end(JSON.stringify(r) + "\n");
        let req: BridgeRequest;
        try {
          req = JSON.parse(line);
        } catch {
          return reply(err("Malformed request."));
        }
        const a = Buffer.from(String(req.token)),
          b = Buffer.from(this.token);
        if (a.length !== b.length || !timingSafeEqual(a, b))
          return reply(err("Not authorized."));
        if (!TOOLS.some((t) => t.name === req.tool))
          return reply(err("Unknown tool."));
        try {
          reply(await handler(req.appId, req.tool, req.args ?? {}));
        } catch (e) {
          reply(err(e instanceof Error ? e.message : String(e)));
        }
      });
    });
  }
  listen() {
    return new Promise<void>((resolve, reject) => {
      this.server.once("error", reject);
      this.server.listen(this.path, () => {
        chmodSync(this.path, 0o600);
        resolve();
      });
    });
  }
  close() {
    this.server.close();
    try {
      rmSync(this.dir, { recursive: true });
    } catch {}
  }
}
const err = (text: string): ToolReply => ({
  content: [{ type: "text", text }],
  isError: true,
});
