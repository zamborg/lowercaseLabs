import { Codex } from "../src/core/codex";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
const c = new Codex(),
  root = await mkdtemp(join(tmpdir(), "mom-codex-"));
let deltas = 0,
  approvals = 0,
  completions: string[] = [];
let resolveTurn: ((s: string) => void) | undefined;
c.on("notification", (m) => {
  if (m.method === "item/agentMessage/delta") deltas++;
  if (m.method === "turn/completed") {
    completions.push(m.params.turn.status);
    resolveTurn?.(m.params.turn.status);
  }
});
c.on("request", (m) => {
  if (m.method.endsWith("requestApproval")) {
    approvals++;
    c.answer(m.id, { decision: "decline" });
  } else c.reject(m.id);
});
c.on("failure", () => {});
function completion() {
  return new Promise<string>((resolve, reject) => {
    resolveTurn = resolve;
    const t = setTimeout(() => reject(Error("Turn timed out")), 90000);
    t.unref();
  });
}
const result: Record<string, unknown> = {};
try {
  await c.open();
  const account = await c.request("account/read", {});
  result.authenticated = !!account.account;
  const m = await c.request("model/list", {});
  const model = (m.data.find((x: any) => x.isDefault) ?? m.data[0]).model;
  const started = await c.request("thread/start", {
    cwd: root,
    sandbox: "read-only",
    approvalPolicy: "untrusted",
    model,
  });
  const threadId = started.thread.id;
  let done = completion();
  await c.request("turn/start", {
    threadId,
    input: [{ type: "text", text: "Reply exactly READY. Do not use tools." }],
  });
  result.firstTurn = await done;
  result.streamingDeltas = deltas;
  c.close();
  await c.open();
  await c.request("thread/resume", {
    threadId,
    cwd: root,
    sandbox: "read-only",
    approvalPolicy: "untrusted",
  });
  result.resumed = true;
  done = completion();
  await c.request("turn/start", {
    threadId,
    input: [
      {
        type: "text",
        text: "Use a shell command to write the text test to approval.txt in the working directory. Request approval if needed. If declined, stop.",
      },
    ],
  });
  result.approvalTurn = await done;
  result.approvals = approvals;
  done = completion();
  const turn = await c.request("turn/start", {
    threadId,
    input: [
      {
        type: "text",
        text: "Describe in detail how to build a compiler, with many examples. Do not use tools.",
      },
    ],
  });
  await new Promise((r) => setTimeout(r, 1000));
  await c.request("turn/interrupt", { threadId, turnId: turn.turn.id });
  result.interrupt = await done;
} catch (e) {
  result.error = String(e);
  process.exitCode = 1;
} finally {
  c.close();
  await writeFile(
    join(process.cwd(), ".spike/codex.json"),
    JSON.stringify(result, null, 2),
  );
  await rm(root, { recursive: true, force: true });
  console.log(JSON.stringify(result, null, 2));
}
