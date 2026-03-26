import test from "node:test";
import assert from "node:assert/strict";
import { classifyPayload } from "../src/lib/trust.mjs";

test("classifyPayload normalizes workspace paths into a stable fingerprint", () => {
  const cwd = "/Users/test/project";
  const first = classifyPayload({
    hook_event_name: "PreToolUse",
    cwd,
    tool_name: "Write",
    tool_input: {
      file_path: "/Users/test/project/src/app.js",
      content: "hello"
    }
  });

  const second = classifyPayload({
    hook_event_name: "PreToolUse",
    cwd,
    tool_name: "Write",
    tool_input: {
      file_path: "/Users/test/project/src/app.js",
      content: "goodbye"
    }
  });

  assert.equal(first.fingerprint, second.fingerprint);
  assert.equal(first.risk, "medium");
});

test("classifyPayload marks external communication MCP tools as high risk", () => {
  const result = classifyPayload({
    hook_event_name: "PreToolUse",
    cwd: "/tmp/project",
    tool_name: "mcp__mail__send_email",
    tool_input: {
      to: "griffin@example.com",
      subject: "status update"
    }
  });

  assert.equal(result.risk, "high");
});
