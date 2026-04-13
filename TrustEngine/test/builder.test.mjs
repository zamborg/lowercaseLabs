import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { rebuildTrustArtifacts } from "../src/lib/trust-builder.mjs";

test("rebuildTrustArtifacts promotes repeated low-risk executions into allow_always suggestions", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "trust-engine-"));
  const config = {
    rootDir: tempDir,
    dataDir: tempDir,
    eventsPath: path.join(tempDir, "events.jsonl"),
    rulesPath: path.join(tempDir, "manual-rules.json"),
    policyPath: path.join(tempDir, "compiled-policy.json"),
    skillPath: path.join(tempDir, "trust-skill.md"),
    zeroTrust: true,
    claudeBuilderEnabled: false
  };

  const event = {
    toolName: "Read",
    fingerprint: "abc",
    summary: "README.md",
    risk: "low"
  };

  const lines = [
    { ...event, hookEventName: "PreToolUse", timestamp: "2026-03-21T00:00:00.000Z" },
    { ...event, hookEventName: "PermissionRequest", timestamp: "2026-03-21T00:00:01.000Z" },
    { ...event, hookEventName: "PostToolUse", timestamp: "2026-03-21T00:00:02.000Z" },
    { ...event, hookEventName: "PreToolUse", timestamp: "2026-03-21T00:01:00.000Z" },
    { ...event, hookEventName: "PermissionRequest", timestamp: "2026-03-21T00:01:01.000Z" },
    { ...event, hookEventName: "PostToolUse", timestamp: "2026-03-21T00:01:02.000Z" },
    { ...event, hookEventName: "PreToolUse", timestamp: "2026-03-21T00:02:00.000Z" },
    { ...event, hookEventName: "PermissionRequest", timestamp: "2026-03-21T00:02:01.000Z" },
    { ...event, hookEventName: "PostToolUse", timestamp: "2026-03-21T00:02:02.000Z" }
  ];

  await fs.writeFile(config.eventsPath, `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`, "utf8");
  await fs.writeFile(config.rulesPath, JSON.stringify({ version: 1, updatedAt: null, rules: [] }), "utf8");

  const result = await rebuildTrustArtifacts(config);
  const promoted = result.compiledPolicy.suggestedRules.find((rule) => rule.fingerprint === "abc");

  assert.equal(promoted.decision, "allow_always");
  assert.equal(promoted.active, true);
});
