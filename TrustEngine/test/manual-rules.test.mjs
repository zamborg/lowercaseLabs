import test from "node:test";
import assert from "node:assert/strict";
import { buildManualRule, deleteRule, upsertRule } from "../src/lib/policy-engine.mjs";

test("upsertRule updates an existing rule by id", () => {
  const original = {
    version: 1,
    updatedAt: null,
    rules: [
      {
        id: "rule-1",
        toolName: "Read",
        fingerprint: "abc",
        summary: "README.md",
        decision: "allow_always",
        notes: "",
        active: true,
        remainingUses: 0,
        source: "manual"
      }
    ]
  };

  const nextRule = buildManualRule(
    {
      id: "rule-1",
      toolName: "Read",
      fingerprint: "abc",
      summary: "README.md",
      decision: "deny",
      notes: "changed my mind",
      active: true
    },
    original.rules[0]
  );

  const updated = upsertRule(original, nextRule);
  assert.equal(updated.rules.length, 1);
  assert.equal(updated.rules[0].decision, "deny");
  assert.equal(updated.rules[0].notes, "changed my mind");
});

test("deleteRule removes a rule by id", () => {
  const original = {
    version: 1,
    updatedAt: null,
    rules: [
      { id: "rule-1", toolName: "Read", fingerprint: "abc" },
      { id: "rule-2", toolName: "Write", fingerprint: "def" }
    ]
  };

  const updated = deleteRule(original, "rule-1");
  assert.equal(updated.rules.length, 1);
  assert.equal(updated.rules[0].id, "rule-2");
});
