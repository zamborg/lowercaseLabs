import crypto from "node:crypto";
import { readJson, writeJson } from "./storage.mjs";
import { classifyPayload, prettyDecision, ruleMatches } from "./trust.mjs";

function emptyRuleSet() {
  return {
    version: 1,
    updatedAt: null,
    rules: []
  };
}

export async function loadManualRules(config) {
  const rules = await readJson(config.rulesPath, emptyRuleSet());
  if (!Array.isArray(rules.rules)) {
    return emptyRuleSet();
  }

  return rules;
}

export async function saveManualRules(config, rules) {
  const nextValue = {
    version: 1,
    updatedAt: new Date().toISOString(),
    rules: rules.rules
  };

  await writeJson(config.rulesPath, nextValue);
  return nextValue;
}

function normalizeRemainingUses(decision, explicitRemainingUses) {
  if (decision !== "allow_once") {
    return 0;
  }

  const parsed = Number.parseInt(String(explicitRemainingUses ?? "1"), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
}

export function upsertRule(ruleSet, nextRule) {
  const rules = [...ruleSet.rules];
  const existingIndex = rules.findIndex((rule) => {
    if (nextRule.id && rule.id === nextRule.id) {
      return true;
    }

    return rule.toolName === nextRule.toolName && rule.fingerprint === nextRule.fingerprint;
  });

  if (existingIndex >= 0) {
    rules[existingIndex] = {
      ...rules[existingIndex],
      ...nextRule,
      remainingUses: normalizeRemainingUses(nextRule.decision, nextRule.remainingUses),
      updatedAt: new Date().toISOString()
    };
  } else {
    rules.unshift({
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      remainingUses: normalizeRemainingUses(nextRule.decision, nextRule.remainingUses),
      ...nextRule
    });
  }

  return {
    version: 1,
    updatedAt: new Date().toISOString(),
    rules
  };
}

export function buildManualRule(input, existingRule = null) {
  return {
    id: input.id || existingRule?.id || crypto.randomUUID(),
    createdAt: existingRule?.createdAt || new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    toolName: input.toolName,
    fingerprint: input.fingerprint,
    summary: input.summary || existingRule?.summary || "",
    decision: input.decision || existingRule?.decision || "escalate",
    notes: input.notes ?? existingRule?.notes ?? "",
    active: input.active ?? existingRule?.active ?? true,
    remainingUses: normalizeRemainingUses(input.decision || existingRule?.decision, input.remainingUses),
    source: input.source || existingRule?.source || "manual"
  };
}

export function deleteRule(ruleSet, id) {
  return {
    version: 1,
    updatedAt: new Date().toISOString(),
    rules: ruleSet.rules.filter((rule) => rule.id !== id)
  };
}

export function findMatchingRule(ruleSet, classified) {
  return ruleSet.rules.find((rule) => {
    if (!rule.active) {
      return false;
    }

    return ruleMatches(rule, classified);
  });
}

function makePretToolDecision(decision, reason, additionalContext) {
  return {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: decision,
      permissionDecisionReason: reason,
      ...(additionalContext ? { additionalContext } : {})
    }
  };
}

function makePermissionRequestAllow() {
  return {
    hookSpecificOutput: {
      hookEventName: "PermissionRequest",
      decision: {
        behavior: "allow"
      }
    }
  };
}

function makePermissionRequestDeny(reason) {
  return {
    hookSpecificOutput: {
      hookEventName: "PermissionRequest",
      decision: {
        behavior: "deny",
        message: reason,
        interrupt: false
      }
    }
  };
}

function buildReason(prefix, classified, rule) {
  const base = prefix || prettyDecision(rule?.decision || "escalate");
  return `${base}: ${classified.toolName} ${classified.summary}`.trim();
}

function decrementAllowOnce(ruleSet, rule) {
  if (rule.decision !== "allow_once") {
    return ruleSet;
  }

  const remainingUses = Math.max((rule.remainingUses || 1) - 1, 0);
  const nextDecision = remainingUses === 0 ? "escalate" : "allow_once";

  return {
    version: 1,
    updatedAt: new Date().toISOString(),
    rules: ruleSet.rules.map((entry) => {
      if (entry.id !== rule.id) {
        return entry;
      }

      return {
        ...entry,
        decision: nextDecision,
        remainingUses,
        updatedAt: new Date().toISOString()
      };
    })
  };
}

export async function evaluateHook(payload, config) {
  const eventName = payload.hook_event_name;
  const classified = classifyPayload(payload);
  const ruleSet = await loadManualRules(config);
  const rule = findMatchingRule(ruleSet, classified);
  let nextRuleSet = null;

  if (eventName === "PreToolUse") {
    if (rule?.decision === "deny") {
      return {
        classified,
        matchedRule: rule,
        hookResponse: makePretToolDecision("deny", buildReason(rule.notes, classified, rule))
      };
    }

    if (rule?.decision === "allow_always") {
      return {
        classified,
        matchedRule: rule,
        hookResponse: makePretToolDecision("allow", buildReason(rule.notes, classified, rule))
      };
    }

    if (rule?.decision === "allow_once" && (rule.remainingUses || 1) > 0) {
      nextRuleSet = decrementAllowOnce(ruleSet, rule);
      await saveManualRules(config, nextRuleSet);
      return {
        classified,
        matchedRule: rule,
        hookResponse: makePretToolDecision("allow", buildReason(rule.notes, classified, rule))
      };
    }

    if (rule?.decision === "allow_more_complex" || rule?.decision === "escalate") {
      return {
        classified,
        matchedRule: rule,
        hookResponse: makePretToolDecision("ask", buildReason(rule.notes, classified, rule))
      };
    }

    if (config.zeroTrust && payload.tool_name) {
      return {
        classified,
        matchedRule: null,
        hookResponse: makePretToolDecision(
          "ask",
          `zero-trust default: ${classified.toolName} ${classified.summary}`.trim()
        )
      };
    }

    return {
      classified,
      matchedRule: null,
      hookResponse: null
    };
  }

  if (eventName === "PermissionRequest") {
    if (rule?.decision === "allow_always") {
      return {
        classified,
        matchedRule: rule,
        hookResponse: makePermissionRequestAllow()
      };
    }

    if (rule?.decision === "allow_once" && (rule.remainingUses || 1) > 0) {
      nextRuleSet = decrementAllowOnce(ruleSet, rule);
      await saveManualRules(config, nextRuleSet);
      return {
        classified,
        matchedRule: rule,
        hookResponse: makePermissionRequestAllow()
      };
    }

    if (rule?.decision === "deny") {
      return {
        classified,
        matchedRule: rule,
        hookResponse: makePermissionRequestDeny(buildReason(rule.notes, classified, rule))
      };
    }

    return {
      classified,
      matchedRule: rule || null,
      hookResponse: null
    };
  }

  return {
    classified,
    matchedRule: rule || null,
    hookResponse: null
  };
}
