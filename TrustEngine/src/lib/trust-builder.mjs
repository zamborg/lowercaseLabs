import { spawn } from "node:child_process";
import { appendJsonl, readJson, readJsonl, readText, writeJson, writeText } from "./storage.mjs";

function groupEvents(events) {
  const groups = new Map();

  for (const event of events) {
    if (!event.toolName) {
      continue;
    }

    const key = `${event.toolName}:${event.fingerprint}`;
    const current = groups.get(key) || {
      key,
      toolName: event.toolName,
      fingerprint: event.fingerprint,
      summary: event.summary,
      risk: event.risk,
      attempts: 0,
      prompts: 0,
      executions: 0,
      failures: 0,
      lastSeenAt: event.timestamp
    };

    current.summary = event.summary;
    current.risk = event.risk;
    current.lastSeenAt = event.timestamp;

    if (event.hookEventName === "PreToolUse") {
      current.attempts += 1;
    }

    if (event.hookEventName === "PermissionRequest") {
      current.prompts += 1;
    }

    if (event.hookEventName === "PostToolUse") {
      current.executions += 1;
    }

    if (event.hookEventName === "PostToolUseFailure") {
      current.executions += 1;
      current.failures += 1;
    }

    groups.set(key, current);
  }

  return [...groups.values()].map((group) => ({
    ...group,
    likelyDeniedOrAbandoned: Math.max(group.prompts - group.executions, 0)
  }));
}

function chooseSuggestedDecision(group) {
  if (group.risk === "high") {
    if (group.executions >= 2) {
      return "allow_more_complex";
    }

    return "escalate";
  }

  if (group.risk === "medium") {
    if (group.executions >= 8 && group.likelyDeniedOrAbandoned === 0) {
      return "allow_always";
    }

    if (group.executions >= 2) {
      return "allow_more_complex";
    }

    return "escalate";
  }

  if (group.executions >= 3 && group.likelyDeniedOrAbandoned === 0) {
    return "allow_always";
  }

  if (group.executions >= 1) {
    return "allow_more_complex";
  }

  return "escalate";
}

function buildSuggestedRules(groups, manualRuleSet) {
  const manualKeys = new Set(manualRuleSet.rules.map((rule) => `${rule.toolName}:${rule.fingerprint}`));

  return groups
    .filter((group) => !manualKeys.has(`${group.toolName}:${group.fingerprint}`))
    .map((group) => ({
      toolName: group.toolName,
      fingerprint: group.fingerprint,
      summary: group.summary,
      risk: group.risk,
      attempts: group.attempts,
      executions: group.executions,
      prompts: group.prompts,
      likelyDeniedOrAbandoned: group.likelyDeniedOrAbandoned,
      decision: chooseSuggestedDecision(group),
      active: chooseSuggestedDecision(group) === "allow_always"
    }))
    .sort((left, right) => right.executions - left.executions);
}

function lineForRule(rule) {
  return `- \`${rule.toolName}\` ${rule.summary}`;
}

function buildSkillMarkdown(manualRules, suggestedRules) {
  const alwaysAllow = [...manualRules, ...suggestedRules.filter((rule) => rule.active && rule.decision === "allow_always")];
  const alwaysEscalate = [
    ...manualRules.filter((rule) => rule.decision === "deny" || rule.decision === "escalate"),
    ...suggestedRules.filter((rule) => rule.decision === "escalate")
  ];
  const conditional = [
    ...manualRules.filter((rule) => rule.decision === "allow_more_complex"),
    ...suggestedRules.filter((rule) => rule.decision === "allow_more_complex")
  ];

  return `# Trust Skill

You are operating behind a user-specific trust proxy. Treat these rules as the current trust posture, not as permission to improvise beyond them.

## Usually Allow
${alwaysAllow.length ? alwaysAllow.map(lineForRule).join("\n") : "- Nothing promoted yet. Default to asking."}

## Always Escalate
${alwaysEscalate.length ? alwaysEscalate.map(lineForRule).join("\n") : "- No explicit hard blocks yet."}

## Conditional Patterns
${conditional.length ? conditional.map(lineForRule).join("\n") : "- No conditional patterns compiled yet."}

## Operating Rules
- If an action is not clearly covered by "Usually Allow", ask.
- High-risk communication and external side effects require escalation unless explicitly trusted.
- Repeated user approvals can justify suggesting an allow rule, but not silently broadening the action class.
- Prefer semantically meaningful tools over generic shell commands when handling email, messaging, calendar, or other life-admin workflows.
`;
}

function renderModelPrompt(groups, manualRules, suggestedRules) {
  return `
You are compiling a trust skill for Claude Code. Keep the output concise Markdown.

Manual rules:
${JSON.stringify(manualRules, null, 2)}

Suggested rules:
${JSON.stringify(suggestedRules.slice(0, 50), null, 2)}

Telemetry summary:
${JSON.stringify(groups.slice(0, 50), null, 2)}

Write a skill with sections:
- Usually Allow
- Always Escalate
- Conditional Patterns
- Operating Rules
`;
}

async function maybeRewriteWithClaude(config, groups, manualRules, suggestedRules, fallbackMarkdown) {
  if (!config.claudeBuilderEnabled) {
    return fallbackMarkdown;
  }

  const prompt = renderModelPrompt(groups, manualRules, suggestedRules);

  try {
    const output = await new Promise((resolve, reject) => {
      const child = spawn("claude", ["-p", prompt], {
        cwd: config.rootDir,
        stdio: ["ignore", "pipe", "pipe"]
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });

      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });

      child.on("error", reject);
      child.on("close", (code) => {
        if (code === 0 && stdout.trim()) {
          resolve(stdout.trim());
          return;
        }

        reject(new Error(stderr.trim() || `claude exited with code ${code}`));
      });
    });

    return output;
  } catch {
    return fallbackMarkdown;
  }
}

export async function rebuildTrustArtifacts(config) {
  const [events, manualRuleSet] = await Promise.all([
    readJsonl(config.eventsPath),
    readJson(config.rulesPath, { version: 1, updatedAt: null, rules: [] })
  ]);

  const groups = groupEvents(events);
  const suggestedRules = buildSuggestedRules(groups, manualRuleSet);
  const heuristicMarkdown = buildSkillMarkdown(manualRuleSet.rules, suggestedRules);
  const skillMarkdown = await maybeRewriteWithClaude(
    config,
    groups,
    manualRuleSet.rules,
    suggestedRules,
    heuristicMarkdown
  );

  const compiledPolicy = {
    version: 1,
    generatedAt: new Date().toISOString(),
    zeroTrust: config.zeroTrust,
    totals: {
      events: events.length,
      manualRules: manualRuleSet.rules.length,
      suggestedRules: suggestedRules.length
    },
    suggestedRules
  };

  await Promise.all([
    writeJson(config.policyPath, compiledPolicy),
    writeText(config.skillPath, skillMarkdown)
  ]);

  return {
    compiledPolicy,
    skillMarkdown
  };
}

export async function loadCurrentState(config) {
  const [events, manualRuleSet, compiledPolicy, skillMarkdown] = await Promise.all([
    readJsonl(config.eventsPath),
    readJson(config.rulesPath, { version: 1, updatedAt: null, rules: [] }),
    readJson(config.policyPath, {
      version: 1,
      generatedAt: null,
      zeroTrust: config.zeroTrust,
      totals: { events: 0, manualRules: 0, suggestedRules: 0 },
      suggestedRules: []
    }),
    readText(config.skillPath, "")
  ]);

  return {
    events,
    manualRules: manualRuleSet.rules,
    compiledPolicy,
    skillMarkdown
  };
}
