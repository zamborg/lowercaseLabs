import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { getConfig } from "./lib/config.mjs";
import { buildManualRule, deleteRule, evaluateHook, loadManualRules, saveManualRules, upsertRule } from "./lib/policy-engine.mjs";
import { appendJsonl, ensureDir } from "./lib/storage.mjs";
import { rebuildTrustArtifacts, loadCurrentState } from "./lib/trust-builder.mjs";
import { buildEventRecord } from "./lib/trust.mjs";

const config = getConfig();

async function readBody(request) {
  let raw = "";
  for await (const chunk of request) {
    raw += chunk.toString();
  }

  return raw ? JSON.parse(raw) : {};
}

function json(response, statusCode, body) {
  response.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store"
  });
  response.end(JSON.stringify(body));
}

function sendStatic(response, body, contentType) {
  response.writeHead(200, {
    "content-type": contentType,
    "cache-control": "no-store"
  });
  response.end(body);
}

async function serveIndex(response) {
  const filePath = path.join(config.rootDir, "public", "index.html");
  const body = await fs.readFile(filePath, "utf8");
  sendStatic(response, body, "text/html; charset=utf-8");
}

async function handleHook(request, response) {
  const payload = await readBody(request);
  const decision = await evaluateHook(payload, config);
  const eventRecord = buildEventRecord(payload);

  await appendJsonl(config.eventsPath, {
    ...eventRecord,
    matchedRuleId: decision.matchedRule?.id || null,
    hookResponse: decision.hookResponse || null
  });

  if (payload.hook_event_name === "SessionEnd" || payload.hook_event_name === "PostToolUse") {
    await rebuildTrustArtifacts(config);
  }

  json(response, 200, {
    ok: true,
    hookResponse: decision.hookResponse
  });
}

async function handleManualRule(request, response) {
  const body = await readBody(request);
  const ruleSet = await loadManualRules(config);
  const existingRule =
    body.id ? ruleSet.rules.find((rule) => rule.id === body.id) : ruleSet.rules.find(
      (rule) => rule.toolName === body.toolName && rule.fingerprint === body.fingerprint
    );

  const nextRule = buildManualRule(
    {
      id: body.id,
      toolName: body.toolName,
      fingerprint: body.fingerprint,
      summary: body.summary,
      decision: body.decision,
      notes: body.notes,
      active: body.active,
      remainingUses: body.remainingUses,
      source: "manual"
    },
    existingRule
  );

  const nextRuleSet = upsertRule(ruleSet, nextRule);

  await saveManualRules(config, nextRuleSet);
  const state = await rebuildTrustArtifacts(config);

  json(response, 200, {
    ok: true,
    manualRules: nextRuleSet.rules,
    compiledPolicy: state.compiledPolicy
  });
}

async function handleDeleteManualRule(response, ruleId) {
  const ruleSet = await loadManualRules(config);
  const nextRuleSet = deleteRule(ruleSet, ruleId);
  await saveManualRules(config, nextRuleSet);
  const state = await rebuildTrustArtifacts(config);

  json(response, 200, {
    ok: true,
    manualRules: nextRuleSet.rules,
    compiledPolicy: state.compiledPolicy
  });
}

async function handleState(response) {
  const state = await loadCurrentState(config);
  const recentEvents = state.events.slice(-200).reverse();

  json(response, 200, {
    ok: true,
    summary: {
      events: state.events.length,
      manualRules: state.manualRules.length,
      suggestedRules: state.compiledPolicy.suggestedRules.length,
      generatedAt: state.compiledPolicy.generatedAt,
      zeroTrust: config.zeroTrust
    },
    manualRules: state.manualRules,
    suggestedRules: state.compiledPolicy.suggestedRules,
    skillMarkdown: state.skillMarkdown,
    recentEvents
  });
}

async function handleRebuild(response) {
  const state = await rebuildTrustArtifacts(config);
  json(response, 200, {
    ok: true,
    compiledPolicy: state.compiledPolicy,
    skillMarkdown: state.skillMarkdown
  });
}

await ensureDir(config.dataDir);
await rebuildTrustArtifacts(config);

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${config.host}:${config.port}`);

    if (request.method === "GET" && url.pathname === "/") {
      await serveIndex(response);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/state") {
      await handleState(response);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/hook") {
      await handleHook(request, response);
      return;
    }

    if ((request.method === "POST" || request.method === "PUT") && url.pathname === "/api/manual-rules") {
      await handleManualRule(request, response);
      return;
    }

    if (request.method === "DELETE" && url.pathname.startsWith("/api/manual-rules/")) {
      const ruleId = decodeURIComponent(url.pathname.replace("/api/manual-rules/", ""));
      await handleDeleteManualRule(response, ruleId);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/rebuild") {
      await handleRebuild(response);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/health") {
      json(response, 200, { ok: true });
      return;
    }

    json(response, 404, { ok: false, error: "not_found" });
  } catch (error) {
    json(response, 500, {
      ok: false,
      error: error.message
    });
  }
});

server.listen(config.port, config.host, () => {
  console.log(`Trust Engine listening on http://${config.host}:${config.port}`);
});
