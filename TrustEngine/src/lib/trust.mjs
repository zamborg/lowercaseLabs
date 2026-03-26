import crypto from "node:crypto";
import path from "node:path";

function sortObject(value) {
  if (Array.isArray(value)) {
    return value.map(sortObject);
  }

  if (value && typeof value === "object") {
    return Object.keys(value)
      .sort()
      .reduce((accumulator, key) => {
        accumulator[key] = sortObject(value[key]);
        return accumulator;
      }, {});
  }

  return value;
}

function stableStringify(value) {
  return JSON.stringify(sortObject(value));
}

function clampText(value, maxLength = 200) {
  if (typeof value !== "string") {
    return value;
  }

  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}…` : value;
}

function normalizeScalar(value, cwd = "") {
  if (typeof value !== "string") {
    if (typeof value === "number") {
      return "<number>";
    }

    if (typeof value === "boolean" || value === null) {
      return value;
    }

    return value;
  }

  const trimmed = value.trim();

  if (!trimmed) {
    return "";
  }

  if (trimmed.startsWith(cwd)) {
    return `<workspace-path:${path.relative(cwd, trimmed) || "."}>`;
  }

  if (/^[0-9a-f]{8}-[0-9a-f-]{27,}$/i.test(trimmed)) {
    return "<uuid>";
  }

  if (/^[0-9a-f]{16,}$/i.test(trimmed)) {
    return "<hex-id>";
  }

  if (/^\+?[0-9][0-9()\-\s]{7,}$/.test(trimmed)) {
    return "<phone>";
  }

  const emailMatch = trimmed.match(/^[^@\s]+@([^@\s]+)$/);
  if (emailMatch) {
    return `<email:${emailMatch[1].toLowerCase()}>`;
  }

  if (/^https?:\/\//i.test(trimmed)) {
    try {
      const url = new URL(trimmed);
      return `<url:${url.hostname}${url.pathname}>`;
    } catch {
      return "<url>";
    }
  }

  if (/^\d+$/.test(trimmed)) {
    return "<number>";
  }

  if (trimmed.length > 100) {
    return `<long-string:${trimmed.slice(0, 32)}>`;
  }

  return trimmed;
}

function normalizeValue(value, cwd = "") {
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeValue(entry, cwd));
  }

  if (value && typeof value === "object") {
    return Object.keys(value)
      .sort()
      .reduce((accumulator, key) => {
        accumulator[key] = normalizeValue(value[key], cwd);
        return accumulator;
      }, {});
  }

  return normalizeScalar(value, cwd);
}

function normalizeToolInput(toolName, toolInput, cwd = "") {
  const normalized = normalizeValue(toolInput, cwd);

  if (!normalized || typeof normalized !== "object" || Array.isArray(normalized)) {
    return normalized;
  }

  if (["Write", "Edit", "MultiEdit"].includes(toolName)) {
    for (const key of ["content", "old_string", "new_string"]) {
      if (key in normalized) {
        normalized[key] = `<${key}>`;
      }
    }
  }

  if (toolName.startsWith("mcp__")) {
    for (const key of ["body", "message", "html", "text", "draft"]) {
      if (key in normalized) {
        normalized[key] = `<${key}>`;
      }
    }
  }

  return normalized;
}

function summarizeToolInput(toolName, toolInput = {}) {
  if (toolName === "Bash") {
    return clampText(toolInput.command || "");
  }

  for (const key of ["file_path", "path", "url", "query", "command", "to", "recipient", "message", "title"]) {
    if (toolInput[key]) {
      return clampText(String(toolInput[key]));
    }
  }

  return clampText(stableStringify(toolInput), 160);
}

function detectRisk(toolName, toolInput = {}, cwd = "") {
  const loweredTool = toolName.toLowerCase();

  if (["read", "ls", "glob", "grep"].includes(loweredTool)) {
    return "low";
  }

  if (["websearch"].includes(loweredTool)) {
    return "low";
  }

  if (["webfetch"].includes(loweredTool)) {
    return "medium";
  }

  if (["write", "edit", "multiedit"].includes(loweredTool)) {
    const target = toolInput.file_path || toolInput.path || "";
    if (typeof target === "string" && target.includes(".env")) {
      return "high";
    }

    if (cwd && typeof target === "string" && target.startsWith("/") && !target.startsWith(cwd)) {
      return "high";
    }

    return "medium";
  }

  if (toolName === "Bash") {
    const command = toolInput.command || "";
    const riskyFragments = [
      "rm ",
      "rm -",
      "curl ",
      "wget ",
      "scp ",
      "ssh ",
      "sudo ",
      "git push",
      "npm publish",
      "pnpm publish",
      "osascript",
      "sqlite3",
      "defaults write",
      "launchctl"
    ];

    return riskyFragments.some((fragment) => command.includes(fragment)) ? "high" : "medium";
  }

  if (toolName.startsWith("mcp__")) {
    if (/(mail|email|message|imessage|sms|calendar|payment|bank|wire|slack|discord|send)/i.test(toolName)) {
      return "high";
    }

    return "medium";
  }

  return "medium";
}

export function classifyPayload(payload) {
  const toolName = payload.tool_name || payload.mcp_server_name || payload.hook_event_name;
  const toolInput = payload.tool_input || payload.content || {};
  const cwd = payload.cwd || "";
  const normalizedInput = normalizeToolInput(toolName, toolInput, cwd);
  const fingerprintSource = `${toolName}:${stableStringify(normalizedInput)}`;
  const fingerprint = crypto.createHash("sha256").update(fingerprintSource).digest("hex").slice(0, 16);
  const summary = summarizeToolInput(toolName, toolInput);
  const risk = detectRisk(toolName, toolInput, cwd);

  return {
    toolName,
    toolInput,
    normalizedInput,
    fingerprint,
    summary,
    risk
  };
}

export function sanitizePayload(value, depth = 0) {
  if (depth > 5) {
    return "<max-depth>";
  }

  if (typeof value === "string") {
    return clampText(value, 500);
  }

  if (Array.isArray(value)) {
    return value.slice(0, 20).map((entry) => sanitizePayload(entry, depth + 1));
  }

  if (value && typeof value === "object") {
    return Object.keys(value)
      .slice(0, 50)
      .reduce((accumulator, key) => {
        accumulator[key] = sanitizePayload(value[key], depth + 1);
        return accumulator;
      }, {});
  }

  return value;
}

export function buildEventRecord(payload) {
  const classified = classifyPayload(payload);

  return {
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    hookEventName: payload.hook_event_name,
    sessionId: payload.session_id || null,
    cwd: payload.cwd || null,
    transcriptPath: payload.transcript_path || null,
    permissionMode: payload.permission_mode || null,
    toolName: classified.toolName || null,
    fingerprint: classified.fingerprint,
    summary: classified.summary,
    risk: classified.risk,
    payload: sanitizePayload(payload)
  };
}

export function ruleMatches(rule, classified) {
  return rule.toolName === classified.toolName && rule.fingerprint === classified.fingerprint;
}

export function prettyDecision(decision) {
  switch (decision) {
    case "allow_always":
      return "allow always";
    case "allow_once":
      return "allow once";
    case "allow_more_complex":
      return "allow-more-complex";
    default:
      return decision;
  }
}
