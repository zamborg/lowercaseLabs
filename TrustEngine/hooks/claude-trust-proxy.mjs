#!/usr/bin/env node

const endpoint = process.env.TRUST_ENGINE_ENDPOINT || "http://127.0.0.1:8787/api/hook";

async function main() {
  let raw = "";
  for await (const chunk of process.stdin) {
    raw += chunk.toString();
  }

  const payload = raw ? JSON.parse(raw) : {};

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      process.exit(0);
    }

    const body = await response.json();
    if (body?.hookResponse) {
      process.stdout.write(`${JSON.stringify(body.hookResponse)}\n`);
    }
  } catch {
    if (payload.hook_event_name === "PreToolUse" && payload.tool_name) {
      process.stdout.write(
        `${JSON.stringify({
          hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "ask",
            permissionDecisionReason: "trust engine unavailable: falling back to manual approval"
          }
        })}\n`
      );
    }
  }
}

main();
