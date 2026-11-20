#!/usr/bin/env node

import { loadConfig, AUTO_PURCHASE_ACK } from "./config.mjs";
import { loadDotEnv } from "./env.mjs";
import { runDrop, runSetup } from "./automation.mjs";

const command = process.argv[2] ?? "help";
const argv = process.argv.slice(3);

function help() {
  console.log(`
pia-pastry

  npm run setup      Prepare a dedicated persistent Hotplate browser profile
  npm run rehearse   Validate selectors against a pinned sold-out July drop
  npm run watch      Poll for the next drop and run the guarded checkout flow
  npm test           Run unit tests

Useful overrides:

  npm run watch -- --release-at=2026-08-04T12:00:00-07:00
  npm run watch -- --mode=assist
  npm run watch -- --mode=auto --max-total=40

Auto purchase additionally requires this exact .env value:

  PIA_AUTO_PURCHASE_ACK=${AUTO_PURCHASE_ACK}
`);
}

async function main() {
  loadDotEnv();
  if (command === "help" || command === "--help" || command === "-h") {
    help();
    return;
  }
  if (!["setup", "rehearse", "watch"].includes(command)) {
    throw new Error(`Unknown command "${command}".`);
  }

  const config = loadConfig(command, argv);
  if (command === "setup") {
    await runSetup(config);
    return;
  }
  const result = await runDrop(config);
  console.log(JSON.stringify({ result }, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exitCode = 1;
});
