import path from "node:path";
import { parseBoolean } from "./logic.mjs";

export const AUTO_PURCHASE_ACK = "BUY_PASTRIES";
export const REHEARSAL_EVENT_URL =
  "https://www.hotplate.com/butterandcrumble/a7565fc1-3bcd-4541-9239-2b5c80508e66";

function list(value, fallback = []) {
  if (!value) return fallback;
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function number(value, fallback, name) {
  if (value == null || value === "") return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${name} must be a finite number.`);
  }
  return parsed;
}

function cliArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) continue;

    const equals = token.indexOf("=");
    if (equals >= 0) {
      parsed[token.slice(2, equals)] = token.slice(equals + 1);
      continue;
    }

    const name = token.slice(2);
    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      parsed[name] = next;
      index += 1;
    } else {
      parsed[name] = "true";
    }
  }
  return parsed;
}

export function loadConfig(command, argv = []) {
  const cli = cliArgs(argv);
  const env = process.env;
  const mode = cli.mode ?? env.PIA_MODE ?? "assist";

  if (!["dry-run", "assist", "auto"].includes(mode)) {
    throw new Error("PIA_MODE must be dry-run, assist, or auto.");
  }

  const pollMs = number(
    cli["poll-ms"] ?? env.PIA_POLL_MS,
    2_500,
    "PIA_POLL_MS",
  );
  if (pollMs < 1_500) {
    throw new Error("PIA_POLL_MS cannot be lower than 1500ms.");
  }

  const quantity = number(
    cli.quantity ?? env.PIA_QUANTITY,
    1,
    "PIA_QUANTITY",
  );
  if (!Number.isInteger(quantity) || quantity < 1 || quantity > 4) {
    throw new Error("PIA_QUANTITY must be an integer from 1 to 4.");
  }

  const releaseAtRaw = cli["release-at"] ?? env.PIA_RELEASE_AT ?? "";
  const releaseAt = releaseAtRaw ? new Date(releaseAtRaw) : null;
  if (releaseAt && Number.isNaN(releaseAt.valueOf())) {
    throw new Error("PIA_RELEASE_AT must be an ISO-8601 timestamp.");
  }

  const maxTotalDollars = number(
    cli["max-total"] ?? env.PIA_MAX_TOTAL_DOLLARS,
    40,
    "PIA_MAX_TOTAL_DOLLARS",
  );
  if (maxTotalDollars <= 0) {
    throw new Error("PIA_MAX_TOTAL_DOLLARS must be positive.");
  }

  const requestedEventUrl = cli["event-url"] ?? env.PIA_EVENT_URL ?? "";

  return {
    command,
    storeUrl:
      cli["store-url"] ??
      env.PIA_STORE_URL ??
      "https://www.hotplate.com/butterandcrumble",
    eventUrl:
      command === "rehearse"
        ? requestedEventUrl || REHEARSAL_EVENT_URL
        : requestedEventUrl || null,
    products: list(cli.products ?? env.PIA_PRODUCTS, ["Small Pastry Box"]),
    quantity,
    pickupPreferences: list(
      cli["pickup-preferences"] ?? env.PIA_PICKUP_PREFERENCES,
      ["Thursday", "Thu", "Friday", "Fri", "Saturday", "Sat", "Sunday", "Sun"],
    ),
    pickupTimePreferences: list(
      cli["pickup-time-preferences"] ?? env.PIA_PICKUP_TIME_PREFERENCES,
    ),
    randomPickup: parseBoolean(
      cli["random-pickup"] ?? env.PIA_RANDOM_PICKUP,
      true,
    ),
    mode: command === "rehearse" ? "dry-run" : mode,
    maxTotalDollars,
    autoPurchaseAcknowledgement: env.PIA_AUTO_PURCHASE_ACK ?? "",
    releaseAt,
    leadSeconds: number(
      env.PIA_LEAD_SECONDS,
      45,
      "PIA_LEAD_SECONDS",
    ),
    pollMs,
    pollJitterMs: number(
      env.PIA_POLL_JITTER_MS,
      350,
      "PIA_POLL_JITTER_MS",
    ),
    browserChannel:
      cli.channel ?? env.PIA_BROWSER_CHANNEL ?? "chromium",
    headless: parseBoolean(cli.headless ?? env.PIA_HEADLESS, false),
    profileDir: path.resolve(
      cli.profile ?? env.PIA_PROFILE_DIR ?? ".pia-profile",
    ),
    artifactDir: path.resolve(
      cli.artifacts ?? env.PIA_ARTIFACT_DIR ?? "artifacts",
    ),
  };
}
