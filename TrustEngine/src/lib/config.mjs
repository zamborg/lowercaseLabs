import path from "node:path";

export function getConfig() {
  const rootDir = process.cwd();
  const dataDir = process.env.TRUST_ENGINE_DATA_DIR || path.join(rootDir, ".trust-engine-data");

  return {
    rootDir,
    dataDir,
    eventsPath: path.join(dataDir, "events.jsonl"),
    rulesPath: path.join(dataDir, "manual-rules.json"),
    policyPath: path.join(dataDir, "compiled-policy.json"),
    skillPath: path.join(dataDir, "trust-skill.md"),
    port: Number.parseInt(process.env.PORT || "8787", 10),
    host: process.env.HOST || "127.0.0.1",
    zeroTrust: process.env.TRUST_ENGINE_ZERO_TRUST !== "0",
    claudeBuilderEnabled: process.env.TRUST_ENGINE_USE_CLAUDE === "1"
  };
}
