import fs from "node:fs";
import path from "node:path";

function unquote(value) {
  if (
    value.length >= 2 &&
    ((value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'")))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

export function loadDotEnv(file = path.resolve(".env")) {
  if (!fs.existsSync(file)) return;

  const source = fs.readFileSync(file, "utf8");
  for (const rawLine of source.split(/\r?\n/u)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const separator = line.indexOf("=");
    if (separator < 1) continue;

    const key = line.slice(0, separator).trim();
    const value = unquote(line.slice(separator + 1).trim());
    if (!(key in process.env)) process.env[key] = value;
  }
}
