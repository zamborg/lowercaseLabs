import { readFileSync, writeFileSync, existsSync } from "fs";
import { basename } from "path";

export const SUPPORTED_MODELS = [
  "claude-opus-4-6",
  "claude-sonnet-4-6",
  "claude-haiku-4-5-20251001",
] as const;

export type ModelId = typeof SUPPORTED_MODELS[number];

export const DEFAULT_MODEL: ModelId = "claude-sonnet-4-6";
export const DEFAULT_VARIANT = "claude-code";

export interface AgentConfig {
  file: string;          // filename e.g. "scout.md"
  name: string;          // display name from frontmatter
  variant: string;       // runner type, defaults to "claude-code"
  model: ModelId;        // model id from frontmatter
  system_prompt: string; // filename in prompts/ dir, defaults to "default.md"
  soul: string;          // personality prompt (body below frontmatter)
}

// Parse frontmatter fields from --- block
function parseFrontmatter(content: string): Record<string, string> {
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return {};
  const fields: Record<string, string> = {};
  for (const line of match[1].split("\n")) {
    const kv = line.match(/^(\w+):\s*(.+)$/);
    if (kv) fields[kv[1]] = kv[2].trim();
  }
  return fields;
}

// Extract body below the frontmatter block
function parseSoul(content: string): string {
  return content.replace(/^---\n[\s\S]*?\n---\n?/, "").trim();
}

export function parseAgentConfig(filePath: string): AgentConfig {
  const content = existsSync(filePath) ? readFileSync(filePath, "utf-8") : "";
  const fm = parseFrontmatter(content);
  const file = basename(filePath);
  const name = fm.name || file.replace(".md", "");
  const model = (SUPPORTED_MODELS as readonly string[]).includes(fm.model)
    ? (fm.model as ModelId)
    : DEFAULT_MODEL;
  const variant = fm.variant || DEFAULT_VARIANT;
  const system_prompt = fm.system_prompt || "default.md";
  const soul = parseSoul(content);
  return { file, name, variant, model, system_prompt, soul };
}

// Serialize an AgentConfig back to a .md file
export function serializeAgentConfig(config: AgentConfig, extraBody?: string): string {
  const body = extraBody ?? config.soul;
  return `---\nname: ${config.name}\nvariant: ${config.variant}\nmodel: ${config.model}\nsystem_prompt: ${config.system_prompt}\n---\n\n${body}\n`;
}
