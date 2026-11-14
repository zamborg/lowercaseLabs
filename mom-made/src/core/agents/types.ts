import { realpathSync } from "node:fs";
import type { ActivityKind, AgentId, Model } from "../contracts";
export type AgentEvent =
  | { type: "text"; delta: string }
  | { type: "textBreak" }
  | { type: "activity"; kind: ActivityKind; text: string; detail?: string }
  | {
      type: "approval";
      id: string | number;
      what: "command" | "files";
      reason?: string;
      command?: string;
    }
  | { type: "session"; id: string }
  | { type: "done"; status: "completed" | "interrupted" | "failed"; error?: string };
export type Help = {
  install: string[];
  loginHint?: string;
  needsKey?: boolean;
  note?: string;
};
export type Probe = {
  installed: boolean;
  ready: boolean;
  account?: string;
  detail?: string;
  models: Model[];
};
export type McpLaunch = {
  command: string;
  args: string[];
  env: Record<string, string>;
};
export type TurnInput = {
  cwd: string;
  text: string;
  images: string[];
  model: string;
  effort: string;
  session?: string;
  instructions: string;
  mcp?: McpLaunch;
};
export type Decision = "accept" | "always" | "decline";
/** One coding agent. Everything provider-specific stays behind this. */
export interface Agent {
  readonly id: AgentId;
  readonly label: string;
  readonly help: Help;
  /** Whether the agent can ask the user before acting. */
  readonly approvals: boolean;
  probe(): Promise<Probe>;
  /** Runs one turn; resolves after emitting a terminal `done` event. */
  run(t: TurnInput, emit: (e: AgentEvent) => void): Promise<void>;
  interrupt(): Promise<void>;
  answer(id: string | number, d: Decision): void;
  login?(): Promise<void>;
  compact?(cwd: string, session: string): Promise<void>;
  close(): void;
}
export const TOOL_LABEL: Record<string, string> = {
  app_screenshot: "Looked at the app",
  app_page_text: "Read the page",
  app_console: "Checked for errors",
  app_logs: "Checked the server",
  app_click: "Tried clicking in the app",
  app_type: "Tried typing in the app",
  app_reload: "Reloaded the app",
};
export function classify(command: string): {
  kind: "look" | "command";
  text: string;
} {
  const c = command.replace(/^\/bin\/(zsh|bash|sh) -lc\s+/, "");
  return /^['"]?\s*(rg|grep|ls|cat|sed -n|find|head|tail|wc|pwd|git (status|diff|log))\b/.test(
    c,
  )
    ? { kind: "look", text: "Looked through the files" }
    : { kind: "command", text: "Ran a command" };
}
export function toolActivity(name: string): AgentEvent | undefined {
  const short = name.replace(/^mcp__mom[-_]made__/, "");
  if (TOOL_LABEL[short])
    return { type: "activity", kind: "look", text: TOOL_LABEL[short] };
  return undefined;
}
/** Makes an agent-reported path relative to the app, even if it reports the realpath. */
export function relPath(p: string, appPath: string) {
  let real = appPath;
  try {
    real = realpathSync(appPath);
  } catch {}
  for (const root of [appPath, real])
    if (p.startsWith(root + "/")) return p.slice(root.length + 1);
  return p;
}
