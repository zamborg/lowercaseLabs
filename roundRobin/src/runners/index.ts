import { ClaudeCodeRunner } from "./claudeCode.ts";

export interface RuntimeAgent {
  id: string;
  name: string;
  model: string;
  session: string;
}

export interface Runner {
  // One-time setup before any agents are spawned (e.g. write ~/.claude.json for all workspaces)
  bootstrap(agents: RuntimeAgent[], runDir: string): Promise<void>;
  // Spawn the agent in its pre-staged workspace
  spawn(agent: RuntimeAgent, workspace: string): Promise<void>;
  // Poll until the agent is at an idle prompt, dismissing any interactive confirmations along the way
  waitUntilReady(agent: RuntimeAgent, timeoutMs?: number): Promise<boolean>;
  // Terminate the agent process
  kill(agent: RuntimeAgent): Promise<void>;
}

export function createRunner(variant: string, log: (msg: string) => void): Runner {
  switch (variant) {
    case "claude-code":
      return new ClaudeCodeRunner(log);
    default:
      throw new Error(`Unknown agent variant: "${variant}". Supported: claude-code`);
  }
}
