import { existsSync, mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { $ } from "bun";
import type { Runner, RuntimeAgent } from "./index.ts";

export class ClaudeCodeRunner implements Runner {
  private log: (msg: string) => void;

  constructor(log: (msg: string) => void) {
    this.log = log;
  }

  // Write ~/.claude.json and ~/.claude/settings.json for the agent user.
  // Must be called once before any spawns — pre-trusts all workspace paths so
  // no interactive dialogs appear when Claude Code starts.
  async bootstrap(agents: RuntimeAgent[], runDir: string): Promise<void> {
    if (process.getuid?.() !== 0) return; // only needed in Docker (running as root)

    const claudeDir = "/home/agent/.claude";
    mkdirSync(claudeDir, { recursive: true });
    writeFileSync(
      join(claudeDir, "settings.json"),
      JSON.stringify({ skipDangerousModePermissionPrompt: true })
    );

    const projects: Record<string, any> = {};
    for (const agent of agents) {
      projects[join(runDir, "workspaces", agent.id)] = {
        hasTrustDialogAccepted: true,
        projectOnboardingSeenCount: 1,
        enabledMcpjsonServers: ["roundrobin-chat"],
      };
    }
    writeFileSync(
      "/home/agent/.claude.json",
      JSON.stringify({
        hasCompletedOnboarding: true,
        hasTrustDialogAccepted: true,
        hasTrustDialogHooksAccepted: true,
        projects,
      })
    );
    await $`chown -R agent:agent ${claudeDir} /home/agent/.claude.json`.quiet();
  }

  async spawn(agent: RuntimeAgent, workspace: string): Promise<void> {
    const isRoot = process.getuid?.() === 0;

    const authEnvParts: string[] = [];
    if (Bun.env.ANTHROPIC_API_KEY) authEnvParts.push(`ANTHROPIC_API_KEY=${Bun.env.ANTHROPIC_API_KEY}`);
    if (Bun.env.CLAUDE_CODE_OAUTH_TOKEN) authEnvParts.push(`CLAUDE_CODE_OAUTH_TOKEN=${Bun.env.CLAUDE_CODE_OAUTH_TOKEN}`);
    const authEnv = authEnvParts.join(" ");

    await $`tmux new-session -d -s ${agent.session} -x 200 -y 50 -c ${workspace}`.quiet();

    // In Docker we run as root but claude refuses --dangerously-skip-permissions as root.
    // Use sudo -u agent with explicit env vars since sudo strips them.
    const cmd = isRoot
      ? `sudo -u agent env -u CLAUDECODE ${authEnv} bash -c 'cd ${workspace} && claude --model ${agent.model} --dangerously-skip-permissions --dangerously-load-development-channels server:roundrobin-chat'`
      : `env -u CLAUDECODE ${authEnv} claude --model ${agent.model} --dangerously-skip-permissions --dangerously-load-development-channels server:roundrobin-chat`;

    await $`tmux send-keys -t ${agent.session} ${cmd} Enter`.quiet();
    this.log(`Spawning ${agent.id} (${agent.name})...`);
  }

  async waitUntilReady(agent: RuntimeAgent, timeoutMs = 90000): Promise<boolean> {
    const maxAttempts = Math.floor(timeoutMs / 1000);
    let dismissedChannelPrompt = false;
    let dismissedPermPrompt = false;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const { stdout } = await $`tmux capture-pane -t ${agent.session} -p -S -50`.quiet();
        const text = stdout.toString();
        const textLower = text.toLowerCase();

        if (!dismissedChannelPrompt && text.includes("I am using this for local development")) {
          await $`tmux send-keys -t ${agent.session} Enter`.quiet();
          dismissedChannelPrompt = true;
          this.log(`${agent.id}: confirmed dev channels`);
          await Bun.sleep(300);
          continue;
        }

        if (!dismissedPermPrompt && textLower.includes("bypass permissions")) {
          await $`tmux send-keys -t ${agent.session} Enter`.quiet();
          dismissedPermPrompt = true;
          this.log(`${agent.id}: confirmed permissions bypass`);
          await Bun.sleep(300);
          continue;
        }

        const isIdle =
          (text.includes(">") || text.includes("What would you like to do?") || text.includes("$"))
          && !text.includes("esc to interrupt")
          && (dismissedChannelPrompt || dismissedPermPrompt);

        if (isIdle) {
          this.log(`${agent.id} (${agent.name}) is ready`);
          return true;
        }
      } catch {}
      await Bun.sleep(1000);
    }

    this.log(`${agent.id} (${agent.name}) timed out waiting for idle`);
    return false;
  }

  async kill(agent: RuntimeAgent): Promise<void> {
    try {
      await $`tmux kill-session -t ${agent.session}`.quiet();
    } catch {}
  }
}
