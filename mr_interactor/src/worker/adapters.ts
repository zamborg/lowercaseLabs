import path from "node:path";
import { createHash } from "node:crypto";
import type { WorkerAdapter, WorkerLaunchSpec, WorkerObservation, WorkerState } from "./types.js";

function buildSignature(input: string): string {
	return createHash("sha1").update(input).digest("hex");
}

function shellEscape(input: string): string {
	return `'${input.replace(/'/g, `'\\''`)}'`;
}

function shellLaunch(command: string, args: string[]): WorkerLaunchSpec {
	return {
		command: process.env.SHELL || "/bin/zsh",
		args: ["-lc", [command, ...args].map(shellEscape).join(" ")],
	};
}

function detectBottomPrompt(observation: WorkerObservation): { waiting: boolean; reason: string } {
	const bottom = observation.bottomText.toLowerCase();
	const promptHints = [
		/\bapprove\b/,
		/\bcontinue\b/,
		/\bpress enter\b/,
		/\benter to\b/,
		/\bconfirm\b/,
		/\by\/n\b/,
		/\byes\/no\b/,
		/>\s*$/m,
		/›\s*$/m,
		/❯\s*$/m,
		/⏵⏵/,
		/\bshift\+tab\b/,
	];

	const cursorNearBottom = observation.screen.cursorRow >= Math.max(0, observation.screen.rows - 4);
	const idleEnough = observation.idleForMs >= 500;
	const hasPrompt = promptHints.some((pattern) => pattern.test(bottom));

	if (idleEnough && cursorNearBottom && hasPrompt) {
		return { waiting: true, reason: "prompt markers near active cursor" };
	}

	if (idleEnough && cursorNearBottom && bottom.trim().length > 0) {
		return { waiting: true, reason: "idle cursor returned to prompt region" };
	}

	return { waiting: false, reason: "worker still producing output" };
}

abstract class BaseAdapter implements WorkerAdapter {
	abstract readonly name: "claude" | "codex" | "fake";
	abstract buildLaunchSpec(): WorkerLaunchSpec;

	classify(observation: WorkerObservation): WorkerState {
		const signature = buildSignature(
			[this.name, observation.screenText, observation.recentOutput, observation.screen.cursorRow, observation.screen.cursorCol].join("\n"),
		);

		if (!observation.isRunning) {
			return {
				status: "finished",
				reason: "worker process exited",
				signature,
			};
		}

		const prompt = detectBottomPrompt(observation);
		if (prompt.waiting) {
			return {
				status: "waiting_for_input",
				reason: prompt.reason,
				signature,
			};
		}

		if (observation.idleForMs < 350) {
			return {
				status: "running",
				reason: "worker is still active",
				signature,
			};
		}

		return {
			status: "starting",
			reason: "waiting for a stable worker prompt",
			signature,
		};
	}
}

export class ClaudeAdapter extends BaseAdapter {
	readonly name = "claude";

	buildLaunchSpec(): WorkerLaunchSpec {
		return shellLaunch("claude", ["--dangerously-skip-permissions"]);
	}
}

export class CodexAdapter extends BaseAdapter {
	readonly name = "codex";

	buildLaunchSpec(): WorkerLaunchSpec {
		return shellLaunch("codex", ["--dangerously-bypass-approvals-and-sandbox"]);
	}
}

export class FakeAdapter extends BaseAdapter {
	readonly name = "fake";

	buildLaunchSpec(): WorkerLaunchSpec {
		return shellLaunch(process.execPath, [path.resolve(process.cwd(), "scripts/fake-worker.mjs")]);
	}

	override classify(observation: WorkerObservation): WorkerState {
		const signature = buildSignature(
			[this.name, observation.screenText, observation.recentOutput, observation.screen.cursorRow, observation.screen.cursorCol].join("\n"),
		);

		if (!observation.isRunning) {
			return {
				status: "finished",
				reason: "worker process exited",
				signature,
			};
		}

		if (observation.idleForMs >= 200 && observation.screenText.includes("fake>")) {
			return {
				status: "waiting_for_input",
				reason: "fake worker prompt visible",
				signature,
			};
		}

		return {
			status: "running",
			reason: "fake worker still busy",
			signature,
		};
	}
}

export function createWorkerAdapter(kind: "claude" | "codex" | "fake"): WorkerAdapter {
	switch (kind) {
		case "claude":
			return new ClaudeAdapter();
		case "codex":
			return new CodexAdapter();
		case "fake":
			return new FakeAdapter();
		default:
			throw new Error(`Unsupported worker: ${kind satisfies never}`);
	}
}
