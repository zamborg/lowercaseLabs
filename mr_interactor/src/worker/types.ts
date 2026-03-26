export type WorkerStatus = "starting" | "running" | "waiting_for_input" | "finished";

export interface WorkerScreen {
	lines: string[];
	rows: number;
	cols: number;
	cursorRow: number;
	cursorCol: number;
}

export interface WorkerObservation {
	screen: WorkerScreen;
	screenText: string;
	bottomText: string;
	recentOutput: string;
	idleForMs: number;
	isRunning: boolean;
}

export interface WorkerState {
	status: WorkerStatus;
	reason: string;
	signature: string;
}

export interface WorkerLaunchSpec {
	command: string;
	args: string[];
	env?: Record<string, string>;
}

export interface WorkerAdapter {
	name: "claude" | "codex" | "fake";
	buildLaunchSpec(): WorkerLaunchSpec;
	classify(observation: WorkerObservation): WorkerState;
}
