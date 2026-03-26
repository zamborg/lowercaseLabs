import chalk from "chalk";
import { CURSOR_MARKER, type Component, type Focusable, Input, matchesKey, truncateToWidth, visibleWidth } from "@mariozechner/pi-tui";
import type { SupervisorAction } from "../supervisor/types.js";
import type { WorkerObservation, WorkerStatus } from "../worker/types.js";

export interface AppViewState {
	mode: "worker" | "footer";
	workerName: string;
	workerStatus: WorkerStatus;
	supervisorPaused: boolean;
	supervisorBusy: boolean;
	goal: string;
	maxSteps: number;
	stepsTaken: number;
	lastAction: string;
	statusReason: string;
	finished: boolean;
	finalMessage?: string;
	observation: WorkerObservation;
}

interface AppViewHandlers {
	sendWorkerInput(data: string): void;
	submitFooterGoal(goal: string): void;
	togglePause(): void;
	forceSupervisorTurn(): void;
	quit(): void;
}

function truncate(text: string, width: number): string {
	if (width <= 0) return "";
	const clean = text.replace(/\s+/g, " ").trim();
	return truncateToWidth(clean, width, "…", true);
}

function padLine(text: string, width: number): string {
	return truncate(text, width);
}

export class AppView implements Component, Focusable {
	focused = true;
	private readonly footerInput = new Input();
	private state: AppViewState;
	private inputMode: "worker" | "footer";

	constructor(
		private readonly getRows: () => number,
		private readonly footerHeight: number,
		initialState: AppViewState,
		private readonly handlers: AppViewHandlers,
	) {
		this.state = initialState;
		this.inputMode = initialState.mode;
		this.footerInput.onSubmit = (value) => {
			const goal = value.trim();
			this.footerInput.setValue("");
			this.setInputMode("worker");
			this.handlers.submitFooterGoal(goal);
		};
		this.footerInput.onEscape = () => {
			this.setInputMode("worker");
		};
	}

	updateState(next: AppViewState): void {
		this.state = next;
		this.footerInput.focused = this.inputMode === "footer";
	}

	invalidate(): void {}

	handleInput(data: string): void {
		if (matchesKey(data, "ctrl+x")) {
			this.handlers.quit();
			return;
		}

		if (matchesKey(data, "ctrl+g")) {
			this.setInputMode(this.inputMode === "footer" ? "worker" : "footer");
			if (this.inputMode === "footer") {
				this.footerInput.setValue(this.state.goal);
			}
			return;
		}

		if (matchesKey(data, "ctrl+p")) {
			this.handlers.togglePause();
			return;
		}

		if (matchesKey(data, "ctrl+r")) {
			this.handlers.forceSupervisorTurn();
			return;
		}

		if (this.inputMode === "footer") {
			this.footerInput.handleInput(data);
			return;
		}

		this.handlers.sendWorkerInput(data);
	}

	render(width: number): string[] {
		const totalRows = this.getRows();
		const workerHeight = Math.max(1, totalRows - this.footerHeight);
		const observation = this.state.observation;
		const rawLines = observation.screen.lines.slice(-workerHeight);
		const paddedLines = [...rawLines];
		while (paddedLines.length < workerHeight) {
			paddedLines.unshift("");
		}

		let workerLines = paddedLines.map((line) => padLine(line, width));
		if (this.inputMode === "worker") {
			const relativeCursorRow = observation.screen.cursorRow - Math.max(0, observation.screen.lines.length - workerHeight);
			if (relativeCursorRow >= 0 && relativeCursorRow < workerLines.length) {
				const line = workerLines[relativeCursorRow] ?? "";
				const safeCol = Math.max(0, Math.min(observation.screen.cursorCol, line.length));
				workerLines[relativeCursorRow] = `${line.slice(0, safeCol)}${CURSOR_MARKER}${line.slice(safeCol)}`;
			}
		}

		const theme = {
			header: chalk.black.bgCyan,
			meta: chalk.black.bgWhite,
			input: this.inputMode === "footer" ? chalk.black.bgYellow : chalk.black.bgGreen,
		};

		const mode = this.state.supervisorPaused ? "off" : this.state.supervisorBusy ? "thinking" : "on";
		const focus = this.inputMode === "footer" ? "goal" : "worker";
		const stepsRemaining = Math.max(0, this.state.maxSteps - this.state.stepsTaken);
		const headerText = ` MR_interactor | worker=${this.state.workerName} | mode=${mode} | focus=${focus} | steps=${this.state.stepsTaken}/${this.state.maxSteps} `;
		const detailText = ` status=${this.state.workerStatus} | idle=${this.state.observation.idleForMs}ms | cursor=${this.state.observation.screen.cursorRow}:${this.state.observation.screen.cursorCol} | reason=${this.state.statusReason} `;
		const goalText = ` goal=${this.state.goal || "(unset)"} | remaining=${stepsRemaining} `;
		const actionText = this.state.finished
			? ` final=${this.state.finalMessage ?? "finished"} `
			: ` last=${this.state.lastAction || this.state.statusReason} `;

		const baseFooterLines = [
			theme.header(padLine(headerText, width)),
			theme.meta(padLine(detailText, width)),
			theme.meta(padLine(goalText, width)),
			theme.meta(padLine(actionText, width)),
			theme.meta(padLine(` hint: Ctrl+G goal | Ctrl+P on/off | Ctrl+R force | Ctrl+X quit `, width)),
			this.renderInputLine(width, theme.input),
		];

		const footerLines =
			this.footerHeight >= baseFooterLines.length
				? [...baseFooterLines]
				: [...baseFooterLines.slice(0, Math.max(0, this.footerHeight - 1)), baseFooterLines[baseFooterLines.length - 1]];

		while (footerLines.length < this.footerHeight) {
			footerLines.splice(footerLines.length - 1, 0, theme.meta(padLine("", width)));
		}

		return [...workerLines, ...footerLines];
	}

	private renderInputLine(width: number, bg: (text: string) => string): string {
		if (this.inputMode === "footer") {
			const prefix = " goal> ";
			const innerWidth = Math.max(0, width - visibleWidth(prefix));
			const rendered = this.footerInput.render(innerWidth)[0] ?? "";
			return bg(truncateToWidth(`${prefix}${rendered}`, width, "", true));
		}

		const label = this.state.goal ? ` goal> ${this.state.goal}` : " goal> (Ctrl+G to set the MR_interactor goal)";
		return bg(padLine(label, width));
	}

	private setInputMode(mode: "worker" | "footer"): void {
		this.inputMode = mode;
		this.footerInput.focused = mode === "footer";
	}
}

export function buildInitialState(rows: number, cols: number, workerName: string, goal: string, maxSteps: number, footerHeight = 6): AppViewState {
	return {
		mode: "worker",
		workerName,
		workerStatus: "starting",
		supervisorPaused: true,
		supervisorBusy: false,
		goal,
		maxSteps,
		stepsTaken: 0,
		lastAction: "",
		statusReason: "booting",
		finished: false,
		observation: {
			screen: {
				lines: Array.from({ length: Math.max(1, rows - footerHeight) }, () => ""),
				rows,
				cols,
				cursorRow: 0,
				cursorCol: 0,
			},
			screenText: "",
			bottomText: "",
			recentOutput: "",
			idleForMs: 0,
			isRunning: true,
		},
	};
}
