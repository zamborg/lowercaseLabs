import { ProcessTerminal, TUI } from "@mariozechner/pi-tui";
import type { AppConfig } from "../config.js";
import type { Logger } from "../logging/logger.js";
import { OpenAiSupervisor } from "../supervisor/openai-supervisor.js";
import { ScriptedSupervisor } from "../supervisor/scripted-supervisor.js";
import type { Supervisor, SupervisorAction, SupervisorContext, SupervisorDecision } from "../supervisor/types.js";
import { AppView, buildInitialState, type AppViewState } from "../ui/app-view.js";
import { createWorkerAdapter } from "../worker/adapters.js";
import { WorkerSession } from "../worker/session.js";
import type { WorkerObservation, WorkerState } from "../worker/types.js";
import { TurnDetector } from "./detector.js";

export interface Controller {
	run(): Promise<void>;
}

class MrInteractorController implements Controller {
	private readonly terminal = new ProcessTerminal();
	private readonly tui = new TUI(this.terminal, true);
	private readonly detector = new TurnDetector();
	private readonly actionHistory: SupervisorAction[] = [];
	private readonly workerAdapter;
	private readonly workerSession;
	private readonly appView;
	private readonly state: AppViewState;
	private lastLoggedStateSignature = "";

	private tickTimer?: NodeJS.Timeout;
	private done = false;
	private startedAt = Date.now();
	private initialTaskSent = false;
	private supervisorBusy = false;
	private forceSupervisorTurn = false;
	private nextSupervisorEligibleAt = 0;

	constructor(
		private readonly config: AppConfig,
		private readonly supervisor: Supervisor,
		private readonly logger: Logger,
	) {
		this.logger.log("controller", "init", {
			worker: this.config.worker,
			startEnabled: this.config.startEnabled,
			maxSteps: this.config.maxSteps,
			maxOutputTokens: this.config.maxOutputTokens,
			logFile: this.config.logFile,
		});
		this.workerAdapter = createWorkerAdapter(this.config.worker);
		this.workerSession = new WorkerSession(
			this.workerAdapter.buildLaunchSpec(),
			24,
			120,
		);
		this.state = buildInitialState(
			this.terminal.rows || 24,
			this.terminal.columns || 120,
			this.config.worker,
			this.config.goal,
			this.config.maxSteps,
			this.config.footerHeight,
		);
		this.state.supervisorPaused = !this.config.startEnabled;
		this.appView = new AppView(
			() => this.tui.terminal.rows,
			this.config.footerHeight,
			this.state,
			{
				sendWorkerInput: (data) => this.workerSession.send(data),
				submitFooterGoal: (goal) => {
					this.state.goal = goal;
					if (goal.length === 0) {
						this.state.supervisorPaused = true;
						this.state.lastAction = "goal cleared; MR_interactor turned off";
					} else {
						this.state.lastAction = `goal updated: ${goal}`;
					}
					this.logger.log("ui", "goal_updated", {
						goal,
						supervisorPaused: this.state.supervisorPaused,
					});
					this.requestRender();
				},
				togglePause: () => {
					this.state.supervisorPaused = !this.state.supervisorPaused;
					this.state.lastAction = this.state.supervisorPaused ? "MR_interactor turned off" : "MR_interactor turned on";
					this.logger.log("ui", "toggle_pause", {
						supervisorPaused: this.state.supervisorPaused,
					});
					this.requestRender();
				},
				forceSupervisorTurn: () => {
					this.forceSupervisorTurn = true;
					this.logger.log("ui", "force_turn", {
						goal: this.state.goal,
						status: this.state.workerStatus,
					});
					this.requestRender();
				},
				quit: () => {
					this.finish(false, "quit by user");
				},
			},
		);
		this.tui.addChild(this.appView);
		this.tui.setFocus(this.appView);
		this.workerSession.on("update", () => {
			this.requestRender();
		});
		this.workerSession.on("exit", () => {
			if (!this.done) {
				this.finish(false, "worker exited");
			}
		});
	}

	async run(): Promise<void> {
		this.tui.start();
		this.workerSession.start(process.cwd());
		this.logger.log("controller", "run_started", {
			cwd: process.cwd(),
		});
		this.tickTimer = setInterval(() => {
			void this.tick();
		}, 200);
		this.requestRender();

		await new Promise<void>((resolve) => {
			const stop = () => {
				if (this.done) {
					resolve();
				} else {
					setTimeout(stop, 50);
				}
			};
			stop();
		});
	}

	private requestRender(): void {
		const observation = this.getObservation();
		this.state.observation = observation;
		this.state.supervisorBusy = this.supervisorBusy;
		this.appView.updateState({ ...this.state });
		this.tui.requestRender();
	}

	private getObservation(): WorkerObservation {
		const footerHeight = this.config.footerHeight;
		const rows = Math.max(1, this.tui.terminal.rows - footerHeight);
		const cols = Math.max(20, this.tui.terminal.columns);
		this.workerSession.resize(rows, cols);
		return this.workerSession.getObservation(rows);
	}

	private async tick(): Promise<void> {
		if (this.done) return;

		const observation = this.getObservation();
		const detection = this.detector.evaluate(this.workerAdapter, observation);
		this.logDetectionIfChanged(detection.state, observation);
		this.applyWorkerState(detection.state, observation);

		if (!this.initialTaskSent && this.config.task && detection.state.status === "waiting_for_input") {
			if (Date.now() - this.startedAt >= this.config.launchDelayMs) {
				this.initialTaskSent = true;
				this.detector.markHandled(detection.state);
				this.state.lastAction = "sent initial task";
				this.logger.log("worker", "send_initial_task", {
					task: this.config.task,
				});
				this.workerSession.sendLine(this.config.task);
				this.requestRender();
			}
			return;
		}

		if (this.supervisorBusy || this.state.supervisorPaused || this.state.goal.trim().length === 0) {
			return;
		}

		if (!this.forceSupervisorTurn && Date.now() < this.nextSupervisorEligibleAt) {
			return;
		}

		if (this.state.stepsTaken >= this.config.maxSteps) {
			this.finish(false, "step budget exhausted");
			return;
		}

		if (!detection.shouldInvoke && !this.forceSupervisorTurn) {
			return;
		}

		this.forceSupervisorTurn = false;
		this.detector.markHandled(detection.state);
		this.logger.log("detector", "invoke_supervisor", {
			reason: detection.state.reason,
			status: detection.state.status,
		});
		await this.invokeSupervisor(observation, detection.state);
	}

	private applyWorkerState(workerState: WorkerState, observation: WorkerObservation): void {
		this.state.workerStatus = workerState.status;
		this.state.statusReason = workerState.reason;
		this.state.observation = observation;
		this.requestRender();
	}

	private async invokeSupervisor(observation: WorkerObservation, workerState: WorkerState): Promise<void> {
		this.supervisorBusy = true;
		this.state.supervisorBusy = true;
		this.state.lastAction = "supervisor thinking";
		this.requestRender();

		try {
			const context: SupervisorContext = {
				goal: this.state.goal,
				maxSteps: this.config.maxSteps,
				stepsTaken: this.state.stepsTaken,
				stepsRemaining: Math.max(0, this.config.maxSteps - this.state.stepsTaken),
				worker: this.config.worker,
				screen: observation.screenText,
				transcript: observation.recentOutput,
				actionHistory: [...this.actionHistory],
				pendingHumanNotes: [],
			};

			const decision = await this.supervisor.decide(context);
			this.state.stepsTaken += 1;
			this.recordAction(decision);
			this.logger.log("supervisor", "decision", decision);
			this.applyDecision(decision, workerState);
		} catch (error) {
			this.logger.log("supervisor", "error", {
				message: error instanceof Error ? error.message : String(error),
			});
			this.finish(false, `supervisor error: ${error instanceof Error ? error.message : String(error)}`);
		} finally {
			this.supervisorBusy = false;
			this.state.supervisorBusy = false;
			this.requestRender();
		}
	}

	private recordAction(decision: SupervisorDecision): void {
		this.actionHistory.push({
			at: new Date().toISOString(),
			decision,
		});
	}

	private applyDecision(decision: SupervisorDecision, _workerState: WorkerState): void {
		if (decision.type === "chat") {
			this.state.lastAction = `chat: ${decision.text}`;
			this.logger.log("worker", "send_chat", {
				text: decision.text,
			});
			this.workerSession.sendLine(decision.text);
			return;
		}

		if (decision.type === "noop") {
			this.state.lastAction = decision.description;
			this.detector.resetHandled();
			this.nextSupervisorEligibleAt = Date.now() + Math.max(800, this.config.idleMs);
			this.logger.log("supervisor", "retry_scheduled", {
				description: decision.description,
				retryAt: new Date(this.nextSupervisorEligibleAt).toISOString(),
			});
			return;
		}

		this.finish(decision.goalAchieved, decision.description);
	}

	private finish(goalAchieved: boolean, message: string): void {
		if (this.done) return;
		this.done = true;
		this.logger.log("controller", "finish", {
			goalAchieved,
			message,
			stepsTaken: this.state.stepsTaken,
		});
		process.exitCode = goalAchieved ? 0 : 1;
		this.state.finished = true;
		this.state.workerStatus = "finished";
		this.state.finalMessage = `${goalAchieved ? "success" : "stopped"}: ${message}`;
		this.state.lastAction = this.state.finalMessage;
		if (this.tickTimer) {
			clearInterval(this.tickTimer);
			this.tickTimer = undefined;
		}
		this.requestRender();
		setTimeout(() => {
			this.workerSession.stop();
			this.tui.stop();
		}, 100);
	}

	private logDetectionIfChanged(workerState: WorkerState, observation: WorkerObservation): void {
		const signature = `${workerState.signature}:${observation.idleForMs}:${observation.screen.cursorRow}:${observation.screen.cursorCol}`;
		if (signature === this.lastLoggedStateSignature) return;
		this.lastLoggedStateSignature = signature;
		this.logger.log("detector", "state", {
			status: workerState.status,
			reason: workerState.reason,
			idleMs: observation.idleForMs,
			cursorRow: observation.screen.cursorRow,
			cursorCol: observation.screen.cursorCol,
			bottomText: observation.bottomText,
		});
	}
}

export async function createController(config: AppConfig): Promise<Controller> {
	const { FileLogger } = await import("../logging/logger.js");
	const logger = new FileLogger(config.logFile);
	const supervisor =
		config.supervisorMode === "scripted"
			? await ScriptedSupervisor.fromFile(config.scriptedDecisionsFile!)
			: new OpenAiSupervisor(config.model, config.maxOutputTokens, logger);

	return new MrInteractorController(config, supervisor, logger);
}
