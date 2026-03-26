import type { WorkerAdapter, WorkerObservation, WorkerState } from "../worker/types.js";

export interface TurnDetectionResult {
	state: WorkerState;
	shouldInvoke: boolean;
}

export class TurnDetector {
	private lastSignature = "";
	private stabilityCount = 0;
	private lastHandledSignature = "";

	evaluate(adapter: WorkerAdapter, observation: WorkerObservation): TurnDetectionResult {
		const state = adapter.classify(observation);
		if (state.signature === this.lastSignature) {
			this.stabilityCount += 1;
		} else {
			this.lastSignature = state.signature;
			this.stabilityCount = 1;
		}

		const shouldInvoke =
			state.status === "waiting_for_input" && this.stabilityCount >= 2 && state.signature !== this.lastHandledSignature;

		return { state, shouldInvoke };
	}

	markHandled(state: WorkerState): void {
		this.lastHandledSignature = state.signature;
	}

	resetHandled(): void {
		this.lastHandledSignature = "";
	}
}
