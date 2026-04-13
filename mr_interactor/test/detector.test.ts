import { describe, expect, it } from "vitest";
import { TurnDetector } from "../src/orchestrator/detector.js";
import { FakeAdapter } from "../src/worker/adapters.js";
import type { WorkerObservation } from "../src/worker/types.js";

function makeObservation(overrides: Partial<WorkerObservation> = {}): WorkerObservation {
	return {
		screen: {
			lines: ["some output", "fake> "],
			rows: 20,
			cols: 80,
			cursorRow: 19,
			cursorCol: 6,
		},
		screenText: "some output\nfake> ",
		bottomText: "some output\nfake> ",
		recentOutput: "some output\n",
		idleForMs: 900,
		isRunning: true,
		...overrides,
	};
}

describe("TurnDetector", () => {
	it("waits for stability before invoking", () => {
		const detector = new TurnDetector();
		const adapter = new FakeAdapter();
		const observation = makeObservation();

		const first = detector.evaluate(adapter, observation);
		const second = detector.evaluate(adapter, observation);

		expect(first.state.status).toBe("waiting_for_input");
		expect(first.shouldInvoke).toBe(false);
		expect(second.shouldInvoke).toBe(true);
	});
});
