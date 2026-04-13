import { beforeEach, describe, expect, it, vi } from "vitest";
import { OpenAiSupervisor } from "../src/supervisor/openai-supervisor.js";
import { ClaudeAdapter } from "../src/worker/adapters.js";
import type { SupervisorContext } from "../src/supervisor/types.js";
import type { WorkerObservation } from "../src/worker/types.js";

function makeContext(): SupervisorContext {
	return {
		goal: "have the worker inspect the repo",
		maxSteps: 3,
		stepsTaken: 0,
		stepsRemaining: 3,
		worker: "claude",
		screen: "❯",
		transcript: "Claude is waiting for input",
		actionHistory: [],
		pendingHumanNotes: [],
	};
}

function makeObservation(bottomText = "❯"): WorkerObservation {
	return {
		screen: {
			lines: ["", "", bottomText],
			rows: 20,
			cols: 80,
			cursorRow: 18,
			cursorCol: 1,
		},
		screenText: bottomText,
		bottomText,
		recentOutput: bottomText,
		idleForMs: 1200,
		isRunning: true,
	};
}

describe("OpenAiSupervisor", () => {
	beforeEach(() => {
		process.env.OPENAI_API_KEY = "test-key";
	});

	it("continues an incomplete response and returns a chat tool call", async () => {
		const supervisor = new OpenAiSupervisor("gpt-5", 100);
		const create = vi
			.fn()
			.mockResolvedValueOnce({
				id: "resp_1",
				status: "incomplete",
				incomplete_details: {
					reason: "max_output_tokens",
				},
				output: [{ type: "reasoning" }],
				output_text: "",
			})
			.mockResolvedValueOnce({
				id: "resp_2",
				status: "completed",
				output: [
					{
						type: "function_call",
						name: "chat",
						arguments: JSON.stringify({ text: "run ls" }),
					},
				],
				output_text: "",
			});

		(supervisor as any).client = {
			responses: {
				create,
			},
		};

		const decision = await supervisor.decide(makeContext());

		expect(decision).toEqual({
			type: "chat",
			text: "run ls",
		});
		expect(create).toHaveBeenCalledTimes(2);
		expect(create.mock.calls[0][0].tool_choice).toBe("required");
		expect(create.mock.calls[0][0].reasoning).toEqual({ effort: "minimal" });
		expect(create.mock.calls[1][0].previous_response_id).toBe("resp_1");
	});

	it("returns noop when the model still emits no tool call", async () => {
		const supervisor = new OpenAiSupervisor("gpt-5", 100);
		const create = vi.fn().mockResolvedValue({
			id: "resp_1",
			status: "completed",
			output: [{ type: "reasoning" }],
			output_text: "",
		});

		(supervisor as any).client = {
			responses: {
				create,
			},
		};

		const decision = await supervisor.decide(makeContext());

		expect(decision).toEqual({
			type: "noop",
			description: "Supervisor returned no tool call",
		});
	});
});

describe("ClaudeAdapter", () => {
	it("accepts a trimmed prompt marker as waiting for input", () => {
		const adapter = new ClaudeAdapter();
		const state = adapter.classify(makeObservation("status line\n❯"));

		expect(state.status).toBe("waiting_for_input");
		expect(state.reason).toContain("prompt");
	});
});
