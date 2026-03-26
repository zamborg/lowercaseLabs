import { visibleWidth } from "@mariozechner/pi-tui";
import { describe, expect, it, vi } from "vitest";
import { AppView, buildInitialState } from "../src/ui/app-view.js";

describe("AppView", () => {
	it("queues footer notes instead of forwarding them to the worker", () => {
		const sendWorkerInput = vi.fn();
		const submitFooterGoal = vi.fn();
		const view = new AppView(
			() => 24,
			5,
			buildInitialState(24, 80, "fake", "", 2, 5),
			{
				sendWorkerInput,
				submitFooterGoal,
				togglePause: vi.fn(),
				forceSupervisorTurn: vi.fn(),
				quit: vi.fn(),
			},
		);

		view.handleInput("\u0007"); // Ctrl+G
		view.handleInput("n");
		view.handleInput("o");
		view.handleInput("t");
		view.handleInput("e");
		view.handleInput("\r");

		expect(submitFooterGoal).toHaveBeenCalledWith("note");
		expect(sendWorkerInput).not.toHaveBeenCalled();
	});

	it("keeps footer edit lines within the terminal width", () => {
		const view = new AppView(
			() => 24,
			5,
			buildInitialState(24, 80, "fake", "A very long goal that should still fit once the footer editor opens and renders the cursor state safely", 2, 5),
			{
				sendWorkerInput: vi.fn(),
				submitFooterGoal: vi.fn(),
				togglePause: vi.fn(),
				forceSupervisorTurn: vi.fn(),
				quit: vi.fn(),
			},
		);

		view.handleInput("\u0007");
		const lines = view.render(40);

		for (const line of lines) {
			expect(visibleWidth(line)).toBeLessThanOrEqual(40);
		}
	});

	it("keeps footer focus across state updates", () => {
		const submitFooterGoal = vi.fn();
		const view = new AppView(
			() => 24,
			5,
			buildInitialState(24, 80, "fake", "current goal", 2, 5),
			{
				sendWorkerInput: vi.fn(),
				submitFooterGoal,
				togglePause: vi.fn(),
				forceSupervisorTurn: vi.fn(),
				quit: vi.fn(),
			},
		);

		view.handleInput("\u0007");
		view.updateState({
			...buildInitialState(24, 80, "fake", "current goal", 2, 5),
			lastAction: "rerender",
		});
		view.handleInput("x");
		view.handleInput("\r");

		expect(submitFooterGoal).toHaveBeenCalledWith("xcurrent goal");
	});
});
