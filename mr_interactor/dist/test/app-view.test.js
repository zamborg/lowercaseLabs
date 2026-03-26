import { describe, expect, it, vi } from "vitest";
import { AppView, buildInitialState } from "../src/ui/app-view.js";
describe("AppView", () => {
    it("queues footer notes instead of forwarding them to the worker", () => {
        const sendWorkerInput = vi.fn();
        const submitFooterNote = vi.fn();
        const view = new AppView(() => 24, 5, buildInitialState(24, 80, "finish the task", 2, 5), {
            sendWorkerInput,
            submitFooterNote,
            togglePause: vi.fn(),
            forceSupervisorTurn: vi.fn(),
            quit: vi.fn(),
        });
        view.handleInput("\u0007"); // Ctrl+G
        view.handleInput("n");
        view.handleInput("o");
        view.handleInput("t");
        view.handleInput("e");
        view.handleInput("\r");
        expect(submitFooterNote).toHaveBeenCalledWith("note");
        expect(sendWorkerInput).not.toHaveBeenCalled();
    });
});
