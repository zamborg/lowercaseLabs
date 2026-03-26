import { once } from "node:events";
import { describe, expect, it } from "vitest";
import { FakeAdapter } from "../src/worker/adapters.js";
import { WorkerSession } from "../src/worker/session.js";
async function waitForPrompt(session, pattern, timeoutMs = 5_000) {
    const startedAt = Date.now();
    for (;;) {
        const observation = session.getObservation(20);
        if (observation.screenText.includes(pattern)) {
            return;
        }
        if (Date.now() - startedAt > timeoutMs) {
            throw new Error(`Timed out waiting for prompt: ${pattern}`);
        }
        await once(session, "update");
    }
}
describe("WorkerSession", () => {
    it("spawns the fake worker and exchanges lines through the pty", async () => {
        const adapter = new FakeAdapter();
        const session = new WorkerSession(adapter.buildLaunchSpec(), 20, 80);
        session.start(process.cwd());
        try {
            await waitForPrompt(session, "fake>");
            session.sendLine("run npm test");
            await waitForPrompt(session, "All tests passed.");
            const observation = session.getObservation(20);
            expect(observation.screenText).toContain("PASS fake-worker smoke test");
            expect(observation.bottomText).toContain("fake>");
        }
        finally {
            session.stop();
        }
    }, 10_000);
});
