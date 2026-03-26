export class TurnDetector {
    lastSignature = "";
    stabilityCount = 0;
    lastHandledSignature = "";
    evaluate(adapter, observation) {
        const state = adapter.classify(observation);
        if (state.signature === this.lastSignature) {
            this.stabilityCount += 1;
        }
        else {
            this.lastSignature = state.signature;
            this.stabilityCount = 1;
        }
        const shouldInvoke = state.status === "waiting_for_input" && this.stabilityCount >= 2 && state.signature !== this.lastHandledSignature;
        return { state, shouldInvoke };
    }
    markHandled(state) {
        this.lastHandledSignature = state.signature;
    }
    resetHandled() {
        this.lastHandledSignature = "";
    }
}
