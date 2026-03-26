export type SupervisorDecision =
	| {
			type: "chat";
			text: string;
	  }
	| {
			type: "noop";
			description: string;
	  }
	| {
			type: "exit";
			goalAchieved: boolean;
			description: string;
	  };

export interface SupervisorAction {
	at: string;
	decision: SupervisorDecision;
}

export interface SupervisorContext {
	goal: string;
	maxSteps: number;
	stepsTaken: number;
	stepsRemaining: number;
	worker: "claude" | "codex" | "fake";
	screen: string;
	transcript: string;
	actionHistory: SupervisorAction[];
	pendingHumanNotes: string[];
}

export interface Supervisor {
	decide(context: SupervisorContext): Promise<SupervisorDecision>;
}
