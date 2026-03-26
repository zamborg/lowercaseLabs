import { readFile } from "node:fs/promises";
import type { Supervisor, SupervisorContext, SupervisorDecision } from "./types.js";

interface RawScriptedDecision {
	type: "chat" | "noop" | "exit";
	text?: string;
	goalAchieved?: boolean;
	description?: string;
}

export class ScriptedSupervisor implements Supervisor {
	private constructor(private readonly decisions: SupervisorDecision[]) {}

	static async fromFile(path: string): Promise<ScriptedSupervisor> {
		const content = await readFile(path, "utf8");
		const parsed = JSON.parse(content) as RawScriptedDecision[];
		const decisions: SupervisorDecision[] = parsed.map((item) => {
			if (item.type === "chat") {
				return {
					type: "chat",
					text: String(item.text ?? ""),
				};
			}
			if (item.type === "noop") {
				return {
					type: "noop",
					description: String(item.description ?? "scripted noop"),
				};
			}
			return {
				type: "exit",
				goalAchieved: Boolean(item.goalAchieved),
				description: String(item.description ?? "scripted exit"),
			};
		});
		return new ScriptedSupervisor(decisions);
	}

	async decide(_context: SupervisorContext): Promise<SupervisorDecision> {
		if (this.decisions.length === 0) {
			return {
				type: "exit",
				goalAchieved: false,
				description: "No scripted decisions left",
			};
		}
		return this.decisions.shift()!;
	}
}
