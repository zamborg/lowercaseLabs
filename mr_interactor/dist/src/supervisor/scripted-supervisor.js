import { readFile } from "node:fs/promises";
export class ScriptedSupervisor {
    decisions;
    constructor(decisions) {
        this.decisions = decisions;
    }
    static async fromFile(path) {
        const content = await readFile(path, "utf8");
        const parsed = JSON.parse(content);
        const decisions = parsed.map((item) => {
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
    async decide(_context) {
        if (this.decisions.length === 0) {
            return {
                type: "exit",
                goalAchieved: false,
                description: "No scripted decisions left",
            };
        }
        return this.decisions.shift();
    }
}
