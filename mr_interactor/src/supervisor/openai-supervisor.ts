import OpenAI from "openai";
import type { Logger } from "../logging/logger.js";
import type { Supervisor, SupervisorContext, SupervisorDecision } from "./types.js";

const SYSTEM_PROMPT = `You are MR_interactor, a supervising agent operating another coding agent through its terminal UI.

You do not execute shell commands yourself.
You only decide what to send back to the worker when it has returned control to the user.

Rules:
- Only produce one tool call: either chat or exit.
- Prefer short, surgical interventions.
- Use pending human notes as extra guidance.
- Treat goal and remaining steps as hard constraints.
- If the worker should run tests or inspect output, tell the worker to do that via chat.
- Exit only when the worker has convincingly completed the goal or is stuck beyond recovery.
`;

const TOOLS = [
	{
		type: "function" as const,
		name: "chat",
		strict: true,
		description: "Send a short text message to the worker. MR_interactor will press Enter after sending it.",
		parameters: {
			type: "object",
			properties: {
				text: {
					type: "string",
					description: "The exact text to send to the worker.",
				},
			},
			required: ["text"],
			additionalProperties: false,
		},
	},
	{
		type: "function" as const,
		name: "exit",
		strict: true,
		description: "Stop MR_interactor and report the outcome.",
		parameters: {
			type: "object",
			properties: {
				goal_achieved: {
					type: "boolean",
				},
				description: {
					type: "string",
					description: "Short explanation of why the supervisor is exiting.",
				},
			},
			required: ["goal_achieved", "description"],
			additionalProperties: false,
		},
	},
];

export class OpenAiSupervisor implements Supervisor {
	private readonly client: OpenAI;

	constructor(
		private readonly model: string,
		private readonly maxOutputTokens: number,
		private readonly logger?: Logger,
	) {
		this.client = new OpenAI();
	}

	async decide(context: SupervisorContext): Promise<SupervisorDecision> {
		this.logger?.log("supervisor", "request", {
			model: this.model,
			maxOutputTokens: this.maxOutputTokens,
			goal: context.goal,
			stepsTaken: context.stepsTaken,
			stepsRemaining: context.stepsRemaining,
			actionHistoryLength: context.actionHistory.length,
			screenTail: context.screen.split("\n").slice(-12),
			transcriptTail: context.transcript.slice(-3000),
		});

		let response: any = await this.createResponse(JSON.stringify(context, null, 2));
		let decision = this.parseDecision(response);
		if (decision) {
			return decision;
		}

		if (response.status === "incomplete" && response.incomplete_details?.reason === "max_output_tokens" && response.id) {
			this.logger?.log("supervisor", "continue_after_incomplete", {
				id: response.id,
				reason: response.incomplete_details.reason,
			});
			response = await this.createResponse(
				"Return exactly one tool call now. Choose either chat(text) or exit(goal_achieved, description).",
				response.id,
			);
			decision = this.parseDecision(response);
			if (decision) {
				return decision;
			}
		}

		this.logger?.log("supervisor", "no_tool_call", {
			status: response.status,
			incompleteDetails: response.incomplete_details ?? null,
			outputTypes: Array.isArray(response.output) ? response.output.map((item: any) => item.type) : [],
			outputText: response.output_text ?? null,
		});
		return {
			type: "noop",
			description: response.status === "incomplete" ? "Supervisor hit output limit before a tool call" : "Supervisor returned no tool call",
		};
	}

	private async createResponse(input: string, previousResponseId?: string): Promise<any> {
		const response: any = await this.client.responses.create({
			model: this.model,
			max_output_tokens: this.maxOutputTokens,
			instructions: SYSTEM_PROMPT,
			parallel_tool_calls: false,
			reasoning: {
				effort: "minimal",
			},
			tool_choice: "required",
			tools: TOOLS,
			input,
			...(previousResponseId ? { previous_response_id: previousResponseId } : {}),
		});

		this.logger?.log("supervisor", "response", {
			id: response.id,
			status: response.status,
			incompleteDetails: response.incomplete_details ?? null,
			output: response.output,
			outputText: response.output_text ?? null,
			usage: response.usage ?? null,
		});
		return response;
	}

	private parseDecision(response: any): SupervisorDecision | null {
		const functionCall = response.output?.find((item: any) => item.type === "function_call");
		if (!functionCall) {
			return null;
		}

		const args = JSON.parse(functionCall.arguments ?? "{}");
		if (functionCall.name === "chat") {
			this.logger?.log("supervisor", "tool_call", {
				name: "chat",
				args,
			});
			return {
				type: "chat",
				text: String(args.text ?? "").trim(),
			};
		}

		this.logger?.log("supervisor", "tool_call", {
			name: "exit",
			args,
		});
		return {
			type: "exit",
			goalAchieved: Boolean(args.goal_achieved),
			description: String(args.description ?? "Supervisor exited").trim(),
		};
	}
}
