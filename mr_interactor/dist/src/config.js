import path from "node:path";
import { readFile } from "node:fs/promises";
function usage() {
    return [
        "Usage:",
        "  mr_interactor <claude|codex|fake> [--goal <text>]",
        "  mr_interactor run --worker <claude|codex|fake> [--goal <text>]",
        "",
        "Options:",
        "  --worker <name>                  Worker CLI to run",
        "  --task <text>                    Initial worker task text",
        "  --task-file <path>               Initial worker task file",
        "  --goal <text>                    Optional initial supervisor goal text",
        "  --goal-file <path>               Supervisor goal file",
        "  --max-steps <n>                  Maximum supervisor turns (default: 12)",
        "  --max-output-tokens <n>          Supervisor output token cap (default: 100)",
        "  --log-file <path>                Append runtime logs to this file (default: .mr_interactor.log)",
        "  --model <name>                   OpenAI model for supervisor (default: gpt-5)",
        "  --footer-height <n>              Footer height in rows (default: 6)",
        "  --idle-ms <n>                    Idle debounce for turn detection (default: 700)",
        "  --launch-delay-ms <n>            Initial delay before first task send (default: 1200)",
        "  --supervisor-mode <openai|scripted>",
        "  --scripted-decisions-file <path> JSON file used with scripted supervisor mode",
    ].join("\n");
}
function parseNumber(value, fallback, label) {
    if (value === undefined)
        return fallback;
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`Invalid ${label}: ${value}`);
    }
    return parsed;
}
async function readTextOrFile(inlineValue, fileValue, label) {
    if (inlineValue && fileValue) {
        throw new Error(`Provide either --${label} or --${label}-file, not both`);
    }
    if (inlineValue)
        return inlineValue;
    if (fileValue) {
        return readFile(fileValue, "utf8");
    }
    return "";
}
function parseRawArgs(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i += 1) {
        const value = argv[i];
        if (!value.startsWith("--")) {
            throw new Error(`Unexpected argument: ${value}`);
        }
        const key = value.slice(2);
        const next = argv[i + 1];
        if (!next || next.startsWith("--")) {
            throw new Error(`Missing value for ${value}`);
        }
        switch (key) {
            case "worker":
                args.worker = next;
                break;
            case "task":
                args.task = next;
                break;
            case "task-file":
                args.taskFile = next;
                break;
            case "goal":
                args.goal = next;
                break;
            case "goal-file":
                args.goalFile = next;
                break;
            case "max-steps":
                args.maxSteps = next;
                break;
            case "max-output-tokens":
                args.maxOutputTokens = next;
                break;
            case "model":
                args.model = next;
                break;
            case "log-file":
                args.logFile = next;
                break;
            case "footer-height":
                args.footerHeight = next;
                break;
            case "idle-ms":
                args.idleMs = next;
                break;
            case "launch-delay-ms":
                args.launchDelayMs = next;
                break;
            case "supervisor-mode":
                args.supervisorMode = next;
                break;
            case "scripted-decisions-file":
                args.scriptedDecisionsFile = next;
                break;
            default:
                throw new Error(`Unknown option: ${value}`);
        }
        i += 1;
    }
    return args;
}
export async function loadConfig(argv) {
    if (argv.length === 0 || argv.includes("--help") || argv.includes("-h")) {
        throw new Error(usage());
    }
    const [command, ...rest] = argv;
    let worker = "";
    let rawArgs;
    if (command === "run") {
        rawArgs = parseRawArgs(rest);
        worker = (rawArgs.worker ?? "").trim();
    }
    else {
        worker = command.trim();
        rawArgs = parseRawArgs(rest);
    }
    if (worker !== "claude" && worker !== "codex" && worker !== "fake") {
        throw new Error(`Worker must be one of claude, codex, or fake\n\n${usage()}`);
    }
    const task = (await readTextOrFile(rawArgs.task, rawArgs.taskFile, "task")).trim();
    const goal = (await readTextOrFile(rawArgs.goal, rawArgs.goalFile, "goal")).trim();
    const supervisorMode = (rawArgs.supervisorMode ?? "openai").trim();
    if (supervisorMode !== "openai" && supervisorMode !== "scripted") {
        throw new Error(`--supervisor-mode must be openai or scripted`);
    }
    if (supervisorMode === "scripted" && !rawArgs.scriptedDecisionsFile) {
        throw new Error(`--scripted-decisions-file is required when --supervisor-mode scripted`);
    }
    return {
        worker,
        task,
        goal,
        startEnabled: goal.length > 0,
        maxSteps: parseNumber(rawArgs.maxSteps, 12, "max-steps"),
        maxOutputTokens: parseNumber(rawArgs.maxOutputTokens, 100, "max-output-tokens"),
        logFile: path.resolve(process.cwd(), rawArgs.logFile ?? ".mr_interactor.log"),
        model: rawArgs.model ?? "gpt-5",
        footerHeight: parseNumber(rawArgs.footerHeight, 6, "footer-height"),
        idleMs: parseNumber(rawArgs.idleMs, 700, "idle-ms"),
        launchDelayMs: parseNumber(rawArgs.launchDelayMs, 1200, "launch-delay-ms"),
        supervisorMode,
        scriptedDecisionsFile: rawArgs.scriptedDecisionsFile,
    };
}
export function formatUsageError(error) {
    if (error instanceof Error) {
        return error.message;
    }
    return String(error);
}
