#!/usr/bin/env node
import { formatUsageError, loadConfig } from "./config.js";
import { createController } from "./orchestrator/controller.js";
async function main() {
    try {
        const config = await loadConfig(process.argv.slice(2));
        const controller = await createController(config);
        await controller.run();
    }
    catch (error) {
        console.error(formatUsageError(error));
        process.exitCode = 1;
    }
}
void main();
