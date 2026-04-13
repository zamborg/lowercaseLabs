#!/usr/bin/env node

import readline from "node:readline";

const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
	terminal: true,
});

let cycle = 0;

function banner() {
	process.stdout.write("Fake worker booting...\n");
	process.stdout.write("Ready for instructions.\n");
	process.stdout.write("fake>\n");
}

function respond(line) {
	cycle += 1;
	const text = line.trim();
	process.stdout.write(`\n[fake-worker] turn ${cycle}\n`);
	process.stdout.write(`[fake-worker] received: ${text || "(empty)"}\n`);

	if (/npm test/i.test(text)) {
		process.stdout.write("PASS fake-worker smoke test\n");
		process.stdout.write("All tests passed.\n");
	} else if (/goal achieved|done|complete/i.test(text)) {
		process.stdout.write("Task looks complete from the fake worker side.\n");
	} else if (text.length > 0) {
		process.stdout.write(`Working on: ${text}\n`);
	}

	setTimeout(() => {
		process.stdout.write("fake>\n");
	}, 75);
}

rl.on("line", respond);
rl.on("close", () => {
	process.stdout.write("\nFake worker exiting.\n");
	process.exit(0);
});

banner();
