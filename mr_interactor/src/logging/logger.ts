import { mkdirSync, createWriteStream, type WriteStream } from "node:fs";
import path from "node:path";

export interface Logger {
	log(scope: string, event: string, data?: unknown): void;
	getPath(): string;
}

export class FileLogger implements Logger {
	private readonly stream: WriteStream;

	constructor(private readonly filePath: string) {
		mkdirSync(path.dirname(filePath), { recursive: true });
		this.stream = createWriteStream(filePath, { flags: "a" });
		this.log("session", "start", { pid: process.pid, cwd: process.cwd() });
	}

	log(scope: string, event: string, data?: unknown): void {
		const record = {
			ts: new Date().toISOString(),
			scope,
			event,
			data: data ?? null,
		};
		this.stream.write(`${JSON.stringify(record)}\n`);
	}

	getPath(): string {
		return this.filePath;
	}
}
