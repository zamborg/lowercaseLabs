import { mkdirSync, createWriteStream } from "node:fs";
import path from "node:path";
export class FileLogger {
    filePath;
    stream;
    constructor(filePath) {
        this.filePath = filePath;
        mkdirSync(path.dirname(filePath), { recursive: true });
        this.stream = createWriteStream(filePath, { flags: "a" });
        this.log("session", "start", { pid: process.pid, cwd: process.cwd() });
    }
    log(scope, event, data) {
        const record = {
            ts: new Date().toISOString(),
            scope,
            event,
            data: data ?? null,
        };
        this.stream.write(`${JSON.stringify(record)}\n`);
    }
    getPath() {
        return this.filePath;
    }
}
