import { EventEmitter } from "node:events";
import { spawn as spawnChild } from "node:child_process";
import { createInterface } from "node:readline";
import { stripVTControlCharacters } from "node:util";
import xterm from "@xterm/headless";
const { Terminal: HeadlessTerminal } = xterm;
function escapeRegExp(input) {
    return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function cleanLine(input) {
    return stripVTControlCharacters(input).replace(/\s+$/g, "");
}
function countVisibleCharacters(lines) {
    return lines.reduce((total, line) => total + line.replace(/\s+/g, "").length, 0);
}
export class WorkerSession extends EventEmitter {
    launchSpec;
    rows;
    cols;
    terminal;
    bridge;
    running = false;
    started = false;
    recentOutput = "";
    lastOutputAt = Date.now();
    maxRecentOutputChars = 16_000;
    currentRows;
    currentCols;
    writeQueue = Promise.resolve();
    constructor(launchSpec, rows, cols) {
        super();
        this.launchSpec = launchSpec;
        this.rows = rows;
        this.cols = cols;
        this.currentRows = rows;
        this.currentCols = cols;
        this.terminal = new HeadlessTerminal({
            rows,
            cols,
            scrollback: 10_000,
            allowProposedApi: true,
            convertEol: true,
        });
    }
    start(cwd) {
        if (this.started)
            return;
        this.started = true;
        const bridgeArgs = [
            "scripts/pty_bridge.py",
            "--rows",
            String(this.rows),
            "--cols",
            String(this.cols),
            "--cwd",
            cwd,
            "--",
            this.launchSpec.command,
            ...this.launchSpec.args,
        ];
        this.bridge = spawnChild("python3", bridgeArgs, {
            cwd,
            stdio: ["pipe", "pipe", "pipe"],
            env: Object.fromEntries(Object.entries({
                ...process.env,
                TERM: "xterm-256color",
                COLORTERM: "truecolor",
                FORCE_COLOR: "1",
                ...(this.launchSpec.env ?? {}),
            }).filter((entry) => typeof entry[1] === "string")),
        });
        this.running = true;
        this.lastOutputAt = Date.now();
        const stdout = createInterface({ input: this.bridge.stdout });
        stdout.on("line", (line) => {
            const event = JSON.parse(line);
            if (event.type === "data" && event.data) {
                const data = Buffer.from(event.data, "base64").toString("utf8");
                this.recentOutput = `${this.recentOutput}${data}`.slice(-this.maxRecentOutputChars);
                this.lastOutputAt = Date.now();
                this.terminal.write(data, () => {
                    this.emit("update");
                });
                return;
            }
            if (event.type === "exit") {
                this.running = false;
                this.emit("update");
                this.emit("exit", event.code ?? 0);
            }
        });
        this.bridge.stderr.on("data", (data) => {
            const text = data.toString("utf8");
            this.recentOutput = `${this.recentOutput}${text}`.slice(-this.maxRecentOutputChars);
        });
        this.bridge.on("exit", (code) => {
            this.running = false;
            this.emit("update");
            this.emit("exit", code ?? 0);
        });
    }
    stop() {
        if (!this.bridge)
            return;
        try {
            this.bridge.stdin.write(`${JSON.stringify({ type: "stop" })}\n`);
            this.bridge.kill();
        }
        finally {
            this.running = false;
        }
    }
    resize(rows, cols) {
        if (!this.bridge)
            return;
        if (rows === this.currentRows && cols === this.currentCols)
            return;
        this.currentRows = rows;
        this.currentCols = cols;
        this.bridge.stdin.write(`${JSON.stringify({ type: "resize", rows, cols })}\n`);
        this.terminal.resize(cols, rows);
        this.emit("update");
    }
    send(data) {
        this.enqueueWrite(data);
    }
    sendLine(text) {
        if (text.length > 0) {
            this.enqueueWrite(text);
        }
        this.enqueueWrite("\r", 35);
    }
    isRunning() {
        return this.running;
    }
    getObservation(topPaneRows) {
        const buffer = this.terminal.buffer.active;
        const viewportRows = Math.max(1, Math.min(topPaneRows, this.currentRows));
        const viewportStart = Math.max(0, Math.min(buffer.viewportY, Math.max(0, buffer.length - viewportRows)));
        const tailStart = Math.max(0, buffer.length - viewportRows);
        const viewportLines = this.collectLines(buffer, viewportStart, viewportRows);
        const tailLines = this.collectLines(buffer, tailStart, viewportRows);
        const useTailFallback = countVisibleCharacters(viewportLines) === 0 && countVisibleCharacters(tailLines) > 0;
        const startRow = useTailFallback ? tailStart : viewportStart;
        const lines = useTailFallback ? tailLines : viewportLines;
        const cursorAbsoluteRow = buffer.baseY + buffer.cursorY;
        const cursorRow = Math.max(0, cursorAbsoluteRow - startRow);
        const cursorCol = buffer.cursorX;
        const screen = {
            lines,
            rows: viewportRows,
            cols: this.currentCols,
            cursorRow,
            cursorCol,
        };
        const literalBottomLines = lines.slice(Math.max(0, lines.length - 6));
        const cursorWindowStart = Math.max(0, Math.min(Math.max(0, lines.length - 6), cursorRow - 2));
        const cursorWindowLines = lines.slice(cursorWindowStart, cursorWindowStart + 6);
        const bottomLines = countVisibleCharacters(literalBottomLines) > 0 ? literalBottomLines : cursorWindowLines;
        return {
            screen,
            screenText: lines.join("\n"),
            bottomText: bottomLines.join("\n"),
            recentOutput: stripVTControlCharacters(this.recentOutput),
            idleForMs: Date.now() - this.lastOutputAt,
            isRunning: this.running,
        };
    }
    collectLines(buffer, startRow, rowCount) {
        const endRow = Math.min(buffer.length, startRow + rowCount);
        const lines = [];
        for (let row = startRow; row < endRow; row += 1) {
            const line = buffer.getLine(row);
            lines.push(line ? cleanLine(line.translateToString(true)) : "");
        }
        return lines;
    }
    insertCursorMarker(lines, cursorRow, cursorCol) {
        if (cursorRow < 0 || cursorRow >= lines.length)
            return lines;
        const raw = lines[cursorRow] ?? "";
        const safeCol = Math.max(0, Math.min(cursorCol, raw.length));
        const updated = `${raw.slice(0, safeCol)}\x1b_pi:c\x07${raw.slice(safeCol)}`;
        const next = [...lines];
        next[cursorRow] = updated;
        return next;
    }
    findInRecentOutput(pattern) {
        return new RegExp(escapeRegExp(pattern), "i").test(stripVTControlCharacters(this.recentOutput));
    }
    enqueueWrite(data, delayMs = 0) {
        this.writeQueue = this.writeQueue
            .then(async () => {
            if (!this.bridge)
                return;
            if (data.length > 0) {
                const payload = Buffer.from(data, "utf8").toString("base64");
                this.bridge.stdin.write(`${JSON.stringify({ type: "write", data: payload })}\n`);
            }
            if (delayMs > 0) {
                await new Promise((resolve) => setTimeout(resolve, delayMs));
            }
        })
            .catch(() => {
            // Keep the queue alive even if a prior write raced with shutdown.
        });
    }
}
