import type { ChildProcess } from "node:child_process";
/** Splits a child's stdout on LF only, with a size bound. */
export function jsonLines(
  child: ChildProcess,
  onRecord: (r: any) => void,
  onBad?: (e: Error) => void,
) {
  let buf = "";
  child.stdout!.setEncoding("utf8");
  child.stdout!.on("data", (s: string) => {
    buf += s;
    if (buf.length > 16_000_000) {
      buf = "";
      onBad?.(new Error("Agent message exceeds transport limit"));
      return;
    }
    let n;
    while ((n = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, n).replace(/\r$/, "");
      buf = buf.slice(n + 1);
      if (!line.trim()) continue;
      try {
        onRecord(JSON.parse(line));
      } catch (e) {
        onBad?.(e as Error);
      }
    }
  });
}
export function tail(s: string, n = 400) {
  return s.trim().slice(-n);
}
