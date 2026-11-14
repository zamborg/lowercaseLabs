// Owns a detached process group. EOF means the desktop owner died, including SIGKILL.
import { spawn } from "node:child_process";
const [cwd, executable, ...args] = process.argv.slice(2);
const child = spawn(executable, args, {
  cwd,
  detached: true,
  stdio: ["ignore", "pipe", "pipe"],
  env: process.env,
});
child.stdout.pipe(process.stdout);
child.stderr.pipe(process.stderr);
let stopping = false;
function stop() {
  if (stopping) return;
  stopping = true;
  try {
    process.kill(-child.pid!, "SIGTERM");
  } catch {}
  setTimeout(() => {
    try {
      process.kill(-child.pid!, "SIGKILL");
    } catch {}
    process.exit(0);
  }, 1800);
}
process.stdin.resume();
process.stdin.on("end", stop);
process.on("SIGTERM", stop);
process.on("SIGINT", stop);
child.on("error", (e) => {
  console.error(e.message);
  process.exit(1);
});
child.on("exit", (code) => {
  if (!stopping) {
    try {
      process.kill(-child.pid!, "SIGTERM");
    } catch {}
    process.exit(code ?? 1);
  }
});
