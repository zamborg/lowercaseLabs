import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { readCodexStatusFromTmux } from './src/index.mjs';

function sleep(ms) {
  return new Promise(resolvePromise => setTimeout(resolvePromise, ms));
}

const target = process.argv[2] ?? 'codex-status:0.0';
const outputPath = resolve(process.argv[3] ?? 'dump/codex-status-parser/out/status.json');
const intervalMs = Number.parseInt(process.argv[4] ?? '5000', 10);

mkdirSync(dirname(outputPath), { recursive: true });

for (;;) {
  const snapshot = await readCodexStatusFromTmux(target, { start: '-250' });
  writeFileSync(
    outputPath,
    `${JSON.stringify({ capturedAt: new Date().toISOString(), ...snapshot }, null, 2)}\n`
  );
  await sleep(intervalMs);
}
