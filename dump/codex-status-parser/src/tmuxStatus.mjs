import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { parseCodexStatusText } from './codexStatus.mjs';

const execFileAsync = promisify(execFile);

export async function captureTmuxPane(target, options = {}) {
  const {
    start = '-200',
    end,
    tmuxBinary = 'tmux',
    maxBuffer = 1024 * 1024,
  } = options;

  const args = ['capture-pane', '-p'];
  if (start !== undefined && start !== null) args.push('-S', String(start));
  if (end !== undefined && end !== null) args.push('-E', String(end));
  if (target) args.push('-t', target);

  const { stdout } = await execFileAsync(tmuxBinary, args, { maxBuffer });
  return stdout;
}

export async function readCodexStatusFromTmux(target, options = {}) {
  try {
    const rawText = await captureTmuxPane(target, options);
    return {
      target: target ?? null,
      ...parseCodexStatusText(rawText),
    };
  } catch (error) {
    return {
      available: false,
      target: target ?? null,
      reason: error.message,
    };
  }
}
