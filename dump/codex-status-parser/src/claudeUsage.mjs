import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { promisify } from 'node:util';
import { stripTerminalControl } from './codexStatus.mjs';

const execFileAsync = promisify(execFile);
const DEFAULT_STRIP_ENV_VARS = [
  'CLAUDE_CODE_OAUTH_TOKEN',
  'ANTHROPIC_API_KEY',
  'ANTHROPIC_AUTH_TOKEN',
];

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function normalizedLines(text) {
  return stripTerminalControl(text)
    .split('\n')
    .map(line => line.replace(/\s+/g, ' ').trim())
    .filter(Boolean);
}

function parseResetBits(text) {
  const match = text.match(/(\d{1,2}:\d{2}(?:\s?[AP]M)?)\s+on\s+(.+)$/i);
  return {
    resetTime: match?.[1] ?? null,
    resetDate: match?.[2] ?? null,
    resetLabel: text || null,
  };
}

function parseUsageLine(line) {
  const patterns = [
    /^(?<label>.+?):\s+(?:\[[^\]]+\]\s+)?(?<pct>\d+)%\s+left\s+\((?:resets?|reset)\s+(?<reset>.+)\)$/i,
    /^(?<label>.+?)\s+(?<pct>\d+)%\s+left\s+(?:·|-)\s*(?:resets?|reset)\s+(?<reset>.+)$/i,
    /^(?<label>.+?):\s+(?<pct>\d+)%\s+left,\s*(?:resets?|reset)\s+(?<reset>.+)$/i,
  ];

  for (const pattern of patterns) {
    const match = line.match(pattern);
    if (!match?.groups) continue;
    const pctLeft = Number.parseInt(match.groups.pct, 10);
    return {
      label: match.groups.label.trim(),
      pctLeft,
      pctUsed: Math.max(0, Math.min(100, 100 - pctLeft)),
      ...parseResetBits(match.groups.reset.trim()),
      raw: line,
    };
  }

  return null;
}

function parseClaudeUsageBlocks(lines) {
  const windows = [];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const next = lines[index + 1] ?? '';
    const resetLine = lines[index + 2] ?? '';

    if (/^Current session$/i.test(line) && /(\d+)%\s+used/i.test(next) && /^Resets /i.test(resetLine)) {
      const pctUsed = Number.parseInt(next.match(/(\d+)%\s+used/i)?.[1] ?? '0', 10);
      windows.push({
        label: '5h limit',
        pctUsed,
        pctLeft: Math.max(0, Math.min(100, 100 - pctUsed)),
        ...parseClaudeResetLine(resetLine),
        raw: `${line} | ${next} | ${resetLine}`,
      });
    }

    if (/^Current week/i.test(line) && /(\d+)%\s+used/i.test(next) && /^Resets /i.test(resetLine)) {
      const pctUsed = Number.parseInt(next.match(/(\d+)%\s+used/i)?.[1] ?? '0', 10);
      windows.push({
        label: 'Weekly limit',
        pctUsed,
        pctLeft: Math.max(0, Math.min(100, 100 - pctUsed)),
        ...parseClaudeResetLine(resetLine),
        raw: `${line} | ${next} | ${resetLine}`,
      });
    }
  }

  return windows;
}

function parseClaudeResetLine(line) {
  const text = line.replace(/^Resets\s+/i, '').trim();
  const timeZoneMatch = text.match(/\(([^)]+)\)$/);
  const withoutTimeZone = text.replace(/\s*\([^)]+\)\s*$/, '').trim();

  let resetTime = null;
  let resetDate = null;
  if (/^[0-9]/.test(withoutTimeZone)) {
    resetTime = withoutTimeZone;
  } else {
    const match = withoutTimeZone.match(/^(.+?)\s+at\s+(.+)$/i);
    if (match) {
      resetDate = match[1];
      resetTime = match[2];
    } else {
      resetDate = withoutTimeZone;
    }
  }

  return {
    resetTime,
    resetDate,
    resetTimeZone: timeZoneMatch?.[1] ?? null,
    resetLabel: text,
  };
}

function parseSessionUsageTable(lines) {
  const sessionUsage = {};

  for (const line of lines) {
    const match = line.trim().match(/^│\s*(.+?)\s*│\s*(.+?)\s*│$/);
    if (!match) continue;

    const key = match[1].trim().toLowerCase();
    const value = match[2].trim();
    if (key === 'metric' && value === 'value') continue;

    sessionUsage[key] = value;
  }

  return Object.keys(sessionUsage).length ? sessionUsage : null;
}

function normalizeWindowLabel(label) {
  const text = label.toLowerCase();
  if (text.includes('5h') || text.includes('5-hour') || text.includes('5 hour')) return 'shortWindow';
  if (text.includes('week')) return 'longWindow';
  return null;
}

export function parseClaudeUsageText(rawText) {
  const cleanText = stripTerminalControl(rawText);
  const lines = normalizedLines(rawText);

  const base = {
    source: 'claude-usage-modal',
    rawText: cleanText,
    lines,
  };

  if (lines.some(line => /Loading usage data/i.test(line))) {
    return {
      available: false,
      state: 'loading',
      reason: 'Claude usage modal is still loading',
      ...base,
    };
  }

  if (lines.some(line => /only available for subscription plans/i.test(line))) {
    return {
      available: false,
      state: 'subscription_required',
      reason: '/usage is only available for subscription plans',
      ...base,
    };
  }

  if (lines.some(line => /\/usage isn't a recognized skill/i.test(line))) {
    return {
      available: false,
      state: 'slash_command_unavailable',
      reason: '/usage is not available as a slash command in this fresh Claude session',
      ...base,
    };
  }

  if (lines.some(line => /usage for this session/i.test(line))) {
    const sessionUsage = parseSessionUsageTable(cleanText.split('\n').map(line => line.trimEnd()));
    return {
      available: false,
      state: 'session_usage_only',
      reason: 'Claude returned session token usage, not plan quota percentages',
      sessionUsage,
      ...base,
    };
  }

  if (
    lines.some(line => /thinking with .* effort/i.test(line))
    || lines.some(line => /ready when you are|standing by|waiting\.$|^ready\.$|token status checked/i.test(line))
    || (lines.some(line => line.includes('/usage')) && lines.some(line => /esc to interrupt/i.test(line)))
  ) {
    return {
      available: false,
      state: 'command_passthrough',
      reason: '/usage was treated like normal chat input instead of opening a usage dialog',
      ...base,
    };
  }

  const windows = parseClaudeUsageBlocks(lines);
  for (const line of lines) {
    const parsed = parseUsageLine(line);
    if (parsed) windows.push(parsed);
  }

  if (!windows.length) {
    return {
      available: false,
      state: 'unparsed',
      reason: 'No Claude usage percentages found in modal',
      ...base,
    };
  }

  const rateLimit = {};
  const otherWindows = [];
  for (const window of windows) {
    const slot = normalizeWindowLabel(window.label);
    if (slot && !rateLimit[slot]) rateLimit[slot] = window;
    else otherWindows.push(window);
  }

  return {
    available: true,
    state: 'parsed',
    rateLimit,
    otherWindows,
    shortWindow: rateLimit.shortWindow ?? null,
    longWindow: rateLimit.longWindow ?? null,
    ...base,
  };
}

async function runTmux(tmuxArgs, options = {}) {
  const command = options.tmuxBinary ?? 'tmux';
  const prefix = [];
  if (options.socketName) prefix.push('-L', options.socketName);
  return execFileAsync(command, [...prefix, ...tmuxArgs], {
    maxBuffer: options.maxBuffer ?? 1024 * 1024,
    env: options.env ?? process.env,
  });
}

export async function readClaudeUsageFromTmux(target, options = {}) {
  const waitAfterOpenMs = options.waitAfterOpenMs ?? 3000;
  const waitAfterEscMs = options.waitAfterEscMs ?? 250;
  const historyLines = options.historyLines ?? '-240';

  try {
    await runTmux(['send-keys', '-t', target, 'C-u', '/usage', 'C-m'], options);
    await sleep(waitAfterOpenMs);
    const { stdout } = await runTmux(['capture-pane', '-p', '-S', String(historyLines), '-t', target], options);
    await runTmux(['send-keys', '-t', target, 'Escape'], options);
    await sleep(waitAfterEscMs);

    return {
      target,
      ...parseClaudeUsageText(stdout),
    };
  } catch (error) {
    return {
      available: false,
      state: 'error',
      target,
      reason: error.message,
    };
  }
}

export async function readClaudeUsageFromFreshSession(options = {}) {
  const id = randomUUID().slice(0, 8);
  const socketName = options.socketName ?? `claudeusage-${id}`;
  const sessionName = options.sessionName ?? 'claude-usage';
  const cwd = options.cwd ?? process.cwd();
  const strippedEnvVars = options.stripEnvVars ?? DEFAULT_STRIP_ENV_VARS;
  const unsetPrefix = strippedEnvVars.map(name => `env -u ${shellQuote(name)}`).join(' ');
  const launchCommand = options.launchCommand ?? `cd ${shellQuote(cwd)} && ${unsetPrefix} claude --model haiku`;
  const target = `${sessionName}:0.0`;
  const startupWaitMs = options.startupWaitMs ?? 6000;
  const postReadyDelayMs = options.postReadyDelayMs ?? 1500;
  const usageWaitMs = options.waitAfterOpenMs ?? 30000;
  const env = buildEnvWithout(strippedEnvVars, options.env ?? process.env);

  try {
    await runTmux(['new-session', '-d', '-s', sessionName, launchCommand], { ...options, socketName, env });
    await sleep(startupWaitMs);
    const ready = await waitForClaudePrompt(target, { ...options, socketName, env });

    if (!ready) {
      const { stdout } = await runTmux(['capture-pane', '-p', '-S', String(options.historyLines ?? '-260'), '-t', target], { ...options, socketName, env });
      return {
        target,
        sessionMode: 'fresh',
        socketName,
        strippedEnvVars,
        attempt: 1,
        available: false,
        state: 'startup_not_ready',
        reason: 'Claude prompt did not reach a stable ready state before /usage',
        source: 'claude-usage-modal',
        rawText: stripTerminalControl(stdout),
        lines: normalizedLines(stdout),
      };
    }

    await sleep(postReadyDelayMs);
    await runTmux(['send-keys', '-t', target, '/usage', 'C-m'], { ...options, socketName, env });
    const parsed = await waitForUsageState(target, { ...options, socketName, env, usageWaitMs });
    const result = {
      target,
      sessionMode: 'fresh',
      socketName,
      strippedEnvVars,
      attempt: 1,
      ...parsed,
    };

    await runTmux(['send-keys', '-t', target, 'Escape'], { ...options, socketName, env });
    await sleep(options.waitAfterEscMs ?? 300);

    return result;
  } catch (error) {
    return {
      available: false,
      state: 'error',
      target,
      sessionMode: 'fresh',
      socketName,
      strippedEnvVars,
      reason: error.message,
    };
  } finally {
    try {
      await runTmux(['kill-server'], { ...options, socketName, env });
    } catch {
      // Ignore cleanup failures.
    }
  }
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function buildEnvWithout(names, sourceEnv) {
  const env = { ...sourceEnv };
  for (const name of names) delete env[name];
  return env;
}

async function waitForClaudePrompt(target, options = {}) {
  const timeoutMs = options.readyTimeoutMs ?? 15000;
  const intervalMs = options.readyPollMs ?? 500;
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const { stdout } = await runTmux(['capture-pane', '-p', '-S', String(options.historyLines ?? '-260'), '-t', target], options);
    const lines = normalizedLines(stdout);
    if (isClaudeReady(lines)) return true;
    await sleep(intervalMs);
  }

  return false;
}

async function waitForUsageResult(target, options = {}) {
  const timeoutMs = options.usageWaitMs ?? 12000;
  const intervalMs = options.usagePollMs ?? 500;
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const { stdout } = await runTmux(['capture-pane', '-p', '-S', String(options.historyLines ?? '-260'), '-t', target], options);
    const lines = normalizedLines(stdout);
    if (isUsageResolved(lines)) return true;
    await sleep(intervalMs);
  }

  return false;
}

async function waitForUsageState(target, options = {}) {
  const timeoutMs = options.usageWaitMs ?? 30000;
  const intervalMs = options.usagePollMs ?? 500;
  const deadline = Date.now() + timeoutMs;
  let lastParsed = null;

  while (Date.now() < deadline) {
    const { stdout } = await runTmux(['capture-pane', '-p', '-S', String(options.historyLines ?? '-260'), '-t', target], options);
    const parsed = parseClaudeUsageText(stdout);
    lastParsed = parsed;
    if (parsed.state !== 'loading') return parsed;
    await sleep(intervalMs);
  }

  return lastParsed ?? {
    available: false,
    state: 'error',
    reason: 'Timed out while waiting for Claude usage result',
    source: 'claude-usage-modal',
    rawText: '',
    lines: [],
  };
}

function isClaudeReady(lines) {
  return (
    lines.some(line => /\? for shortcuts/i.test(line))
    || (
      lines.some(line => /^❯$|^❯\s*$/.test(line))
      && !lines.some(line => /thinking with .* effort|esc to interrupt/i.test(line))
    )
  );
}

function isUsageResolved(lines) {
  return lines.some(line =>
    /Esc to cancel|Loading usage data|only available for subscription plans|\/usage isn't a recognized skill|Usage for this session|5h limit|Current session|Current week/i.test(line)
  );
}
