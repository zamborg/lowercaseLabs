import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const DEFAULT_SESSIONS_DIR = join(homedir(), '.codex', 'sessions');

function walkSessionFiles(dir, files = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      walkSessionFiles(path, files);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.jsonl')) files.push(path);
  }
  return files;
}

function clampPercent(value) {
  return Math.max(0, Math.min(100, value));
}

function formatReset(epochSeconds, timeZone) {
  const date = new Date(epochSeconds * 1000);
  const resetTime = new Intl.DateTimeFormat('en-GB', {
    timeZone,
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date);
  const resetDate = new Intl.DateTimeFormat('en-GB', {
    timeZone,
    day: '2-digit',
    month: 'short',
  }).format(date);

  return {
    resetAtEpochSeconds: epochSeconds,
    resetAtIso: date.toISOString(),
    resetTime,
    resetDate,
    resetLabel: `${resetTime} on ${resetDate}`,
  };
}

function normalizeWindow(limit, label, timeZone) {
  const pctUsed = clampPercent(Number(limit?.used_percent ?? 0));
  const pctLeft = clampPercent(100 - pctUsed);

  return {
    label,
    windowMinutes: Number(limit?.window_minutes ?? 0),
    pctUsed,
    pctLeft,
    ...formatReset(Number(limit?.resets_at ?? 0), timeZone),
  };
}

export function parseRateLimitPayload(rateLimits, options = {}) {
  const timeZone = options.timeZone ?? Intl.DateTimeFormat().resolvedOptions().timeZone;

  return {
    planType: rateLimits.plan_type ?? null,
    credits: rateLimits.credits ?? null,
    rateLimit: {
      shortWindow: normalizeWindow(rateLimits.primary, '5h limit', timeZone),
      longWindow: normalizeWindow(rateLimits.secondary, 'Weekly limit', timeZone),
    },
  };
}

export function getLatestCodexSessionUsage(options = {}) {
  const sessionsDir = options.sessionsDir ?? DEFAULT_SESSIONS_DIR;
  const timeZone = options.timeZone ?? Intl.DateTimeFormat().resolvedOptions().timeZone;

  try {
    const files = walkSessionFiles(sessionsDir).sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
    if (!files.length) {
      return { available: false, reason: 'No Codex session files found', sessionsDir };
    }

    for (const filePath of files) {
      const lines = readFileSync(filePath, 'utf8').split('\n').filter(Boolean);
      for (let index = lines.length - 1; index >= 0; index -= 1) {
        try {
          const event = JSON.parse(lines[index]);
          const rateLimits = event?.payload?.rate_limits;
          if (!rateLimits?.primary || !rateLimits?.secondary) continue;

          const normalized = parseRateLimitPayload(rateLimits, { timeZone });
          return {
            available: true,
            source: 'codex-session-file',
            timeZone,
            sessionFile: filePath,
            eventTimestamp: event.timestamp ?? null,
            ...normalized,
            shortWindow: normalized.rateLimit.shortWindow,
            longWindow: normalized.rateLimit.longWindow,
          };
        } catch {
          // Skip malformed lines.
        }
      }
    }

    return {
      available: false,
      reason: 'No rate limit payload found in Codex session files',
      sessionsDir,
    };
  } catch (error) {
    return {
      available: false,
      reason: error.message,
      sessionsDir,
    };
  }
}
