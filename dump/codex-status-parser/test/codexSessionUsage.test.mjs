import test from 'node:test';
import assert from 'node:assert/strict';
import { getLatestCodexSessionUsage, parseRateLimitPayload } from '../src/index.mjs';

test('parseRateLimitPayload converts used percent to left percent and reset labels', () => {
  const parsed = parseRateLimitPayload({
    primary: { used_percent: 3, window_minutes: 300, resets_at: 1774346242 },
    secondary: { used_percent: 16, window_minutes: 10080, resets_at: 1774721883 },
    credits: null,
    plan_type: 'plus',
  }, { timeZone: 'America/Los_Angeles' });

  assert.equal(parsed.planType, 'plus');
  assert.equal(parsed.rateLimit.shortWindow.pctUsed, 3);
  assert.equal(parsed.rateLimit.shortWindow.pctLeft, 97);
  assert.equal(parsed.rateLimit.shortWindow.windowMinutes, 300);
  assert.equal(parsed.rateLimit.longWindow.pctLeft, 84);
  assert.match(parsed.rateLimit.shortWindow.resetLabel, /^\d{2}:\d{2} on \d{2} \w{3}$/);
});

test('getLatestCodexSessionUsage returns an unavailable result for missing directories', () => {
  const parsed = getLatestCodexSessionUsage({ sessionsDir: '/tmp/definitely-not-a-real-codex-dir' });
  assert.equal(parsed.available, false);
});
