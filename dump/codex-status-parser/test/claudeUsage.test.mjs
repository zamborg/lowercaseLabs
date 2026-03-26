import test from 'node:test';
import assert from 'node:assert/strict';
import { parseClaudeUsageText } from '../src/index.mjs';

test('parseClaudeUsageText recognizes loading state', () => {
  const parsed = parseClaudeUsageText(`
  Settings:  Status   Config   Usage

  Loading usage data…

  Esc to cancel
  `);

  assert.equal(parsed.available, false);
  assert.equal(parsed.state, 'loading');
});

test('parseClaudeUsageText recognizes subscription gating', () => {
  const parsed = parseClaudeUsageText(`
  Status   Config   Usage

  /usage is only available for subscription plans.

  Esc to cancel
  `);

  assert.equal(parsed.available, false);
  assert.equal(parsed.state, 'subscription_required');
});

test('parseClaudeUsageText extracts short and long windows when present', () => {
  const parsed = parseClaudeUsageText(`
  Status   Config   Usage

  5h limit: 92% left (resets 02:57 on 24 Mar)
  Weekly limit: 81% left (resets 11:18 on 28 Mar)

  Esc to cancel
  `);

  assert.equal(parsed.available, true);
  assert.equal(parsed.shortWindow?.pctLeft, 92);
  assert.equal(parsed.shortWindow?.resetTime, '02:57');
  assert.equal(parsed.longWindow?.pctLeft, 81);
  assert.equal(parsed.longWindow?.resetDate, '28 Mar');
});

test('parseClaudeUsageText recognizes slash-command passthrough', () => {
  const parsed = parseClaudeUsageText(`
  ❯ /usage

  ✢ Levitating… (thinking with medium effort)
  `);

  assert.equal(parsed.available, false);
  assert.equal(parsed.state, 'command_passthrough');
});

test('parseClaudeUsageText recognizes session usage tables', () => {
  const parsed = parseClaudeUsageText(`
  ❯ /usage

  ⏺ Usage for this session:

  ┌────────────────────┬───────────────────┐
  │       Metric       │       Value       │
  ├────────────────────┼───────────────────┤
  │ Model              │ claude-sonnet-4-6 │
  ├────────────────────┼───────────────────┤
  │ Input tokens       │ ~9,500            │
  ├────────────────────┼───────────────────┤
  │ Output tokens      │ ~50               │
  └────────────────────┴───────────────────┘
  `);

  assert.equal(parsed.available, false);
  assert.equal(parsed.state, 'session_usage_only');
  assert.equal(parsed.sessionUsage.model, 'claude-sonnet-4-6');
  assert.equal(parsed.sessionUsage['input tokens'], '~9,500');
});

test('parseClaudeUsageText recognizes unavailable slash command output', () => {
  const parsed = parseClaudeUsageText(`
  ❯ /usage

  ⏺ /usage isn't a recognized skill. Available skills are:
  `);

  assert.equal(parsed.available, false);
  assert.equal(parsed.state, 'slash_command_unavailable');
});

test('parseClaudeUsageText extracts claude usage dialog percentages', () => {
  const parsed = parseClaudeUsageText(`
  Status   Config   Usage

  Current session
  █▌                                                 3% used
  Resets 1:59am (America/Los_Angeles)

  Current week (all models)
  ████                                               8% used
  Resets Mar 28 at 7pm (America/Los_Angeles)
  `);

  assert.equal(parsed.available, true);
  assert.equal(parsed.shortWindow?.pctUsed, 3);
  assert.equal(parsed.shortWindow?.pctLeft, 97);
  assert.equal(parsed.shortWindow?.resetTime, '1:59am');
  assert.equal(parsed.longWindow?.pctUsed, 8);
  assert.equal(parsed.longWindow?.pctLeft, 92);
  assert.equal(parsed.longWindow?.resetDate, 'Mar 28');
  assert.equal(parsed.longWindow?.resetTime, '7pm');
});
