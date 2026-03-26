import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  extractLatestCodexStatusBlock,
  parseCodexStatusBlock,
  parseCodexStatusText,
  stripTerminalControl,
} from '../src/index.mjs';

const fixture = readFileSync(
  join(import.meta.dirname, '../fixtures/sample-status.txt'),
  'utf8'
);

test('stripTerminalControl removes ANSI and OSC sequences', () => {
  const noisy = `hello \u001b[31mred\u001b[0m \u001b]11;rgb:158e/193a/1e75\u001b\\ world`;
  assert.equal(stripTerminalControl(noisy), 'hello red  world');
});

test('extractLatestCodexStatusBlock finds the codex card', () => {
  const combined = `${fixture}\n\nnot this one\n╭─╮\n│ x │\n╰─╯`;
  const block = extractLatestCodexStatusBlock(combined);
  assert.ok(block);
  assert.match(block, /OpenAI Codex/);
});

test('parseCodexStatusBlock parses the scheduler-relevant fields', () => {
  const block = extractLatestCodexStatusBlock(fixture);
  const parsed = parseCodexStatusBlock(block);

  assert.equal(parsed.header.version, 'v0.116.0');
  assert.equal(parsed.model.name, 'gpt-5.4');
  assert.equal(parsed.model.settings.reasoning, 'high');
  assert.equal(parsed.model.settings.summaries, 'auto');
  assert.equal(parsed.account.email, 'zubinaysola@gmail.com');
  assert.equal(parsed.account.plan, 'Plus');
  assert.equal(parsed.shortWindow.pctLeft, 100);
  assert.equal(parsed.shortWindow.resetTime, '02:57');
  assert.equal(parsed.shortWindow.resetDate, '24 Mar');
  assert.equal(parsed.longWindow.pctLeft, 85);
  assert.equal(parsed.longWindow.bar.filled, 17);
  assert.equal(parsed.longWindow.bar.total, 20);
});

test('parseCodexStatusText survives ANSI junk inside the card', () => {
  const noisy = fixture.replace(
    '│  Account:              zubinaysola@gmail.com (Plus)                              │',
    '\u001b]11;rgb:158e/193a/1e75\u001b\\│  Account:              zubinaysola@gmail.com (Plus)                              │'
  );

  const parsed = parseCodexStatusText(noisy);
  assert.equal(parsed.available, true);
  assert.equal(parsed.account.email, 'zubinaysola@gmail.com');
  assert.equal(parsed.session, '019d1e35-3952-7740-b03f-b6224a666d69');
});

test('parseCodexStatusText returns an unavailable result when no card is present', () => {
  const parsed = parseCodexStatusText('nothing to see here');
  assert.equal(parsed.available, false);
  assert.match(parsed.reason, /No Codex status block/);
});
