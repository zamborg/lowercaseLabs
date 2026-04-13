# Codex Status Parser

Temporary scratch module for parsing the `codex` `/status` card into a structured object.

It is intentionally split into two steps:

1. `extractLatestCodexStatusBlock(rawText)` isolates the last Codex status card from raw tmux or terminal output.
2. `parseCodexStatusBlock(blockText)` turns that card into normalized fields, including `rateLimit.shortWindow` and `rateLimit.longWindow`.

Combined helper:

```js
import { parseCodexStatusText } from './src/index.mjs';

const parsed = parseCodexStatusText(rawTerminalText);
```

tmux helper:

```js
import { readCodexStatusFromTmux } from './src/index.mjs';

const parsed = await readCodexStatusFromTmux('my-session:0.0');
```

Current live helper:

```js
import { getLatestCodexSessionUsage } from './src/index.mjs';

const usage = getLatestCodexSessionUsage();
```

That reads the newest local `~/.codex/sessions/**/*.jsonl` file containing `payload.rate_limits` and returns:

- `shortWindow.pctLeft`
- `shortWindow.resetTime`
- `shortWindow.resetDate`
- `longWindow.pctLeft`
- `longWindow.resetTime`
- `longWindow.resetDate`

Claude tmux helper:

```js
import { readClaudeUsageFromTmux } from './src/index.mjs';

const usage = await readClaudeUsageFromTmux('cc_test:0.0');
```

That does:

1. clear the Claude prompt input
2. send `/usage`
3. wait for the modal
4. capture the pane
5. send `Esc` to dismiss the modal

The return shape is designed to be easy to adapt into the scheduler's existing `getCodexUsage()` contract.

Run tests:

```bash
node --test dump/codex-status-parser/test/*.test.mjs
```

Run the temp dashboard:

```bash
node dump/codex-status-parser/server.mjs
```
