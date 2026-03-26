const ANSI_CSI_RE = /\u001b\[[0-?]*[ -/]*[@-~]/g;
const ANSI_OSC_RE = /\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)/g;
const ANSI_SINGLE_RE = /\u001b[@-_]/g;

const LIMIT_LABELS = {
  '5h limit': 'shortWindow',
  'Weekly limit': 'longWindow',
};

export function stripTerminalControl(text) {
  return String(text ?? '')
    .replace(/\r/g, '')
    .replace(ANSI_OSC_RE, '')
    .replace(ANSI_CSI_RE, '')
    .replace(ANSI_SINGLE_RE, '');
}

export function extractLatestCodexStatusBlock(rawText) {
  const text = stripTerminalControl(rawText);
  const lines = text.split('\n');
  const blocks = [];
  let start = -1;

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes('╭')) {
      start = i;
      continue;
    }

    if (start >= 0 && lines[i].includes('╰')) {
      blocks.push(lines.slice(start, i + 1).join('\n'));
      start = -1;
    }
  }

  return [...blocks].reverse().find(block => block.includes('OpenAI Codex')) ?? null;
}

function unwrapBoxLine(line) {
  const trimmed = line.trimEnd();

  if (/^\s*[╭╰].*[╮╯]\s*$/.test(trimmed)) return '';

  const boxed = trimmed.match(/^\s*│\s?(.*?)\s*│\s*$/u);
  if (boxed) return boxed[1].trimEnd();

  return trimmed.trim();
}

function parseHeader(line) {
  const match = line.match(/^>_\s+OpenAI Codex\s+\(([^)]+)\)$/);
  if (!match) return null;
  return {
    product: 'OpenAI Codex',
    version: match[1],
    raw: line,
  };
}

function parseModel(raw) {
  const match = raw.match(/^(.*?)\s+\((.*)\)$/);
  if (!match) return { raw, name: raw };

  const [, name, settingsRaw] = match;
  const settings = {};

  for (const chunk of settingsRaw.split(/\s*,\s*/)) {
    const settingMatch = chunk.match(/^([a-zA-Z][a-zA-Z ]*?)\s+(.+)$/);
    if (!settingMatch) continue;
    const key = camelCase(settingMatch[1]);
    settings[key] = settingMatch[2];
  }

  return {
    raw,
    name,
    settings,
  };
}

function parseAccount(raw) {
  const match = raw.match(/^(.*?)\s+\(([^)]+)\)$/);
  if (!match) return { raw };
  return {
    raw,
    email: match[1],
    plan: match[2],
  };
}

function parseLimit(raw, label) {
  const match = raw.match(/^\[([^\]]+)\]\s+(\d+)%\s+left\s+\(resets\s+(.+)\)$/u);
  if (!match) {
    return {
      label,
      raw,
      pctLeft: null,
      reset: null,
      bar: null,
    };
  }

  const [, barText, pctLeftText, resetRaw] = match;
  const glyphs = [...barText].filter(char => !/\s/.test(char));
  const filled = glyphs.filter(char => char !== '░').length;
  const total = glyphs.length;
  const resetMatch = resetRaw.match(/^(\d{1,2}:\d{2})\s+on\s+(.+)$/);

  return {
    label,
    raw,
    pctLeft: Number.parseInt(pctLeftText, 10),
    reset: resetRaw,
    resetTime: resetMatch?.[1] ?? null,
    resetDate: resetMatch?.[2] ?? null,
    bar: {
      raw: barText,
      filled,
      total,
    },
  };
}

function camelCase(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+([a-z0-9])/g, (_, char) => char.toUpperCase());
}

export function parseCodexStatusBlock(blockText) {
  const lines = stripTerminalControl(blockText)
    .split('\n')
    .map(unwrapBoxLine)
    .filter(Boolean);

  const normalizedLines = lines.map(line => line.trim());
  const headerLine = normalizedLines.find(line => line.includes('OpenAI Codex'));
  const header = headerLine ? parseHeader(headerLine) : null;
  const status = {
    source: 'codex-status',
    rawBlock: stripTerminalControl(blockText),
    header,
    notes: [],
    metadata: {},
    rateLimit: {},
  };

  for (const rawLine of normalizedLines) {
    const line = rawLine.trim();
    if (line === headerLine) continue;
    if (line.startsWith('Visit ')) {
      status.notes.push(line);
      continue;
    }

    const rowMatch = line.match(/^(.+?):\s{2,}(.*)$/);
    if (!rowMatch) continue;

    const [, label, value] = rowMatch;
    if (LIMIT_LABELS[label]) {
      status.rateLimit[LIMIT_LABELS[label]] = parseLimit(value, label);
      continue;
    }

    switch (label) {
      case 'Model':
        status.model = parseModel(value);
        break;
      case 'Directory':
        status.directory = value;
        break;
      case 'Permissions':
        status.permissions = value;
        break;
      case 'Agents.md':
        status.agentsMd = value;
        break;
      case 'Account':
        status.account = parseAccount(value);
        break;
      case 'Collaboration mode':
        status.collaborationMode = value;
        break;
      case 'Session':
        status.session = value;
        break;
      default:
        status.metadata[camelCase(label)] = value;
        break;
    }
  }

  status.shortWindow = status.rateLimit.shortWindow ?? null;
  status.longWindow = status.rateLimit.longWindow ?? null;

  return status;
}

export function parseCodexStatusText(rawText) {
  const block = extractLatestCodexStatusBlock(rawText);
  if (!block) {
    return {
      available: false,
      reason: 'No Codex status block found in input',
    };
  }

  return {
    available: true,
    ...parseCodexStatusBlock(block),
  };
}
