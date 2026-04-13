const POLL_MS = 60000;
const state = {
  codex: null,
  claude: null,
  lastUpdated: null,
};

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function setButtonBusy(id, busy, busyLabel) {
  const button = document.getElementById(id);
  if (!button) return;
  if (!button.dataset.defaultLabel) button.dataset.defaultLabel = button.textContent;
  button.disabled = busy;
  button.textContent = busy ? busyLabel : button.dataset.defaultLabel;
}

function setPercent(id, value) {
  document.getElementById(id).style.width = `${value}%`;
}

function renderWindow(prefix, windowData) {
  setText(`${prefix}-pct`, `${windowData.pctLeft}%`);
  setPercent(`${prefix}-fill`, windowData.pctLeft);
  setText(`${prefix}-reset`, `resets ${windowData.resetTime} on ${windowData.resetDate}`);
}

function renderUnavailableWindow(prefix, label = '--') {
  setText(`${prefix}-pct`, '--%');
  setPercent(`${prefix}-fill`, 0);
  setText(`${prefix}-reset`, label);
}

function renderCodex(codex) {
  setText('codex-state', codex?.available ? 'Live' : (codex?.reason ?? 'Unavailable'));
  if (codex?.available) {
    renderWindow('short', codex.shortWindow);
    renderWindow('long', codex.longWindow);
    setText('plan-type', codex.planType ?? '--');
    setText('short-window', `${codex.shortWindow.windowMinutes} min`);
    setText('long-window', `${codex.longWindow.windowMinutes} min`);
    setText('event-ts', codex.eventTimestamp ?? '--');
  } else {
    renderUnavailableWindow('short', 'reset --');
    renderUnavailableWindow('long', 'reset --');
    setText('plan-type', '--');
    setText('short-window', '--');
    setText('long-window', '--');
    setText('event-ts', '--');
  }
}

function renderClaude(claude) {
  setText('claude-target', claude?.target ?? '--');
  setText('claude-state', claude?.available ? 'Captured' : (claude?.state ?? 'Idle'));
  setText('claude-state-detail', claude?.state ?? '--');
  setText('claude-reason', claude?.reason ?? '--');

  if (claude?.available) {
    renderWindow('claude-short', claude.shortWindow);
    renderWindow('claude-long', claude.longWindow);
  } else {
    renderUnavailableWindow('claude-short', claude?.reason ?? 'reset --');
    renderUnavailableWindow('claude-long', claude?.reason ?? 'reset --');
  }
}

function renderRaw() {
  setText('raw-json', JSON.stringify({
    fetchedAt: state.lastUpdated,
    codex: state.codex,
    claude: state.claude,
  }, null, 2));
}

async function refreshCodex() {
  setButtonBusy('codex-refresh-btn', true, 'Refreshing…');
  try {
    const response = await fetch(`/api/codex-usage?ts=${Date.now()}`, { cache: 'no-store' });
    const payload = await response.json();
    state.codex = payload;
    state.lastUpdated = payload.fetchedAt;
    setText('last-updated', `Codex fetched ${new Date(payload.fetchedAt).toLocaleTimeString()}`);
    setText('codex-last-checked', `Last checked ${new Date(payload.fetchedAt).toLocaleTimeString()}`);
    setText('source-file', payload.sessionFile ?? '');
    renderCodex(payload);
    renderRaw();
  } finally {
    setButtonBusy('codex-refresh-btn', false, 'Refreshing…');
  }
}

async function refreshClaude() {
  setButtonBusy('claude-check-btn', true, 'Checking Claude…');
  setText('claude-state', 'Checking…');

  try {
    const response = await fetch(`/api/claude-usage?mode=fresh&ts=${Date.now()}`, { cache: 'no-store' });
    const payload = await response.json();
    state.claude = payload;
    setText('claude-last-checked', `Last checked ${new Date(payload.fetchedAt).toLocaleTimeString()}`);
    renderClaude(payload);
    renderRaw();
  } catch (error) {
    const failed = {
      available: false,
      state: 'error',
      reason: String(error),
    };
    state.claude = failed;
    setText('claude-last-checked', 'Claude check failed');
    renderClaude(failed);
    renderRaw();
  } finally {
    setButtonBusy('claude-check-btn', false, 'Checking Claude…');
  }
}

document.getElementById('claude-check-btn').addEventListener('click', () => {
  refreshClaude();
});

document.getElementById('codex-refresh-btn').addEventListener('click', () => {
  refreshCodex().catch(error => {
    setText('raw-json', String(error));
  });
});

refreshCodex().catch(error => {
  setText('raw-json', String(error));
});

renderClaude({
  available: false,
  state: 'idle',
  reason: 'Manual check only',
});
renderRaw();

setInterval(() => {
  refreshCodex().catch(error => {
    setText('raw-json', String(error));
  });
}, POLL_MS);
