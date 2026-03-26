import http from 'node:http';
import { readFileSync } from 'node:fs';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getLatestCodexSessionUsage, readClaudeUsageFromFreshSession, readClaudeUsageFromTmux } from './src/index.mjs';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const publicDir = join(__dirname, 'web');
const port = Number.parseInt(process.env.PORT ?? '3874', 10);

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
};

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, { 'content-type': 'application/json; charset=utf-8' });
  response.end(`${JSON.stringify(payload, null, 2)}\n`);
}

function serveStatic(response, fileName) {
  const path = join(publicDir, fileName);
  const body = readFileSync(path);
  response.writeHead(200, { 'content-type': contentTypes[extname(path)] ?? 'text/plain; charset=utf-8' });
  response.end(body);
}

const server = http.createServer((request, response) => {
  const url = new URL(request.url, `http://${request.headers.host ?? '127.0.0.1'}`);

  if (url.pathname === '/api/codex-usage') {
    return sendJson(response, 200, {
      fetchedAt: new Date().toISOString(),
      ...getLatestCodexSessionUsage(),
    });
  }

  if (url.pathname === '/api/claude-usage') {
    const target = url.searchParams.get('target') ?? process.env.CLAUDE_USAGE_TMUX_TARGET ?? 'cc_test:0.0';
    const mode = url.searchParams.get('mode') ?? 'fresh';
    const cwd = url.searchParams.get('cwd') ?? process.cwd();
    const reader = mode === 'attached'
      ? readClaudeUsageFromTmux(target)
      : readClaudeUsageFromFreshSession({ cwd });
    return Promise.resolve(reader)
      .then(payload => sendJson(response, 200, { fetchedAt: new Date().toISOString(), ...payload }))
      .catch(error => sendJson(response, 500, { error: error.message }));
  }

  if (url.pathname === '/' || url.pathname === '/index.html') {
    return serveStatic(response, 'index.html');
  }

  if (url.pathname === '/app.js') {
    return serveStatic(response, 'app.js');
  }

  if (url.pathname === '/styles.css') {
    return serveStatic(response, 'styles.css');
  }

  sendJson(response, 404, { error: 'Not found' });
});

server.listen(port, '127.0.0.1', () => {
  console.log(`codex-status-parser listening on http://127.0.0.1:${port}`);
});
