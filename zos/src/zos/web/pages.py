"""Minimal server-rendered HTML pages for owner OAuth approval and Google connect."""

from __future__ import annotations

import html
from urllib.parse import urlparse

from zos.web.theme import CSS as _CSS


def render_authorize_page(
    *,
    client_name: str,
    redirect_uri: str,
    scopes: list[str],
    hidden_fields: dict[str, str],
    error: str | None = None,
) -> str:
    """Render minimal HTML form for owner passphrase sign-in and OAuth client approval."""
    hidden_html = "\n".join(
        f'<input type="hidden" name="{html.escape(k, quote=True)}" value="{html.escape(v, quote=True)}">'
        for k, v in hidden_fields.items()
    )
    error_html = (
        f'<div class="notice notice-err" role="alert">{html.escape(error)}</div>' if error else ""
    )
    scopes_text = html.escape(" ".join(scopes) if scopes else "(default)")
    parsed_uri = urlparse(redirect_uri)
    redirect_host = parsed_uri.netloc or parsed_uri.hostname or redirect_uri
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>ZOS — Authorize Client</title>
  <style>{_CSS}</style>
</head>
<body>
  <main class="narrow">
    <div class="card">
      <h1>Authorize MCP Client</h1>
      <p class="muted">An application is requesting access to your ZOS MCP server. Check the redirect host before approving.</p>
      {error_html}
      <dl class="kv">
        <dt>Client</dt>
        <dd>{html.escape(client_name)}</dd>
        <dt>Redirect Host</dt>
        <dd class="host-callout">{html.escape(redirect_host)}</dd>
        <dt>Full Redirect URI</dt>
        <dd>{html.escape(redirect_uri)}</dd>
        <dt>Scopes</dt>
        <dd>{scopes_text}</dd>
      </dl>
      <form class="auth-form" method="post" action="/oauth/authorize">
        {hidden_html}
        <label class="field" for="passphrase">Owner passphrase
          <input id="passphrase" name="passphrase" type="password" autocomplete="current-password" autofocus>
        </label>
        <div class="auth-actions">
          <button class="primary" type="submit" name="decision" value="approve">Approve</button>
          <button class="secondary" type="submit" name="decision" value="deny">Deny</button>
        </div>
      </form>
    </div>
  </main>
</body>
</html>
"""


def render_google_connect_page(*, error: str | None = None, status_message: str | None = None) -> str:
    """Render minimal HTML form for owner passphrase verification before connecting Google."""
    error_html = (
        f'<div class="notice notice-err" role="alert">{html.escape(error)}</div>' if error else ""
    )
    status_html = (
        f'<div class="notice notice-ok" role="status">{html.escape(status_message)}</div>'
        if status_message
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>ZOS — Connect Google</title>
  <style>{_CSS}</style>
</head>
<body>
  <main class="narrow">
    <div class="card">
      <h1>Connect Google</h1>
      <p class="muted">Enter your ZOS owner passphrase to authorize Google Calendar and Gmail access.</p>
      {error_html}
      {status_html}
      <form class="auth-form" method="post" action="/connect/google">
        <label class="field" for="passphrase">Owner passphrase
          <input id="passphrase" name="passphrase" type="password" autocomplete="current-password" required autofocus>
        </label>
        <button type="submit" class="block">Continue to Google</button>
      </form>
    </div>
  </main>
</body>
</html>
"""


def render_upstream_connect_page(
    *,
    slug: str,
    upstream_url: str,
    error: str | None = None,
    status_message: str | None = None,
    connected: bool = False,
) -> str:
    """Render mobile-friendly owner passphrase gate for `/connect/<slug>` OAuth flow."""
    error_html = (
        f'<div class="notice notice-err" role="alert">{html.escape(error)}</div>' if error else ""
    )
    status_html = (
        f'<div class="notice notice-ok" role="status">{html.escape(status_message)}</div>'
        if status_message
        else ""
    )
    parsed = urlparse(upstream_url)
    host = parsed.netloc or parsed.hostname or upstream_url
    form_html = (
        '<p><a class="btn block" href="/admin/upstreams">Open MCP Servers in Admin</a></p>'
        if connected
        else f"""
    <form class="auth-form" method="post" action="/connect/{html.escape(slug, quote=True)}">
      <label class="field" for="passphrase">Owner passphrase
        <input id="passphrase" name="passphrase" type="password" autocomplete="current-password" required autofocus>
      </label>
      <button type="submit" class="block">Continue to {html.escape(slug)}</button>
    </form>"""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>ZOS — Connect {html.escape(slug)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <main class="narrow">
    <div class="card">
      <h1>Connect MCP Server: <code>{html.escape(slug)}</code></h1>
      <div class="meta-box"><strong>Host:</strong> {html.escape(host)}<br><strong>URL:</strong> {html.escape(upstream_url)}</div>
      {error_html}
      {status_html}
      {form_html}
    </div>
  </main>
</body>
</html>
"""
