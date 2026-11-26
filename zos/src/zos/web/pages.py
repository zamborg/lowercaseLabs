"""Server-rendered auth pages: owner sign-in shell, OAuth approval, connector connect."""

from __future__ import annotations

import html
from urllib.parse import urlparse

from zos.web.theme import BRAND_MARK
from zos.web.theme import CSS as _CSS


def render_auth_shell(*, title: str, card: str) -> str:
    """Centered single-card page used by every unauthenticated owner flow."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{html.escape(title)} · zos</title>
  <style>{_CSS}</style>
</head>
<body>
  <main class="page narrow">
    <div class="auth-brand"><span class="brand">{BRAND_MARK}zos</span></div>
    <div class="auth-card">{card}</div>
  </main>
</body>
</html>
"""


def render_authorize_page(
    *,
    client_name: str,
    redirect_uri: str,
    scopes: list[str],
    hidden_fields: dict[str, str],
    error: str | None = None,
) -> str:
    """Render owner passphrase sign-in and OAuth client approval."""
    hidden_html = "\n".join(
        f'<input type="hidden" name="{html.escape(k, quote=True)}" value="{html.escape(v, quote=True)}">'
        for k, v in hidden_fields.items()
    )
    error_html = (
        f'<div class="notice notice-err" role="alert"><div>{html.escape(error)}</div></div>' if error else ""
    )
    scopes_text = html.escape(" ".join(scopes) if scopes else "(default)")
    parsed_uri = urlparse(redirect_uri)
    redirect_host = parsed_uri.netloc or parsed_uri.hostname or redirect_uri
    card = f"""
      <h1>Authorize access</h1>
      <p class="lede"><strong>{html.escape(client_name)}</strong> wants to connect to your zos. Approve only if you started this and recognise the redirect host.</p>
      {error_html}
      <dl class="kv">
        <dt>Redirect host</dt>
        <dd class="host-callout">{html.escape(redirect_host)}</dd>
        <dt>Client</dt>
        <dd>{html.escape(client_name)}</dd>
        <dt>Full redirect URI</dt>
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
          <button class="secondary" type="submit" name="decision" value="deny">Deny</button>
          <button type="submit" name="decision" value="approve">Approve</button>
        </div>
      </form>"""
    return render_auth_shell(title="Authorize", card=card)


def render_upstream_connect_page(
    *,
    slug: str,
    upstream_url: str,
    error: str | None = None,
    status_message: str | None = None,
    connected: bool = False,
) -> str:
    """Render owner passphrase gate for the `/connect/<slug>` OAuth flow."""
    error_html = (
        f'<div class="notice notice-err" role="alert"><div>{html.escape(error)}</div></div>' if error else ""
    )
    status_html = (
        f'<div class="notice notice-ok" role="status"><div>{html.escape(status_message)}</div></div>'
        if status_message
        else ""
    )
    parsed = urlparse(upstream_url)
    host = parsed.netloc or parsed.hostname or upstream_url
    form_html = (
        '<a class="btn block" href="/admin/upstreams">Back to MCP servers</a>'
        if connected
        else f"""
      <form class="auth-form" method="post" action="/connect/{html.escape(slug, quote=True)}">
        <label class="field" for="passphrase">Owner passphrase
          <input id="passphrase" name="passphrase" type="password" autocomplete="current-password" required autofocus>
        </label>
        <button type="submit" class="block">Continue to {html.escape(host)}</button>
      </form>"""
    )
    card = f"""
      <h1>Connect <code>{html.escape(slug)}</code></h1>
      <p class="lede">You'll be sent to the server to sign in, then returned here.</p>
      {error_html}
      {status_html}
      <dl class="kv">
        <dt>Host</dt>
        <dd>{html.escape(host)}</dd>
        <dt>URL</dt>
        <dd>{html.escape(upstream_url)}</dd>
      </dl>
      {form_html}"""
    return render_auth_shell(title=f"Connect {slug}", card=card)
