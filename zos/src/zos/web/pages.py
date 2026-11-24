"""Minimal server-rendered HTML pages for owner OAuth approval and Google connect."""

from __future__ import annotations

import html
from urllib.parse import urlparse


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
        f'<p style="color:#b91c1c;font-weight:600;">{html.escape(error)}</p>' if error else ""
    )
    scopes_text = html.escape(" ".join(scopes) if scopes else "(default)")
    parsed_uri = urlparse(redirect_uri)
    redirect_host = parsed_uri.netloc or parsed_uri.hostname or redirect_uri
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ZOS — Authorize Client</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 520px; margin: 3rem auto; padding: 0 1rem; color: #111; }}
    .card {{ border: 1px solid #d4d4d8; border-radius: 8px; padding: 1.5rem; }}
    dt {{ font-weight: 600; margin-top: 0.75rem; }}
    dd {{ margin: 0.25rem 0 0; word-break: break-all; font-family: ui-monospace, monospace; font-size: 0.9rem; }}
    .host-callout {{ background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; padding: 0.55rem 0.75rem; font-size: 1.05rem; font-weight: 700; color: #78350f; }}
    input[type=password] {{ width: 100%; padding: 0.5rem; margin: 0.5rem 0 1rem; box-sizing: border-box; }}
    .actions {{ display: flex; gap: 0.75rem; }}
    button {{ padding: 0.55rem 1rem; border-radius: 6px; border: 1px solid #18181b; cursor: pointer; }}
    button.primary {{ background: #18181b; color: #fff; }}
    button.secondary {{ background: #fff; color: #18181b; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Authorize MCP Client</h1>
    <p>An application is requesting access to your ZOS MCP server.</p>
    {error_html}
    <dl>
      <dt>Client</dt>
      <dd>{html.escape(client_name)}</dd>
      <dt>Redirect Host</dt>
      <dd class="host-callout">{html.escape(redirect_host)}</dd>
      <dt>Full Redirect URI</dt>
      <dd>{html.escape(redirect_uri)}</dd>
      <dt>Scopes</dt>
      <dd>{scopes_text}</dd>
    </dl>
    <form method="post" action="/oauth/authorize" style="margin-top:1.25rem;">
      {hidden_html}
      <label for="passphrase"><strong>Owner passphrase</strong></label>
      <input id="passphrase" name="passphrase" type="password" autocomplete="current-password" autofocus>
      <div class="actions">
        <button class="primary" type="submit" name="decision" value="approve">Approve</button>
        <button class="secondary" type="submit" name="decision" value="deny">Deny</button>
      </div>
    </form>
  </div>
</body>
</html>
"""


def render_google_connect_page(*, error: str | None = None, status_message: str | None = None) -> str:
    """Render minimal HTML form for owner passphrase verification before connecting Google."""
    error_html = (
        f'<p style="color:#b91c1c;font-weight:600;">{html.escape(error)}</p>' if error else ""
    )
    status_html = (
        f'<p style="color:#15803d;font-weight:600;">{html.escape(status_message)}</p>'
        if status_message
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ZOS — Connect Google</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 520px; margin: 3rem auto; padding: 0 1rem; color: #111; }}
    .card {{ border: 1px solid #d4d4d8; border-radius: 8px; padding: 1.5rem; }}
    input[type=password] {{ width: 100%; padding: 0.5rem; margin: 0.5rem 0 1rem; box-sizing: border-box; }}
    button {{ padding: 0.55rem 1rem; border-radius: 6px; border: 1px solid #18181b; background: #18181b; color: #fff; cursor: pointer; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Connect Google (Calendar + Gmail)</h1>
    <p>Enter your ZOS owner passphrase to authorize Google Calendar and Gmail access.</p>
    {error_html}
    {status_html}
    <form method="post" action="/connect/google">
      <label for="passphrase"><strong>Owner passphrase</strong></label>
      <input id="passphrase" name="passphrase" type="password" autocomplete="current-password" required autofocus>
      <button type="submit">Continue to Google</button>
    </form>
  </div>
</body>
</html>
"""
