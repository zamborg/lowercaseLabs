"""Server-rendered HTML templates for ZOS owner admin dashboard (/admin)."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any
from urllib.parse import urlencode


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_ts(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(value)


_CSS = """
:root { --bg:#fafafa; --fg:#111827; --muted:#6b7280; --border:#e5e7eb; --card:#ffffff;
        --accent:#111827; --ok:#15803d; --warn:#b45309; --err:#b91c1c; --ok-bg:#dcfce7;
        --warn-bg:#fef3c7; --err-bg:#fee2e2; --mono:ui-monospace,SFMono-Regular,Menlo,monospace; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--fg); line-height: 1.45; }
header { background: var(--card); border-bottom: 1px solid var(--border); padding: 0.75rem 1.5rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
header .brand { font-weight: 700; font-size: 1rem; letter-spacing: 0.02em; }
nav { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
nav a { text-decoration: none; color: var(--fg); padding: 0.35rem 0.7rem; border-radius: 6px; font-size: 0.9rem; }
nav a.active { background: var(--accent); color: #fff; }
nav a:hover:not(.active) { background: #f3f4f6; }
main { max-width: 1180px; margin: 1.5rem auto; padding: 0 1.25rem 3rem; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.15rem 1.25rem; margin-bottom: 1.25rem; }
h1 { font-size: 1.35rem; margin: 0 0 1rem; }
h2 { font-size: 1.05rem; margin: 0 0 0.75rem; }
.stats { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
.stat { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 0.85rem 1.1rem; min-width: 170px; }
.stat .label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.stat .value { font-size: 1.45rem; font-weight: 700; margin-top: 0.2rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th, td { text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border); vertical-align: middle; }
th { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); background: #f9fafb; }
code, .mono { font-family: var(--mono); font-size: 0.82rem; }
.badge { display: inline-block; padding: 0.15rem 0.45rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
.badge-ok { background: var(--ok-bg); color: var(--ok); }
.badge-warn { background: var(--warn-bg); color: var(--warn); }
.badge-err { background: var(--err-bg); color: var(--err); }
.badge-muted { background: #f3f4f6; color: #374151; }
.notice { padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem; border: 1px solid; font-size: 0.9rem; }
.notice-ok { background: var(--ok-bg); border-color: #86efac; color: #14532d; }
.notice-err { background: var(--err-bg); border-color: #fca5a5; color: #7f1d1d; }
.notice-token { background: #fefce8; border-color: #fde047; color: #713f12; }
.token-box { margin-top: 0.5rem; padding: 0.55rem 0.75rem; background: #fff; border: 1px solid #eab308; border-radius: 4px; font-family: var(--mono); word-break: break-all; font-size: 0.85rem; }
form.inline { display: inline-flex; gap: 0.35rem; align-items: center; margin: 0; }
.form-row { display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: flex-end; }
label.field { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.82rem; font-weight: 600; }
input[type=text], input[type=password], select { padding: 0.4rem 0.55rem; border: 1px solid #d1d5db; border-radius: 5px; font-size: 0.88rem; background: #fff; }
button, a.btn { display: inline-block; text-decoration: none; padding: 0.38rem 0.75rem; border-radius: 5px; border: 1px solid var(--accent); background: var(--accent); color: #fff; font-size: 0.84rem; cursor: pointer; }
button.secondary, a.btn.secondary { background: #fff; color: var(--fg); border-color: #d1d5db; }
button.danger { background: #fff; color: var(--err); border-color: #fca5a5; }
button.sm { padding: 0.2rem 0.45rem; font-size: 0.76rem; }
.scroll-x { overflow-x: auto; }
.muted { color: var(--muted); font-size: 0.84rem; }
.cell-controls { display: flex; flex-direction: column; gap: 0.3rem; }
"""


def _layout(*, title: str, active: str, csrf_token: str, body: str) -> str:
    nav_items = [
        ("overview", "/admin", "Overview"),
        ("upstreams", "/admin/upstreams", "MCP Servers"),
        ("clients", "/admin/clients", "Clients"),
        ("policy", "/admin/policy", "Tools &amp; Policy"),
        ("activity", "/admin/activity", "Activity"),
    ]
    nav_html = "".join(
        f'<a href="{href}" class="{"active" if active == key else ""}">{label}</a>'
        for key, href, label in nav_items
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ZOS Admin — {_e(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <div class="brand">ZOS Control Panel</div>
    <nav>
      {nav_html}
      <form class="inline" method="post" action="/admin/logout">
        <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
        <button type="submit" class="secondary">Sign out</button>
      </form>
    </nav>
  </header>
  <main>{body}</main>
</body>
</html>
"""


def render_admin_login_page(*, error: str | None = None) -> str:
    err_html = f'<div class="notice notice-err">{_e(error)}</div>' if error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ZOS Admin — Sign In</title>
  <style>{_CSS}
    .login-wrap {{ max-width: 420px; margin: 5rem auto; padding: 0 1rem; }}
  </style>
</head>
<body>
  <div class="login-wrap">
    <div class="card">
      <h1>ZOS Owner Admin</h1>
      <p class="muted">Enter your owner passphrase to open the control panel.</p>
      {err_html}
      <form method="post" action="/admin/login">
        <label class="field" style="margin-bottom:0.85rem;">
          Owner passphrase
          <input name="passphrase" type="password" autocomplete="current-password" required autofocus>
        </label>
        <button type="submit">Sign in</button>
      </form>
    </div>
  </div>
</body>
</html>
"""


def _fmt_bytes(size: int | None) -> str:
    if size is None:
        return "—"
    val = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024.0 or unit == "GB":
            return f"{int(val)} B" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{size} B"


def render_overview_page(
    *,
    overview: dict[str, Any],
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
) -> str:
    banners = ""
    if error:
        banners += f'<div class="notice notice-err">{_e(error)}</div>'
    if notice:
        banners += f'<div class="notice notice-ok">{_e(notice)}</div>'

    providers = overview["providers"]
    online_count = sum(1 for p in providers if p["online"])
    prov_rows = []
    for p in providers:
        badge = (
            '<span class="badge badge-ok">online</span>'
            if p["online"]
            else '<span class="badge badge-muted">offline</span>'
        )
        cur_backend = p.get("backend")
        backend_cell = f"<code>{_e(cur_backend)}</code>" if cur_backend else "—"
        switchable = p.get("switchable_backends") or []
        if switchable:
            toggles = []
            for target in switchable:
                if target == cur_backend:
                    continue
                toggles.append(
                    f'<form class="inline" method="post" action="/admin/providers/backend" style="margin-left:0.5rem;">'
                    f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                    f'<input type="hidden" name="slug" value="{_e(p["namespace"])}">'
                    f'<input type="hidden" name="backend" value="{_e(target)}">'
                    f'<button type="submit" class="secondary sm">Switch to {_e(target)}</button></form>'
                )
            backend_cell += "".join(toggles)
        prov_rows.append(
            f"<tr>"
            f"<td><code>{_e(p['namespace'])}</code></td>"
            f"<td>{_e(p['kind'])}</td>"
            f"<td>{backend_cell}</td>"
            f"<td>{badge}</td>"
            f"<td>{_e(_fmt_ts(p['last_seen_at']))}</td>"
            f"</tr>"
        )

    cred = overview["google_credential"]
    if cred:
        scopes_html = " ".join(f'<span class="badge badge-muted">{_e(s.rsplit("/", 1)[-1])}</span>' for s in cred["scopes"])
        cred_html = f"""
        <p><span class="badge badge-ok">connected</span> &nbsp; <strong>{_e(cred['account'])}</strong></p>
        <p class="muted">Access token expires: <code>{_e(_fmt_ts(cred['access_expires_at']))}</code> &middot; Updated: <code>{_e(_fmt_ts(cred['updated_at']))}</code></p>
        <p style="margin:0.5rem 0 1rem;">{scopes_html}</p>
        <a class="btn secondary" href="/connect/google">Reconnect Google</a>
        """
    else:
        cred_html = """
        <p><span class="badge badge-muted">not connected</span></p>
        <p class="muted">Connect Google once to enable both <code>gcal</code> (Calendar MCP) and <code>gmail</code> (REST).</p>
        <a class="btn" href="/connect/google">Connect Google</a>
        """

    last_backup = overview.get("last_backup")
    if last_backup:
        backup_ts = _fmt_ts(last_backup.get("completed_at"))
        backup_size = _fmt_bytes(last_backup.get("size_bytes"))
        backup_val = f'{_e(backup_size)}<div class="muted" style="font-size:0.78rem;font-weight:400;margin-top:0.2rem;">{_e(backup_ts)}</div>'
    else:
        backup_val = '<span class="muted" style="font-size:1rem;">Never</span>'

    body = f"""
    <h1>Overview</h1>
    {banners}
    <div class="stats">
      <div class="stat"><div class="label">Active MCP Sessions</div><div class="value">{int(overview['active_sessions'])}</div></div>
      <div class="stat"><div class="label">Online Providers</div><div class="value">{online_count} / {len(providers)}</div></div>
      <div class="stat"><div class="label">Last Backup</div><div class="value">{backup_val}</div></div>
    </div>
    <div class="card">
      <h2>Providers</h2>
      <div class="scroll-x">
        <table>
          <thead><tr><th>Namespace</th><th>Kind</th><th>Backend</th><th>Status</th><th>Last Seen</th></tr></thead>
          <tbody>{''.join(prov_rows) or '<tr><td colspan="5" class="muted">No providers registered.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Google Credential</h2>
      {cred_html}
    </div>
    """
    return _layout(title="Overview", active="overview", csrf_token=csrf_token, body=body)


def render_upstreams_page(
    *,
    upstreams: list[dict[str, Any]],
    csrf_token: str,
    catalog: list[dict[str, Any]] | None = None,
    notice: str | None = None,
    error: str | None = None,
) -> str:
    banners = ""
    if error:
        banners += f'<div class="notice notice-err">{_e(error)}</div>'
    if notice:
        banners += f'<div class="notice notice-ok">{_e(notice)}</div>'

    status_badges = {
        "online": '<span class="badge badge-ok">online</span>',
        "needs_reconnect": '<span class="badge badge-err">needs_reconnect</span>',
        "not_configured": '<span class="badge badge-warn">not_configured</span>',
        "offline": '<span class="badge badge-muted">offline</span>',
    }

    installed_slugs = {str(u["slug"]) for u in upstreams}
    installed_catalog_slugs = {str(u["catalog_slug"]) for u in upstreams if u.get("catalog_slug")}

    rows = []
    for u in upstreams:
        slug = str(u["slug"])
        status = str(u["status"])
        badge = status_badges.get(status, f'<span class="badge badge-muted">{_e(status)}</span>')
        err_note = (
            f'<div class="muted" style="color:var(--err);max-width:260px;word-break:break-word;">{_e(u["last_error"])}</div>'
            if u.get("last_error")
            else ""
        )
        help_note = ""
        if u["auth_mode"] == "bearer" and u.get("token_help") and not u.get("has_credential"):
            help_note = f'<div class="muted" style="max-width:300px;margin-top:0.25rem;">{_e(u["token_help"])}</div>'
        elif u["auth_mode"] == "oauth" and u.get("requires_client_registration") and u.get("client_help"):
            help_note = f'<div class="muted" style="max-width:300px;margin-top:0.25rem;">{_e(u["client_help"])}</div>'

        actions: list[str] = []
        if u["auth_mode"] == "bearer":
            actions.append(
                f'<form class="inline" method="post" action="/admin/upstreams/token">'
                f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                f'<input type="hidden" name="slug" value="{_e(slug)}">'
                f'<input type="password" name="token" placeholder="Paste token" autocomplete="new-password" required style="width:140px;">'
                f'<button type="submit" class="secondary sm">Set Token</button></form>'
            )
            if u.get("has_credential"):
                actions.append(
                    f'<form class="inline" method="post" action="/admin/upstreams/disconnect">'
                    f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                    f'<input type="hidden" name="slug" value="{_e(slug)}">'
                    f'<button type="submit" class="secondary sm">Disconnect</button></form>'
                )
        elif u["auth_mode"] == "oauth":
            if u.get("requires_client_registration"):
                actions.append(
                    f'<form class="inline" method="post" action="/admin/upstreams/oauth-client">'
                    f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                    f'<input type="hidden" name="slug" value="{_e(slug)}">'
                    f'<input type="text" name="client_id" placeholder="Client ID" required style="width:100px;">'
                    f'<input type="password" name="client_secret" placeholder="Client Secret" autocomplete="new-password" required style="width:110px;">'
                    f'<button type="submit" class="secondary sm">Set Client</button></form>'
                )
            label = "Reconnect" if status == "needs_reconnect" else "Connect"
            actions.append(f'<a class="btn sm" href="/connect/{_e(slug)}">{label}</a>')
            if u.get("has_credential") or u.get("online"):
                actions.append(
                    f'<form class="inline" method="post" action="/admin/upstreams/disconnect">'
                    f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                    f'<input type="hidden" name="slug" value="{_e(slug)}">'
                    f'<button type="submit" class="secondary sm">Disconnect</button></form>'
                )
        actions.append(
            f'<form class="inline" method="post" action="/admin/upstreams/remove">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f'<input type="hidden" name="slug" value="{_e(slug)}">'
            f'<button type="submit" class="danger sm">Remove</button></form>'
        )
        rows.append(
            f"<tr>"
            f'<td><code>{_e(slug)}</code><div class="muted mono">{_e(u["url"])}</div>{help_note}</td>'
            f'<td><code>{_e(u["auth_mode"])}</code></td>'
            f'<td><code>{_e(u["default_tier"])}</code></td>'
            f"<td>{badge}{err_note}</td>"
            f"<td>{int(u['tool_count'])}</td>"
            f'<td><div style="display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;">{"".join(actions)}</div></td>'
            f"</tr>"
        )

    cat_rows = []
    for c in catalog or []:
        c_slug = str(c["slug"])
        is_installed = c_slug in installed_slugs or c_slug in installed_catalog_slugs
        auth_label = f"{c['auth_mode']} (pre-reg)" if c.get("requires_client_registration") else c["auth_mode"]
        inst_badge = ' <span class="badge badge-ok">installed</span>' if is_installed else ""
        action_form = (
            f'<form class="inline" method="post" action="/admin/upstreams/install">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f'<input type="hidden" name="catalog_slug" value="{_e(c_slug)}">'
            f'<input type="text" name="as_slug" placeholder="alias (optional)" pattern="[a-z][a-z0-9]{{1,15}}" style="width:115px;">'
            f'<button type="submit" class="{"secondary" if is_installed else ""} sm">{"Install another" if is_installed else "Install"}</button></form>'
        )
        cat_rows.append(
            f"<tr>"
            f'<td><strong>{_e(c["name"])}</strong>{inst_badge} <code class="muted">{_e(c_slug)}</code>'
            f'<div class="muted">{_e(c["description"])}</div></td>'
            f'<td><code>{_e(auth_label)}</code></td>'
            f'<td><a href="{_e(c["docs_url"])}" target="_blank" rel="noopener noreferrer">Docs</a></td>'
            f"<td>{action_form}</td>"
            f"</tr>"
        )

    body = f"""
    <h1>MCP Servers</h1>
    {banners}
    <div class="card">
      <h2>Registered Remote MCP Servers</h2>
      <div class="scroll-x">
        <table>
          <thead><tr><th>Server</th><th>Auth</th><th>Default Tier</th><th>Status</th><th>Tools</th><th>Actions</th></tr></thead>
          <tbody>{''.join(rows) or '<tr><td colspan="6" class="muted">No remote MCP servers registered yet. Install one from the catalog below or add a custom URL.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Available Connectors</h2>
      <p class="muted">Verified official remote MCP servers. Nothing is installed by default; click <strong>Install</strong> to register with manifest tier rules, then connect.</p>
      <div class="scroll-x">
        <table>
          <thead><tr><th>Connector</th><th>Auth</th><th>Docs</th><th>Install</th></tr></thead>
          <tbody>{''.join(cat_rows) or '<tr><td colspan="4" class="muted">No catalog manifests found.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Add Custom Remote MCP Server</h2>
      <p class="muted">Tools are exposed as <code>&lt;slug&gt;_&lt;tool&gt;</code>. Default tier is <code>external</code> (approval required) unless opted down; override individual tools on <a href="/admin/policy">Tools &amp; Policy</a>.</p>
      <form class="form-row" method="post" action="/admin/upstreams/add">
        <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
        <label class="field">
          Slug (prefix)
          <input type="text" name="slug" placeholder="e.g. github" pattern="[a-z][a-z0-9]{{1,15}}" required>
        </label>
        <label class="field" style="flex:1;min-width:240px;">
          Remote MCP URL
          <input type="text" name="url" placeholder="https://api.githubcopilot.com/mcp/" required style="width:100%;">
        </label>
        <label class="field">
          Auth mode
          <select name="auth_mode">
            <option value="oauth">oauth</option>
            <option value="bearer">bearer</option>
            <option value="none">none</option>
          </select>
        </label>
        <label class="field">
          Default tier
          <select name="default_tier">
            <option value="external" selected>external</option>
            <option value="write">write</option>
            <option value="read">read</option>
          </select>
        </label>
        <button type="submit">Add Server</button>
      </form>
    </div>
    """
    return _layout(title="MCP Servers", active="upstreams", csrf_token=csrf_token, body=body)


def render_clients_page(
    *,
    clients: list[dict[str, Any]],
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
    new_token_info: dict[str, str] | None = None,
    max_oauth_clients: int = 50,
) -> str:
    banners = ""
    if error:
        banners += f'<div class="notice notice-err">{_e(error)}</div>'
    if notice:
        banners += f'<div class="notice notice-ok">{_e(notice)}</div>'
    if new_token_info:
        banners += f"""
        <div class="notice notice-token">
          <strong>New bearer token for <code>{_e(new_token_info['client_name'])}</code> — shown once, copy now:</strong>
          <div class="token-box">{_e(new_token_info['token'])}</div>
        </div>
        """

    oauth_count = sum(1 for c in clients if c["kind"] == "oauth")
    rows = []
    for c in clients:
        cid = str(c["id"])
        ceiling_badge = (
            '<span class="badge badge-warn">untrusted</span>'
            if c["trust_ceiling"] == "untrusted"
            else '<span class="badge badge-muted">owner</span>'
        )
        actions = []
        if c["kind"] == "bearer":
            actions.append(
                f'<form class="inline" method="post" action="/admin/clients/token">'
                f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                f'<input type="hidden" name="client_name" value="{_e(c["name"])}">'
                f'<button type="submit" class="secondary sm">New Token</button></form>'
            )
        actions.append(
            f'<form class="inline" method="post" action="/admin/clients/revoke">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f'<input type="hidden" name="client_id" value="{_e(cid)}">'
            f'<button type="submit" class="danger sm">Revoke</button></form>'
        )
        if c["kind"] == "oauth":
            actions.append(
                f'<form class="inline" method="post" action="/admin/clients/delete">'
                f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                f'<input type="hidden" name="client_id" value="{_e(cid)}">'
                f'<button type="submit" class="secondary sm">Prune</button></form>'
            )
        rows.append(
            f"<tr>"
            f"<td><strong>{_e(c['name'])}</strong><div class=\"muted mono\">{_e(c['oauth_client_id'] or cid)}</div></td>"
            f"<td><code>{_e(c['kind'])}</code></td>"
            f"<td>{ceiling_badge}</td>"
            f"<td>{int(c['active_token_count'])}</td>"
            f"<td>{_e(_fmt_ts(c['last_seen_at']))}</td>"
            f"<td>{' '.join(actions)}</td>"
            f"</tr>"
        )

    body = f"""
    <h1>Clients &amp; Tokens</h1>
    {banners}
    <div class="card">
      <h2>Create Bearer Client</h2>
      <form class="form-row" method="post" action="/admin/clients/create">
        <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
        <label class="field">
          Client name
          <input type="text" name="name" placeholder="e.g. codex-cli" required>
        </label>
        <label class="field">
          Trust ceiling
          <select name="trust_ceiling">
            <option value="owner">owner</option>
            <option value="untrusted">untrusted</option>
          </select>
        </label>
        <button type="submit">Create &amp; Mint Token</button>
      </form>
    </div>
    <div class="card">
      <h2>Registered Clients <span class="badge badge-muted">OAuth: {oauth_count} / {int(max_oauth_clients)}</span></h2>
      <div class="scroll-x">
        <table>
          <thead><tr><th>Client</th><th>Kind</th><th>Trust Ceiling</th><th>Active Tokens</th><th>Last Seen</th><th>Actions</th></tr></thead>
          <tbody>{''.join(rows) or '<tr><td colspan="6" class="muted">No clients yet.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    """
    return _layout(title="Clients", active="clients", csrf_token=csrf_token, body=body)


def _decision_badge(decision: str) -> str:
    cls = {
        "allow": "badge-ok",
        "require_approval": "badge-warn",
        "deny": "badge-err",
    }.get(decision, "badge-muted")
    return f'<span class="badge {cls}">{_e(decision)}</span>'


def _decision_form(
    *,
    csrf_token: str,
    pattern: str,
    client_id: str | None,
    current_explicit: str | None,
) -> str:
    cid_input = (
        f'<input type="hidden" name="client_id" value="{_e(client_id)}">' if client_id else ""
    )
    opts = []
    for val in ("allow", "require_approval", "deny"):
        sel = " selected" if current_explicit == val else ""
        opts.append(f'<option value="{val}"{sel}>{val}</option>')
    clear_btn = ""
    if current_explicit is not None:
        clear_btn = (
            f'<form class="inline" method="post" action="/admin/policy/clear">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f"{cid_input}"
            f'<input type="hidden" name="pattern" value="{_e(pattern)}">'
            f'<button type="submit" class="secondary sm">Clear</button></form>'
        )
    return (
        f'<form class="inline" method="post" action="/admin/policy/set">'
        f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
        f"{cid_input}"
        f'<input type="hidden" name="pattern" value="{_e(pattern)}">'
        f'<select name="decision">{"".join(opts)}</select>'
        f'<button type="submit" class="secondary sm">Set</button></form> {clear_btn}'
    )


def render_policy_page(
    *,
    matrix: dict[str, Any],
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
) -> str:
    banners = ""
    if error:
        banners += f'<div class="notice notice-err">{_e(error)}</div>'
    if notice:
        banners += f'<div class="notice notice-ok">{_e(notice)}</div>'

    clients = matrix["clients"]
    tools = matrix["tools"]
    custom_rules = matrix["custom_rules"]

    # Tool catalog + global override + per-client matrix
    client_headers = "".join(
        f'<th>{_e(c["name"])}<div class="muted">{_e(c["trust_ceiling"])}</div></th>' for c in clients
    )

    tool_rows = []
    for t in tools:
        tier_badge = {
            "read": '<span class="badge badge-ok">read</span>',
            "write": '<span class="badge badge-muted">write</span>',
            "external": '<span class="badge badge-warn">external</span>',
        }.get(t["tier"], f'<span class="badge badge-muted">{_e(t["tier"])}</span>')

        ann_parts = []
        if t["read_only_hint"]:
            ann_parts.append("readOnly")
        if t["destructive_hint"]:
            ann_parts.append("destructive")
        if t["open_world_hint"]:
            ann_parts.append("openWorld")
        ann_str = ", ".join(ann_parts) or "—"

        global_cell = (
            f'<div class="cell-controls">'
            f'{_decision_badge(t["global_override"]) if t["global_override"] else "<span class=muted>tier default</span>"}'
            f'{_decision_form(csrf_token=csrf_token, pattern=t["name"], client_id=None, current_explicit=t["global_override"])}'
            f"</div>"
        )

        client_cells = []
        for cell in t["per_client"]:
            override_note = " (override)" if cell["explicit"] else ""
            client_cells.append(
                f'<td><div class="cell-controls">'
                f'<div>{_decision_badge(cell["effective"])}<span class="muted">{override_note}</span></div>'
                f'{_decision_form(csrf_token=csrf_token, pattern=t["name"], client_id=str(cell["client_id"]), current_explicit=cell["explicit"])}'
                f"</div></td>"
            )

        tier_opts = "".join(
            f'<option value="{val}"{" selected" if t["tier"] == val else ""}>{val}</option>'
            for val in ("read", "write", "external")
        )
        tier_cell = (
            f'<div class="cell-controls">'
            f"<div>{tier_badge}</div>"
            f'<form class="inline" method="post" action="/admin/policy/tier">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f'<input type="hidden" name="pattern" value="{_e(t["name"])}">'
            f'<select name="tier">{tier_opts}</select>'
            f'<button type="submit" class="secondary sm">Set</button></form>'
            f"</div>"
        )

        status_dot = "" if t["online"] else ' <span class="badge badge-muted">offline</span>'
        tool_rows.append(
            f"<tr>"
            f"<td><code>{_e(t['name'])}</code>{status_dot}<div class=\"muted\">{_e(t['provider'])}</div></td>"
            f"<td>{tier_cell}</td>"
            f"<td class=\"mono\">{_e(ann_str)}</td>"
            f"<td>{global_cell}</td>"
            f"{''.join(client_cells)}"
            f"</tr>"
        )

    client_options = '<option value="">* (all clients)</option>' + "".join(
        f'<option value="{_e(c["id"])}">{_e(c["name"])} ({_e(c["kind"])})</option>' for c in clients
    )

    custom_rows = []
    for r in custom_rules:
        cid_input = (
            f'<input type="hidden" name="client_id" value="{_e(r["client_id"])}">'
            if r["client_id"]
            else ""
        )
        custom_rows.append(
            f"<tr>"
            f"<td>{_e(r['client_name'])}</td>"
            f"<td><code>{_e(r['pattern'])}</code></td>"
            f"<td>{_decision_badge(r['decision'])}</td>"
            f"<td>"
            f'<form class="inline" method="post" action="/admin/policy/clear">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f"{cid_input}"
            f'<input type="hidden" name="pattern" value="{_e(r["pattern"])}">'
            f'<button type="submit" class="danger sm">Clear</button></form>'
            f"</td></tr>"
        )

    custom_section = ""
    if custom_rows:
        custom_section = f"""
        <div class="card">
          <h2>Custom Pattern Overrides</h2>
          <table>
            <thead><tr><th>Client</th><th>Pattern</th><th>Decision</th><th>Action</th></tr></thead>
            <tbody>{''.join(custom_rows)}</tbody>
          </table>
        </div>
        """

    body = f"""
    <h1>Tools &amp; Policy Matrix</h1>
    {banners}
    <div class="card">
      <h2>Set Custom Pattern Policy</h2>
      <form class="form-row" method="post" action="/admin/policy/set">
        <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
        <label class="field">
          Client
          <select name="client_id">{client_options}</select>
        </label>
        <label class="field">
          Tool pattern
          <input type="text" name="pattern" placeholder="e.g. mac_* or gmail_send" required>
        </label>
        <label class="field">
          Decision
          <select name="decision">
            <option value="allow">allow</option>
            <option value="require_approval">require_approval</option>
            <option value="deny">deny</option>
          </select>
        </label>
        <button type="submit">Set Policy</button>
      </form>
    </div>
    {custom_section}
    <div class="card">
      <h2>Merged Tools &amp; Per-Client Decisions</h2>
      <div class="scroll-x">
        <table>
          <thead>
            <tr>
              <th>Tool</th>
              <th>Tier</th>
              <th>Annotations</th>
              <th>All Clients</th>
              {client_headers}
            </tr>
          </thead>
          <tbody>{''.join(tool_rows)}</tbody>
        </table>
      </div>
    </div>
    """
    return _layout(title="Tools & Policy", active="policy", csrf_token=csrf_token, body=body)


def render_activity_page(
    *,
    data: dict[str, Any],
    filters: dict[str, str],
    csrf_token: str,
) -> str:
    client_opts = '<option value="">All clients</option>' + "".join(
        f'<option value="{_e(c["id"])}"{" selected" if filters.get("client_id") == str(c["id"]) else ""}>{_e(c["name"])}</option>'
        for c in data["filter_clients"]
    )
    prov_opts = '<option value="">All providers</option>' + "".join(
        f'<option value="{_e(p["id"])}"{" selected" if filters.get("provider_id") == str(p["id"]) else ""}>{_e(p["slug"])}</option>'
        for p in data["filter_providers"]
    )
    type_opts = '<option value="">All types</option>' + "".join(
        f'<option value="{_e(t)}"{" selected" if filters.get("type") == t else ""}>{_e(t)}</option>'
        for t in data["filter_types"]
    )
    outcome_opts = '<option value="">All outcomes</option>' + "".join(
        f'<option value="{o}"{" selected" if filters.get("outcome") == o else ""}>{o}</option>'
        for o in ("ok", "denied", "error")
    )

    rows = []
    for ev in data["events"]:
        outcome_badge = {
            "ok": '<span class="badge badge-ok">ok</span>',
            "denied": '<span class="badge badge-warn">denied</span>',
            "error": '<span class="badge badge-err">error</span>',
        }.get(ev["outcome"], f'<span class="badge badge-muted">{_e(ev["outcome"])}</span>')

        digest_hex = ev["args_digest_hex"]
        digest_cell = (
            f'<code title="{_e(digest_hex)}">{_e(digest_hex[:16])}…</code>' if digest_hex else "—"
        )
        rows.append(
            f"<tr>"
            f"<td class=\"mono\">{int(ev['id'])}</td>"
            f"<td class=\"mono\">{_e(_fmt_ts(ev['created_at']))}</td>"
            f"<td><code>{_e(ev['type'])}</code></td>"
            f"<td>{_e(ev['client_name'] or '—')}</td>"
            f"<td><code>{_e(ev['provider_slug'] or '—')}</code></td>"
            f"<td><code>{_e(ev['tool'] or '—')}</code></td>"
            f"<td>{_e(ev['decision'] or '—')}</td>"
            f"<td>{outcome_badge}</td>"
            f"<td>{digest_cell}</td>"
            f"</tr>"
        )

    pager = ""
    if data["next_before_id"] is not None:
        qp = {k: v for k, v in filters.items() if v}
        qp["before_id"] = str(data["next_before_id"])
        pager = f'<p style="margin-top:1rem;"><a class="btn secondary" href="/admin/activity?{urlencode(qp)}">Older events &rarr;</a></p>'

    body = f"""
    <h1>Activity Audit Trail</h1>
    <div class="card">
      <form class="form-row" method="get" action="/admin/activity">
        <label class="field">Client<select name="client_id">{client_opts}</select></label>
        <label class="field">Provider<select name="provider_id">{prov_opts}</select></label>
        <label class="field">Type<select name="type">{type_opts}</select></label>
        <label class="field">Outcome<select name="outcome">{outcome_opts}</select></label>
        <button type="submit">Filter</button>
        <a class="btn secondary" href="/admin/activity">Reset</a>
      </form>
    </div>
    <div class="card">
      <p class="muted">Append-only audit log. Only SHA-256 argument digests are recorded and displayed; raw arguments are never stored or shown.</p>
      <div class="scroll-x">
        <table>
          <thead>
            <tr><th>ID</th><th>Time</th><th>Type</th><th>Client</th><th>Provider</th><th>Tool</th><th>Decision</th><th>Outcome</th><th>Args Digest</th></tr>
          </thead>
          <tbody>{''.join(rows) or '<tr><td colspan="9" class="muted">No matching events.</td></tr>'}</tbody>
        </table>
      </div>
      {pager}
    </div>
    """
    return _layout(title="Activity", active="activity", csrf_token=csrf_token, body=body)
