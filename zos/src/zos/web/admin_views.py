"""Server-rendered HTML templates for ZOS owner admin dashboard (/admin)."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from zos.web.theme import CSS as _CSS


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_ts(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(value)


def _layout(
    *,
    title: str,
    active: str,
    csrf_token: str,
    body: str,
    pending_approvals: int = 0,
    ui_modules: list[dict[str, Any]] | None = None,
) -> str:
    approvals_label = (
        f'Approvals <span class="badge badge-warn">{int(pending_approvals)}</span>'
        if pending_approvals > 0
        else "Approvals"
    )
    nav_items = [
        ("overview", "/admin", "Overview"),
        ("approvals", "/admin/approvals", approvals_label),
        ("modules", "/admin/modules", "Modules"),
        ("upstreams", "/admin/upstreams", "MCP Servers"),
        ("clients", "/admin/clients", "Clients"),
        ("policy", "/admin/policy", "Tools &amp; Policy"),
        ("activity", "/admin/activity", "Activity"),
    ]
    active_attrs = ' class="active" aria-current="page"'
    nav_html = "".join(
        f'<a href="{href}"{active_attrs if active == key else ""}>{label}</a>'
        for key, href, label in nav_items
    )
    if ui_modules:
        nav_html += "".join(
            f'<a href="{_e(m["href"])}">{_e(m["title"])}</a>'
            for m in ui_modules
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>ZOS Admin — {_e(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <a class="skip" href="#main">Skip to content</a>
  <header class="topbar">
    <div class="topbar-in">
      <a class="brand" href="/admin">ZOS</a>
      <form class="inline" method="post" action="/admin/logout">
        <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
        <button type="submit" class="secondary sm">Sign out</button>
      </form>
    </div>
    <nav class="main-nav" aria-label="Admin">{nav_html}</nav>
  </header>
  <main id="main">{body}</main>
</body>
</html>
"""


def render_admin_login_page(*, error: str | None = None, next_url: str | None = None) -> str:
    err_html = f'<div class="notice notice-err">{_e(error)}</div>' if error else ""
    next_input = (
        f'<input type="hidden" name="next" value="{_e(next_url)}">' if next_url else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>ZOS Admin — Sign In</title>
  <style>{_CSS}</style>
</head>
<body>
  <main class="narrow">
    <div class="card">
      <h1>ZOS Owner Admin</h1>
      <p class="muted">Enter your owner passphrase to open the control panel.</p>
      {err_html}
      <form class="auth-form" method="post" action="/admin/login">
        {next_input}
        <label class="field">
          Owner passphrase
          <input name="passphrase" type="password" autocomplete="current-password" required autofocus>
        </label>
        <button type="submit" class="block">Sign in</button>
      </form>
    </div>
  </main>
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
    ui_modules: list[dict[str, Any]] | None = None,
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
                    f'<form class="inline" method="post" action="/admin/providers/backend">'
                    f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                    f'<input type="hidden" name="slug" value="{_e(p["namespace"])}">'
                    f'<input type="hidden" name="backend" value="{_e(target)}">'
                    f'<button type="submit" class="secondary sm">Switch to {_e(target)}</button></form>'
                )
            backend_cell = f'<div class="row"><span>{backend_cell}</span>{"".join(toggles)}</div>'
        prov_rows.append(
            f"<tr>"
            f"<td data-label=\"Namespace\"><code>{_e(p['namespace'])}</code></td>"
            f"<td data-label=\"Kind\">{_e(p['kind'])}</td>"
            f"<td data-label=\"Backend\" class=\"cell-wide\">{backend_cell}</td>"
            f"<td data-label=\"Status\">{badge}</td>"
            f"<td data-label=\"Last seen\" class=\"mono nowrap\">{_e(_fmt_ts(p['last_seen_at']))}</td>"
            f"</tr>"
        )

    cred = overview["google_credential"]
    if cred:
        scopes_html = " ".join(f'<span class="badge badge-muted">{_e(s.rsplit("/", 1)[-1])}</span>' for s in cred["scopes"])
        cred_html = f"""
        <p><span class="badge badge-ok">connected</span> <strong>{_e(cred['account'])}</strong></p>
        <p class="muted">Access token expires: <code>{_e(_fmt_ts(cred['access_expires_at']))}</code> &middot; Updated: <code>{_e(_fmt_ts(cred['updated_at']))}</code></p>
        <p class="row">{scopes_html}</p>
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
        backup_val = f'{_e(backup_size)}<span class="sub">{_e(backup_ts)}</span>'
    else:
        backup_val = '<span class="muted">Never</span>'

    pending_count = int(overview.get("pending_approvals_count", 0))
    pending_badge = (
        f'<span class="badge badge-warn">{pending_count} pending</span>'
        if pending_count > 0
        else '<span class="badge badge-muted">0 pending</span>'
    )

    module_ui_card = ""
    if ui_modules:
        mod_links = " ".join(
            f'<a class="btn secondary" href="{_e(m["href"])}">{_e(m["title"])}</a>'
            for m in ui_modules
        )
        module_ui_card = f"""
    <div class="card">
      <h2>Module UIs</h2>
      <p class="row">{mod_links}</p>
    </div>
    """

    body = f"""
    <h1>Overview</h1>
    {banners}
    <div class="stats">
      <div class="stat"><div class="label">Active MCP Sessions</div><div class="value">{int(overview['active_sessions'])}</div></div>
      <div class="stat"><div class="label">Online Providers</div><div class="value">{online_count} / {len(providers)}</div></div>
      <div class="stat"><div class="label">Pending Approvals</div><div class="value"><a href="/admin/approvals">{pending_badge}</a></div></div>
      <div class="stat"><div class="label">Last Backup</div><div class="value">{backup_val}</div></div>
    </div>
    {module_ui_card}
    <div class="card">
      <h2>Providers</h2>
      <div class="scroll-x">
        <table class="stack">
          <thead><tr><th>Namespace</th><th>Kind</th><th>Backend</th><th>Status</th><th>Last Seen</th></tr></thead>
          <tbody>{''.join(prov_rows) or '<tr><td colspan="5" class="empty">No providers registered.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Google Credential</h2>
      {cred_html}
    </div>
    """
    return _layout(
        title="Overview",
        active="overview",
        csrf_token=csrf_token,
        body=body,
        pending_approvals=pending_count,
        ui_modules=ui_modules,
    )


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
            f'<div class="muted err-text">{_e(u["last_error"])}</div>'
            if u.get("last_error")
            else ""
        )
        help_note = ""
        if u["auth_mode"] == "bearer" and u.get("token_help") and not u.get("has_credential"):
            help_note = f'<div class="muted">{_e(u["token_help"])}</div>'
        elif u["auth_mode"] == "oauth" and u.get("requires_client_registration") and u.get("client_help"):
            help_note = f'<div class="muted">{_e(u["client_help"])}</div>'

        actions: list[str] = []
        if u["auth_mode"] == "bearer":
            actions.append(
                f'<form class="inline" method="post" action="/admin/upstreams/token">'
                f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                f'<input type="hidden" name="slug" value="{_e(slug)}">'
                f'<input type="password" name="token" placeholder="Paste token" autocomplete="new-password" required aria-label="Bearer token">'
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
                    f'<input type="text" name="client_id" placeholder="Client ID" required aria-label="Client ID">'
                    f'<input type="password" name="client_secret" placeholder="Client Secret" autocomplete="new-password" required aria-label="Client secret">'
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
            f'<td data-label="Server" class="cell-wide"><code>{_e(slug)}</code><div class="muted mono">{_e(u["url"])}</div>{help_note}</td>'
            f'<td data-label="Auth"><code>{_e(u["auth_mode"])}</code></td>'
            f'<td data-label="Default tier"><code>{_e(u["default_tier"])}</code></td>'
            f'<td data-label="Status">{badge}{err_note}</td>'
            f'<td data-label="Tools">{int(u["tool_count"])}</td>'
            f'<td data-label="Actions" class="cell-wide"><div class="row">{"".join(actions)}</div></td>'
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
            f'<input type="text" name="as_slug" placeholder="alias (optional)" pattern="[a-z][a-z0-9]{{1,15}}" aria-label="Alias">'
            f'<button type="submit" class="{"secondary" if is_installed else ""} sm">{"Install another" if is_installed else "Install"}</button></form>'
        )
        cat_rows.append(
            f"<tr>"
            f'<td data-label="Connector" class="cell-wide"><strong>{_e(c["name"])}</strong>{inst_badge} <code class="muted">{_e(c_slug)}</code>'
            f'<div class="muted">{_e(c["description"])}</div></td>'
            f'<td data-label="Auth"><code>{_e(auth_label)}</code></td>'
            f'<td data-label="Docs"><a href="{_e(c["docs_url"])}" target="_blank" rel="noopener noreferrer">Docs</a></td>'
            f'<td data-label="Install" class="cell-wide">{action_form}</td>'
            f"</tr>"
        )

    body = f"""
    <h1>MCP Servers</h1>
    {banners}
    <div class="card">
      <h2>Registered Remote MCP Servers</h2>
      <div class="scroll-x">
        <table class="stack">
          <thead><tr><th>Server</th><th>Auth</th><th>Default Tier</th><th>Status</th><th>Tools</th><th>Actions</th></tr></thead>
          <tbody>{''.join(rows) or '<tr><td colspan="6" class="empty">No remote MCP servers registered yet. Install one from the catalog below or add a custom URL.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Available Connectors</h2>
      <p class="muted">Verified official remote MCP servers. Nothing is installed by default; click <strong>Install</strong> to register with manifest tier rules, then connect.</p>
      <div class="scroll-x">
        <table class="stack">
          <thead><tr><th>Connector</th><th>Auth</th><th>Docs</th><th>Install</th></tr></thead>
          <tbody>{''.join(cat_rows) or '<tr><td colspan="4" class="empty">No catalog manifests found.</td></tr>'}</tbody>
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
        <label class="field grow">
          Remote MCP URL
          <input type="text" name="url" placeholder="https://api.githubcopilot.com/mcp/" required>
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
        else:
            actions.append(
                f'<form class="inline" method="post" action="/admin/clients/disable">'
                f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                f'<input type="hidden" name="client_id" value="{_e(cid)}">'
                f'<button type="submit" class="secondary sm">Disable</button></form>'
            )
        rows.append(
            f"<tr>"
            f"<td data-label=\"Client\" class=\"cell-wide\"><strong>{_e(c['name'])}</strong><div class=\"muted mono\">{_e(c['oauth_client_id'] or cid)}</div></td>"
            f"<td data-label=\"Kind\"><code>{_e(c['kind'])}</code></td>"
            f"<td data-label=\"Trust ceiling\">{ceiling_badge}</td>"
            f"<td data-label=\"Active tokens\">{int(c['active_token_count'])}</td>"
            f"<td data-label=\"Last seen\" class=\"mono nowrap\">{_e(_fmt_ts(c['last_seen_at']))}</td>"
            f"<td data-label=\"Actions\" class=\"cell-wide\"><div class=\"row\">{''.join(actions)}</div></td>"
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
        <table class="stack">
          <thead><tr><th>Client</th><th>Kind</th><th>Trust Ceiling</th><th>Active Tokens</th><th>Last Seen</th><th>Actions</th></tr></thead>
          <tbody>{''.join(rows) or '<tr><td colspan="6" class="empty">No clients yet.</td></tr>'}</tbody>
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
            f"<td data-label=\"Client\">{_e(r['client_name'])}</td>"
            f"<td data-label=\"Pattern\"><code>{_e(r['pattern'])}</code></td>"
            f"<td data-label=\"Decision\">{_decision_badge(r['decision'])}</td>"
            f"<td data-label=\"Action\">"
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
          <table class="stack">
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
            f'<code class="digest" title="{_e(digest_hex)}">{_e(digest_hex[:16])}…</code>' if digest_hex else "—"
        )
        rows.append(
            f"<tr>"
            f"<td data-label=\"ID\" class=\"mono\">{int(ev['id'])}</td>"
            f"<td data-label=\"Time\" class=\"mono nowrap\">{_e(_fmt_ts(ev['created_at']))}</td>"
            f"<td data-label=\"Type\"><code>{_e(ev['type'])}</code></td>"
            f"<td data-label=\"Client\">{_e(ev['client_name'] or '—')}</td>"
            f"<td data-label=\"Provider\"><code>{_e(ev['provider_slug'] or '—')}</code></td>"
            f"<td data-label=\"Tool\"><code>{_e(ev['tool'] or '—')}</code></td>"
            f"<td data-label=\"Decision\">{_e(ev['decision'] or '—')}</td>"
            f"<td data-label=\"Outcome\">{outcome_badge}</td>"
            f"<td data-label=\"Args digest\">{digest_cell}</td>"
            f"</tr>"
        )

    pager = ""
    if data["next_before_id"] is not None:
        qp = {k: v for k, v in filters.items() if v}
        qp["before_id"] = str(data["next_before_id"])
        pager = f'<p class="pager"><a class="btn secondary" href="/admin/activity?{urlencode(qp)}">Older events &rarr;</a></p>'

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
        <table class="stack">
          <thead>
            <tr><th>ID</th><th>Time</th><th>Type</th><th>Client</th><th>Provider</th><th>Tool</th><th>Decision</th><th>Outcome</th><th>Args Digest</th></tr>
          </thead>
          <tbody>{''.join(rows) or '<tr><td colspan="9" class="empty">No matching events.</td></tr>'}</tbody>
        </table>
      </div>
      {pager}
    </div>
    """
    return _layout(title="Activity", active="activity", csrf_token=csrf_token, body=body)


def _approval_status_badge(status: str) -> str:
    cls = {
        "pending": "badge-warn",
        "approved": "badge-ok",
        "consumed": "badge-ok",
        "denied": "badge-err",
        "expired": "badge-muted",
    }.get(status, "badge-muted")
    return f'<span class="badge {cls}">{_e(status)}</span>'


def render_approvals_page(
    *,
    approvals: dict[str, list[dict[str, Any]]],
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
) -> str:
    banners = ""
    if error:
        banners += f'<div class="notice notice-err">{_e(error)}</div>'
    if notice:
        banners += f'<div class="notice notice-ok">{_e(notice)}</div>'

    pending = approvals.get("pending") or []
    recent = approvals.get("recent") or []

    pending_rows = []
    for item in pending:
        aid = item["id"]
        digest_hex = item.get("args_digest_hex") or ""
        digest_cell = f'<code class="digest" title="{_e(digest_hex)}">{_e(digest_hex[:12])}…</code>' if digest_hex else "—"
        pending_rows.append(
            f"<tr>"
            f'<td data-label="Client"><strong>{_e(item["client_name"])}</strong></td>'
            f'<td data-label="Tool"><code>{_e(item["tool"])}</code></td>'
            f'<td data-label="Args digest">{digest_cell}</td>'
            f'<td data-label="Requested" class="mono nowrap">{_e(_fmt_ts(item["created_at"]))}</td>'
            f'<td data-label="Expires" class="mono nowrap">{_e(_fmt_ts(item["expires_at"]))}</td>'
            f'<td data-label="Action"><a class="btn sm" href="/admin/approvals/{_e(aid)}">Review</a></td>'
            f"</tr>"
        )

    recent_rows = []
    for item in recent:
        aid = item["id"]
        digest_hex = item.get("args_digest_hex") or ""
        digest_cell = f'<code class="digest" title="{_e(digest_hex)}">{_e(digest_hex[:12])}…</code>' if digest_hex else "—"
        recent_rows.append(
            f"<tr>"
            f'<td data-label="Status">{_approval_status_badge(item["status"])}</td>'
            f'<td data-label="Client">{_e(item["client_name"])}</td>'
            f'<td data-label="Tool"><code>{_e(item["tool"])}</code></td>'
            f'<td data-label="Args digest">{digest_cell}</td>'
            f'<td data-label="Requested" class="mono nowrap">{_e(_fmt_ts(item["created_at"]))}</td>'
            f'<td data-label="Decided / expired" class="mono nowrap">{_e(_fmt_ts(item["decided_at"] or item["expires_at"]))}</td>'
            f'<td data-label="Details"><a class="btn secondary sm" href="/admin/approvals/{_e(aid)}">View</a></td>'
            f"</tr>"
        )

    body = f"""
    <h1>Tool Approvals</h1>
    {banners}
    <div class="card">
      <h2>Pending Approvals ({len(pending)})</h2>
      <p class="muted">Requests from clients without MCP elicitation. Open a request to inspect its exact arguments and approve or deny.</p>
      <div class="scroll-x">
        <table class="stack">
          <thead><tr><th>Client</th><th>Tool</th><th>Args Digest</th><th>Requested</th><th>Expires</th><th>Action</th></tr></thead>
          <tbody>{''.join(pending_rows) or '<tr><td colspan="6" class="empty">Nothing waiting on you.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Recent Approvals</h2>
      <div class="scroll-x">
        <table class="stack">
          <thead><tr><th>Status</th><th>Client</th><th>Tool</th><th>Args Digest</th><th>Requested</th><th>Decided / Expired</th><th>Details</th></tr></thead>
          <tbody>{''.join(recent_rows) or '<tr><td colspan="7" class="empty">No recent approvals yet.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    """
    return _layout(
        title="Approvals",
        active="approvals",
        csrf_token=csrf_token,
        body=body,
        pending_approvals=len(pending),
    )


def render_approval_detail_page(
    *,
    approval: dict[str, Any] | None,
    csrf_token: str,
    pending_approvals: int = 0,
    notice: str | None = None,
    error: str | None = None,
) -> str:
    banners = ""
    if error:
        banners += f'<div class="notice notice-err">{_e(error)}</div>'
    if notice:
        banners += f'<div class="notice notice-ok">{_e(notice)}</div>'

    if approval is None:
        body = f"""
        <h1>Approval Not Found</h1>
        {banners}
        <div class="card">
          <p class="muted">This approval request does not exist or has been removed.</p>
          <p><a class="btn secondary" href="/admin/approvals">&larr; Back to Approvals</a></p>
        </div>
        """
        return _layout(
            title="Approval Not Found",
            active="approvals",
            csrf_token=csrf_token,
            body=body,
            pending_approvals=pending_approvals,
        )

    aid = approval["id"]
    is_pending = approval["status"] == "pending"
    if is_pending and approval.get("args_pretty") is not None:
        args_section = f"""
        <div class="card">
          <h2>Arguments</h2>
          <pre class="approval-pre">{_e(approval["args_pretty"])}</pre>
          <div class="approval-actions">
            <form method="post" action="/admin/approvals/{_e(aid)}/approve">
              <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
              <button type="submit" class="approve-btn">Approve</button>
            </form>
            <form method="post" action="/admin/approvals/{_e(aid)}/deny">
              <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
              <button type="submit" class="deny-btn">Deny</button>
            </form>
          </div>
        </div>
        """
    else:
        args_section = """
        <div class="card">
          <h2>Arguments</h2>
          <p class="muted">Encrypted arguments have been purged after decision or expiration.</p>
        </div>
        """

    body = f"""
    <p class="back"><a class="btn secondary sm" href="/admin/approvals">&larr; All Approvals</a></p>
    <h1>Approve Tool Call</h1>
    {banners}
    <div class="card">
      <div class="meta-grid">
        <div class="meta-item"><div class="label">Status</div><div class="val">{_approval_status_badge(approval["status"])}</div></div>
        <div class="meta-item"><div class="label">Client</div><div class="val"><strong>{_e(approval["client_name"])}</strong></div></div>
        <div class="meta-item"><div class="label">Tool</div><div class="val"><code>{_e(approval["tool"])}</code></div></div>
        <div class="meta-item"><div class="label">Requested</div><div class="val mono">{_e(_fmt_ts(approval["created_at"]))}</div></div>
        <div class="meta-item"><div class="label">Expires</div><div class="val mono">{_e(_fmt_ts(approval["expires_at"]))}</div></div>
        <div class="meta-item"><div class="label">Args SHA-256</div><div class="val digest">{_e(approval.get("args_digest_hex", "")[:24])}…</div></div>
      </div>
    </div>
    {args_section}
    """
    return _layout(
        title=f"Approval — {approval['tool']}",
        active="approvals",
        csrf_token=csrf_token,
        body=body,
        pending_approvals=pending_approvals,
    )


def render_modules_page(
    *,
    installed: list[dict[str, Any]],
    catalog: list[Any],
    host_kind: str,
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
    instructions: str | None = None,
    pending_approvals: int = 0,
    ui_modules: list[dict[str, Any]] | None = None,
) -> str:
    banners = ""
    if error:
        banners += f'<div class="notice notice-err">{_e(error)}</div>'
    if notice:
        banners += f'<div class="notice notice-ok">{_e(notice)}</div>'
    if instructions:
        banners += (
            '<div class="card">'
            '<h2>Manual Setup Instructions</h2>'
            f'<pre class="approval-pre">{_e(instructions)}</pre>'
            '</div>'
        )

    installed_slugs = {str(m["slug"]) for m in installed}

    inst_rows = []
    for m in installed:
        slug = str(m["slug"])
        status = str(m.get("status") or "installed")
        is_disabled = m.get("disabled_at") is not None or status == "disabled"
        online = bool(m.get("online", False))
        if is_disabled:
            badge = '<span class="badge badge-muted">disabled</span>'
        elif status == "pending_attach":
            badge = '<span class="badge badge-warn">pending attach</span>'
        elif online:
            badge = f'<span class="badge badge-ok">online ({int(m.get("tool_count", 0))} tools)</span>'
        else:
            badge = f'<span class="badge badge-warn">{_e(m.get("upstream_status") or "offline")}</span>'

        ui_link = ""
        if m.get("ui_url") and not is_disabled:
            ui_title = m.get("ui_title") or slug
            ui_link = f'<a class="btn secondary sm" href="/admin/m/{_e(slug)}/">Open {_e(ui_title)}</a> '

        toggle_btn = (
            f'<form class="inline" method="post" action="/admin/modules/enable">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f'<input type="hidden" name="slug" value="{_e(slug)}">'
            f'<button type="submit" class="secondary sm">Enable</button></form>'
            if is_disabled
            else f'<form class="inline" method="post" action="/admin/modules/disable">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f'<input type="hidden" name="slug" value="{_e(slug)}">'
            f'<button type="submit" class="secondary sm">Disable</button></form>'
        )
        uninstall_btn = (
            f'<form class="inline" method="post" action="/admin/modules/uninstall" '
            f'onsubmit="return confirm(\'Uninstall module {_e(slug)}? Data will be kept.\')">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f'<input type="hidden" name="slug" value="{_e(slug)}">'
            f'<button type="submit" class="danger sm">Uninstall</button></form>'
            f'<form class="inline" method="post" action="/admin/modules/uninstall" '
            f'onsubmit="return confirm(\'Uninstall AND purge all data for {_e(slug)}?\')">'
            f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
            f'<input type="hidden" name="slug" value="{_e(slug)}">'
            f'<input type="hidden" name="purge" value="1">'
            f'<button type="submit" class="danger sm">Purge</button></form>'
        )
        attach_form = ""
        if status == "pending_attach":
            attach_form = (
                f'<form class="inline" method="post" action="/admin/modules/attach">'
                f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                f'<input type="hidden" name="slug" value="{_e(slug)}">'
                f'<input type="text" name="mcp_url" placeholder="http://127.0.0.1:8081/mcp" aria-label="MCP URL" required>'
                f'<input type="text" name="ui_url" placeholder="UI URL (optional)" aria-label="UI URL">'
                f'<button type="submit" class="sm">Attach</button></form>'
            )

        inst_rows.append(
            f"<tr>"
            f"<td data-label=\"Module\"><code>{_e(slug)}</code></td>"
            f"<td data-label=\"Version\"><code>v{_e(m.get('version', ''))}</code></td>"
            f"<td data-label=\"Host\"><code>{_e(m.get('host', ''))}</code></td>"
            f"<td data-label=\"Status\">{badge}</td>"
            f"<td data-label=\"Actions\" class=\"cell-wide\"><div class=\"cell-controls\"><div class=\"row\">{ui_link}{toggle_btn} {uninstall_btn}</div>{attach_form}</div></td>"
            f"</tr>"
        )

    installed_table = (
        f'<div class="scroll-x"><table class="stack"><thead><tr>'
        f'<th>Module</th><th>Version</th><th>Host</th><th>Status</th><th>Actions</th>'
        f'</tr></thead><tbody>{"".join(inst_rows)}</tbody></table></div>'
        if inst_rows
        else '<p class="empty">No modules installed yet. Install one from the catalog below.</p>'
    )

    cat_rows = []
    for entry in catalog:
        already = entry.slug in installed_slugs
        cfg_inputs = "".join(
            f'<label class="field">{_e(f.name)}'
            f'<input type="{"password" if f.secret else "text"}" name="cfg_{_e(f.name)}" '
            f'placeholder="{_e(f.help or f.name)}" {"required" if f.required else ""}></label>'
            for f in entry.config
        )
        badges = []
        if entry.storage.postgres:
            badges.append('<span class="badge badge-muted">postgres</span>')
        if entry.network.egress:
            badges.append('<span class="badge badge-warn">egress</span>')
        else:
            badges.append('<span class="badge badge-ok">no-egress</span>')
        if entry.runtime.ui_port:
            badges.append('<span class="badge badge-muted">UI</span>')

        if already:
            action_html = '<span class="badge badge-ok">installed</span>'
        else:
            action_html = (
                f'<form class="inline" method="post" action="/admin/modules/install">'
                f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                f'<input type="hidden" name="slug" value="{_e(entry.slug)}">'
                f'{cfg_inputs}'
                f'<button type="submit" class="sm">Install</button>'
                f'</form>'
            )
        cat_rows.append(
            f"<tr>"
            f"<td data-label=\"Module\" class=\"cell-wide\"><strong>{_e(entry.name)}</strong><br><code>{_e(entry.slug)}</code> "
            f"<span class=\"muted\">v{_e(entry.version)}</span></td>"
            f"<td data-label=\"Description\" class=\"cell-wide\">{_e(entry.description)}<div class=\"row\">{' '.join(badges)}</div></td>"
            f"<td data-label=\"Action\" class=\"cell-wide\">{action_html}</td>"
            f"</tr>"
        )

    catalog_table = (
        f'<div class="scroll-x"><table class="stack"><thead><tr>'
        f'<th>Module</th><th>Description</th><th>Action</th>'
        f'</tr></thead><tbody>{"".join(cat_rows)}</tbody></table></div>'
        if cat_rows
        else '<p class="empty">No container module manifests found in <code>catalog/modules/</code>.</p>'
    )

    body = f"""
    <h1>Modules</h1>
    <p class="muted">Active module host: <code>{_e(host_kind)}</code></p>
    {banners}
    <div class="card">
      <h2>Installed Modules</h2>
      {installed_table}
    </div>
    <div class="card">
      <h2>Available Catalog Modules</h2>
      {catalog_table}
    </div>
    """
    return _layout(
        title="Modules",
        active="modules",
        csrf_token=csrf_token,
        body=body,
        pending_approvals=pending_approvals,
        ui_modules=ui_modules,
    )
