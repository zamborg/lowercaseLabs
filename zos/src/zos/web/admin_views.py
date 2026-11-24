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


def render_overview_page(*, overview: dict[str, Any], csrf_token: str) -> str:
    providers = overview["providers"]
    online_count = sum(1 for p in providers if p["online"])
    prov_rows = []
    for p in providers:
        badge = (
            '<span class="badge badge-ok">online</span>'
            if p["online"]
            else '<span class="badge badge-muted">offline</span>'
        )
        backend = _e(p["backend"]) if p.get("backend") else "—"
        prov_rows.append(
            f"<tr>"
            f"<td><code>{_e(p['namespace'])}</code></td>"
            f"<td>{_e(p['kind'])}</td>"
            f"<td><code>{backend}</code></td>"
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

    body = f"""
    <h1>Overview</h1>
    <div class="stats">
      <div class="stat"><div class="label">Active MCP Sessions</div><div class="value">{int(overview['active_sessions'])}</div></div>
      <div class="stat"><div class="label">Online Providers</div><div class="value">{online_count} / {len(providers)}</div></div>
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


def render_clients_page(
    *,
    clients: list[dict[str, Any]],
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
    new_token_info: dict[str, str] | None = None,
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
      <h2>Registered Clients</h2>
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

        status_dot = "" if t["online"] else ' <span class="badge badge-muted">offline</span>'
        tool_rows.append(
            f"<tr>"
            f"<td><code>{_e(t['name'])}</code>{status_dot}<div class=\"muted\">{_e(t['provider'])}</div></td>"
            f"<td>{tier_badge}</td>"
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
