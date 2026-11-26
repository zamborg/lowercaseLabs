"""Server-rendered HTML templates for the ZOS owner admin (/admin)."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

from zos.web.pages import render_auth_shell
from zos.web.theme import BRAND_MARK
from zos.web.theme import CSS as _CSS

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _rel(dt: datetime, now: datetime) -> str:
    secs = (now - dt).total_seconds()
    future, s = secs < 0, abs(secs)
    if s < 45:
        return "just now" if not future else "in a moment"
    if s < 3600:
        n, unit = max(1, round(s / 60)), "m"
    elif s < 86400:
        n, unit = round(s / 3600), "h"
    elif s < 7 * 86400:
        n, unit = round(s / 86400), "d"
    else:
        day = f"{dt:%b} {dt.day}"
        return day if dt.year == now.year else f"{day}, {dt.year}"
    return f"in {n}{unit}" if future else f"{n}{unit} ago"


def _ts(value: Any) -> str:
    """Relative time with the exact UTC timestamp on hover."""
    if value is None:
        return '<span class="faint">—</span>'
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=UTC)
        full = dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f'<time datetime="{_e(dt.isoformat())}" title="{full}">{_rel(dt, datetime.now(UTC))}</time>'
    return f'<time title="{_e(value)}">{_e(value)}</time>'


def _fmt_bytes(size: int | None) -> str:
    if size is None:
        return "—"
    val = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024.0 or unit == "GB":
            return f"{int(val)} B" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{size} B"


def _csrf(token: str) -> str:
    return f'<input type="hidden" name="csrf_token" value="{_e(token)}">'


def _hidden(**fields: Any) -> str:
    return "".join(
        f'<input type="hidden" name="{name}" value="{_e(val)}">' for name, val in fields.items() if val is not None
    )


def _post(action: str, csrf: str, inner: str, *, cls: str = "inline", **fields: Any) -> str:
    return f'<form class="{cls}" method="post" action="{action}">{_csrf(csrf)}{_hidden(**fields)}{inner}</form>'


def _confirm(label: str, action: str, csrf: str, confirm_label: str, **fields: Any) -> str:
    """Two-step destructive action without JS: the first click reveals the real button."""
    button = f'<button type="submit" class="danger solid sm">{confirm_label}</button>'
    return (
        f'<details class="disclose"><summary class="btn ghost sm">{label}</summary>'
        f'<div class="disclose-body">{_post(action, csrf, button, **fields)}</div></details>'
    )


def _badge(text: str, kind: str = "muted") -> str:
    return f'<span class="badge badge-{kind}">{_e(text)}</span>'


def _notices(notice: str | None, error: str | None) -> str:
    out = ""
    if error:
        out += f'<div class="notice notice-err" role="alert"><div>{_e(error)}</div></div>'
    if notice:
        out += f'<div class="notice notice-ok" role="status"><div>{_e(notice)}</div></div>'
    return out


def _head(title: str, lede: str = "", actions: str = "", back: tuple[str, str] | None = None) -> str:
    back_html = f'<a class="back" href="{back[0]}">&larr; {_e(back[1])}</a>' if back else ""
    lede_html = f'<p class="lede">{lede}</p>' if lede else ""
    actions_html = f'<div class="row">{actions}</div>' if actions else ""
    return f'{back_html}<header class="page-head"><div><h1>{title}</h1>{lede_html}</div>{actions_html}</header>'


def _section(title: str, body: str, *, meta: str = "", desc: str = "", aside: str = "") -> str:
    meta_html = f'<span class="meta">{meta}</span>' if meta else ""
    aside_html = f'<div class="aside">{aside}</div>' if aside else ""
    desc_html = f'<p class="section-desc">{desc}</p>' if desc else ""
    return (
        f'<section class="section"><div class="section-head"><h2>{title}</h2>{meta_html}{aside_html}</div>'
        f"{desc_html}{body}</section>"
    )


def _table(headers: list[str | tuple[str, str]], rows: list[str], *, empty: str, cls: str = "stack") -> str:
    ths = "".join(f'<th class="{h[1]}">{h[0]}</th>' if isinstance(h, tuple) else f"<th>{h}</th>" for h in headers)
    body = "".join(rows) or f'<tr><td colspan="{len(headers)}" class="empty">{empty}</td></tr>'
    table = f'<table class="{cls}"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    # Stacked tables reflow into cards on narrow screens, so they never need to scroll
    # (and must not clip row popovers); everything else scrolls inside its panel.
    return f'<div class="panel">{table if cls == "stack" else f"<div class=scroll-x>{table}</div>"}</div>'


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------


def _layout(
    *,
    title: str,
    active: str,
    csrf_token: str,
    body: str,
    pending_approvals: int = 0,
    ui_modules: list[dict[str, Any]] | None = None,
    nav: dict[str, Any] | None = None,
    wide: bool = False,
) -> str:
    if nav:
        pending_approvals = int(nav.get("pending", pending_approvals))
        ui_modules = nav.get("ui_modules") or ui_modules
    count = f'<span class="count">{int(pending_approvals)}</span>' if pending_approvals > 0 else ""
    groups: list[tuple[str | None, list[tuple[str, str, str]]]] = [
        (
            None,
            [
                ("overview", "/admin", "Overview"),
                ("approvals", "/admin/approvals", f"Approvals{count}"),
                ("activity", "/admin/activity", "Activity"),
            ],
        ),
        (
            "Connect",
            [
                ("modules", "/admin/modules", "Modules"),
                ("upstreams", "/admin/upstreams", "MCP servers"),
                ("clients", "/admin/clients", "Clients"),
                ("policy", "/admin/policy", "Tools &amp; policy"),
            ],
        ),
    ]
    if ui_modules:
        groups.append(
            (
                "Apps",
                [
                    (
                        f"m:{m['slug']}" if "slug" in m else "",
                        _e(m["href"]),
                        _e(m["title"]) + ("" if m.get("online", True) else '<span class="dot" title="offline"></span>'),
                    )
                    for m in ui_modules
                ],
            )
        )
    nav_html = ""
    for label, items in groups:
        if label:
            nav_html += f'<span class="nav-label">{label}</span>'
        for key, href, text in items:
            cur = ' class="active" aria-current="page"' if key == active else ""
            nav_html += f'<a href="{href}"{cur}>{text}</a>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{_e(title)} · zos</title>
  <style>{_CSS}</style>
</head>
<body>
  <a class="skip" href="#main">Skip to content</a>
  <div class="shell">
    <aside class="side">
      <a class="brand" href="/admin">{BRAND_MARK}zos</a>
      <nav class="nav" aria-label="Admin">{nav_html}</nav>
      <div class="side-foot">
        <form method="post" action="/admin/logout">{_csrf(csrf_token)}<button type="submit" class="ghost sm">Sign out</button></form>
      </div>
    </aside>
    <main class="content{" wide" if wide else ""}" id="main">{body}</main>
  </div>
</body>
</html>
"""


def render_admin_login_page(*, error: str | None = None, next_url: str | None = None) -> str:
    err_html = f'<div class="notice notice-err" role="alert"><div>{_e(error)}</div></div>' if error else ""
    next_input = f'<input type="hidden" name="next" value="{_e(next_url)}">' if next_url else ""
    card = f"""
      <h1>Sign in</h1>
      <p class="lede">Enter your owner passphrase to open the control panel.</p>
      {err_html}
      <form class="auth-form" method="post" action="/admin/login">
        {next_input}
        <label class="field">Owner passphrase
          <input name="passphrase" type="password" autocomplete="current-password" required autofocus>
        </label>
        <button type="submit" class="block">Sign in</button>
      </form>"""
    return render_auth_shell(title="Sign in", card=card)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def render_overview_page(
    *,
    overview: dict[str, Any],
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
    ui_modules: list[dict[str, Any]] | None = None,
    nav: dict[str, Any] | None = None,
) -> str:
    providers = overview["providers"]
    online_count = sum(1 for p in providers if p["online"])
    pending_count = int(overview.get("pending_approvals_count", 0))
    sessions = int(overview["active_sessions"])

    prov_rows = []
    for p in providers:
        badge = _badge("online", "ok") if p["online"] else _badge("offline")
        backend = p.get("backend")
        backend_cell = f'<code class="muted">{_e(backend)}</code>' if backend else '<span class="faint">—</span>'
        prov_rows.append(
            "<tr>"
            f'<td class="lead" data-label="Namespace"><code>{_e(p["namespace"])}</code></td>'
            f'<td data-label="Kind" class="muted">{_e(p["kind"])}</td>'
            f'<td data-label="Backend">{backend_cell}</td>'
            f'<td data-label="Status">{badge}</td>'
            f'<td data-label="Last seen" class="num muted">{_ts(p["last_seen_at"])}</td>'
            "</tr>"
        )

    last_backup = overview.get("last_backup")
    if last_backup:
        backup_val = (
            f'{_e(_fmt_bytes(last_backup.get("size_bytes")))}'
            f'<span class="sub muted">{_ts(last_backup.get("completed_at"))}</span>'
        )
    else:
        backup_val = '<span class="faint">Never</span>'

    pending_val = (
        f'<a class="value warn" href="/admin/approvals">{pending_count}</a>'
        if pending_count
        else '<div class="value">0</div>'
    )
    stats = f"""
    <div class="stats">
      <div class="stat"><div class="label">Active sessions</div><div class="value">{sessions}</div></div>
      <div class="stat"><div class="label">Providers online</div><div class="value">{online_count}<span class="of"> / {len(providers)}</span></div></div>
      <div class="stat"><div class="label">Pending approvals</div>{pending_val}</div>
      <div class="stat"><div class="label">Last backup</div><div class="value">{backup_val}</div></div>
    </div>"""

    apps = ""
    if ui_modules:
        tiles = "".join(
            f'<a class="tile" href="{_e(m["href"])}"><div class="tile-head">'
            f'<span class="name">{_e(m["title"])}</span>'
            f'<span class="aside"><span class="dot{" on" if m.get("online", True) else ""}"></span></span>'
            f'</div><span class="sub mono">{_e(m["href"])}</span></a>'
            for m in ui_modules
        )
        apps = _section("Apps", f'<div class="tiles">{tiles}</div>')

    offline = len(providers) - online_count
    lede_bits = [
        f"{online_count} of {len(providers)} providers online" if offline else "All providers online",
        f'<a href="/admin/approvals">{pending_count} pending approval{"s" if pending_count != 1 else ""}</a>'
        if pending_count
        else "nothing waiting on you",
    ]
    body = (
        _head("Overview", " · ".join(lede_bits))
        + _notices(notice, error)
        + stats
        + apps
        + _section(
            "Providers",
            _table(
                ["Namespace", "Kind", "Backend", "Status", ("Last seen", "num")],
                prov_rows,
                empty="No providers registered.",
            ),
            meta=str(len(providers)),
        )
    )
    return _layout(
        title="Overview",
        active="overview",
        csrf_token=csrf_token,
        body=body,
        pending_approvals=pending_count,
        ui_modules=ui_modules,
        nav=nav,
    )


# ---------------------------------------------------------------------------
# MCP servers
# ---------------------------------------------------------------------------

_UPSTREAM_STATUS = {
    "online": ("online", "ok"),
    "needs_reconnect": ("needs reconnect", "err"),
    "not_configured": ("not configured", "warn"),
    "offline": ("offline", "muted"),
}


def render_upstreams_page(
    *,
    upstreams: list[dict[str, Any]],
    csrf_token: str,
    catalog: list[dict[str, Any]] | None = None,
    notice: str | None = None,
    error: str | None = None,
    nav: dict[str, Any] | None = None,
) -> str:
    installed_slugs = {str(u["slug"]) for u in upstreams}
    installed_catalog_slugs = {str(u["catalog_slug"]) for u in upstreams if u.get("catalog_slug")}

    rows = []
    for u in upstreams:
        slug = str(u["slug"])
        status = str(u["status"])
        label, kind = _UPSTREAM_STATUS.get(status, (status, "muted"))
        err_note = f'<span class="sub err-text">{_e(u["last_error"])}</span>' if u.get("last_error") else ""
        help_note = ""
        if u["auth_mode"] == "bearer" and u.get("token_help") and not u.get("has_credential"):
            help_note = f'<span class="sub">{_e(u["token_help"])}</span>'
        elif u["auth_mode"] == "oauth" and u.get("requires_client_registration") and u.get("client_help"):
            help_note = f'<span class="sub">{_e(u["client_help"])}</span>'

        actions: list[str] = []
        if u["auth_mode"] == "bearer":
            token_form = _post(
                "/admin/upstreams/token",
                csrf_token,
                '<input type="password" name="token" placeholder="Paste token" autocomplete="new-password" required aria-label="Bearer token">'
                '<button type="submit" class="sm">Save</button>',
                slug=slug,
            )
            actions.append(
                f'<details class="disclose"><summary class="btn secondary sm">'
                f'{"Replace token" if u.get("has_credential") else "Set token"}</summary>'
                f'<div class="disclose-body">{token_form}</div></details>'
            )
            if u.get("has_credential"):
                actions.append(
                    _post("/admin/upstreams/disconnect", csrf_token, '<button type="submit" class="ghost sm">Disconnect</button>', slug=slug)
                )
        elif u["auth_mode"] == "oauth":
            if u.get("requires_client_registration"):
                client_form = _post(
                    "/admin/upstreams/oauth-client",
                    csrf_token,
                    '<input type="text" name="client_id" placeholder="Client ID" required aria-label="Client ID">'
                    '<input type="password" name="client_secret" placeholder="Client secret" autocomplete="new-password" required aria-label="Client secret">'
                    '<button type="submit" class="sm">Save</button>',
                    slug=slug,
                )
                actions.append(
                    f'<details class="disclose"><summary class="btn secondary sm">OAuth client</summary>'
                    f'<div class="disclose-body">{client_form}</div></details>'
                )
            reconnect = status == "needs_reconnect"
            if reconnect or status != "online":
                actions.append(f'<a class="btn sm" href="/connect/{_e(slug)}">{"Reconnect" if reconnect else "Connect"}</a>')
            if u.get("has_credential") or u.get("online"):
                actions.append(
                    _post("/admin/upstreams/disconnect", csrf_token, '<button type="submit" class="ghost sm">Disconnect</button>', slug=slug)
                )
        actions.append(_confirm("Remove", "/admin/upstreams/remove", csrf_token, f"Remove {_e(slug)}", slug=slug))
        rows.append(
            "<tr>"
            f'<td class="lead" data-label="Server"><code>{_e(slug)}</code><span class="sub mono">{_e(u["url"])}</span>{help_note}</td>'
            f'<td data-label="Auth" class="muted">{_e(u["auth_mode"])}</td>'
            f'<td data-label="Default tier" class="muted">{_e(u["default_tier"])}</td>'
            f'<td data-label="Status">{_badge(label, kind)}{err_note}</td>'
            f'<td data-label="Tools" class="num">{int(u["tool_count"])}</td>'
            f'<td class="actions"><div class="row">{"".join(actions)}</div></td>'
            "</tr>"
        )

    tiles = []
    for c in catalog or []:
        c_slug = str(c["slug"])
        is_installed = c_slug in installed_slugs or c_slug in installed_catalog_slugs
        auth_label = f"{c['auth_mode']} · pre-registered" if c.get("requires_client_registration") else c["auth_mode"]
        install = _post(
            "/admin/upstreams/install",
            csrf_token,
            f'<button type="submit" class="{"ghost" if is_installed else "secondary"} sm">{"Install another" if is_installed else "Install"}</button>',
            catalog_slug=c_slug,
        )
        alias = (
            '<details class="disclose"><summary class="btn ghost sm">Install as…</summary><div class="disclose-body">'
            + _post(
                "/admin/upstreams/install",
                csrf_token,
                '<input type="text" name="as_slug" placeholder="alias" pattern="[a-z][a-z0-9]{1,15}" aria-label="Alias" required>'
                '<button type="submit" class="sm">Install</button>',
                catalog_slug=c_slug,
            )
            + "</div></details>"
        )
        tiles.append(
            f'<div class="tile"><div class="tile-head"><span class="name">{_e(c["name"])}</span>'
            f'<span class="aside">{_badge("installed", "ok") if is_installed else install}</span></div>'
            f'<div class="desc">{_e(c["description"])}</div>'
            f'<div class="foot"><code class="faint small">{_e(c_slug)}</code>'
            f'<span class="faint small">{_e(auth_label)}</span>'
            f'<a class="faint small" href="{_e(c["docs_url"])}" target="_blank" rel="noopener noreferrer">Docs ↗</a>'
            f'<span class="aside">{alias if not is_installed else install}</span></div></div>'
        )
    catalog_html = (
        f'<div class="tiles">{"".join(tiles)}</div>'
        if tiles
        else '<div class="panel"><p class="empty">No catalog manifests found.</p></div>'
    )

    add_form = f"""
    <div class="panel"><div class="panel-body">
      <form class="form-row" method="post" action="/admin/upstreams/add">
        {_csrf(csrf_token)}
        <label class="field">Slug
          <input type="text" name="slug" placeholder="github" pattern="[a-z][a-z0-9]{{1,15}}" required>
        </label>
        <label class="field grow">Remote MCP URL
          <input type="text" name="url" placeholder="https://api.githubcopilot.com/mcp/" required>
        </label>
        <label class="field">Auth
          <select name="auth_mode">
            <option value="oauth">oauth</option>
            <option value="bearer">bearer</option>
            <option value="none">none</option>
          </select>
        </label>
        <label class="field">Default tier
          <select name="default_tier">
            <option value="external" selected>external</option>
            <option value="write">write</option>
            <option value="read">read</option>
          </select>
        </label>
        <button type="submit">Add server</button>
      </form>
    </div></div>"""

    body = (
        _head("MCP servers", "Remote MCP servers routed through zos. Their tools appear to clients as <code>slug_tool</code>.")
        + _notices(notice, error)
        + _section(
            "Connected",
            _table(
                ["Server", "Auth", "Default tier", "Status", ("Tools", "num"), ("", "actions")],
                rows,
                empty="No remote servers yet. Install one from the catalog below.",
            ),
            meta=str(len(upstreams)),
        )
        + _section(
            "Available connectors",
            catalog_html,
            meta=str(len(tiles)),
            desc="Verified official remote MCP servers. Installing registers the server with its manifest tier rules; connect it afterwards.",
        )
        + _section(
            "Custom server",
            add_form,
            desc='Default tier <code>external</code> requires approval for every call. Override individual tools in <a href="/admin/policy">Tools &amp; policy</a>.',
        )
    )
    return _layout(title="MCP servers", active="upstreams", csrf_token=csrf_token, body=body, nav=nav)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


def render_clients_page(
    *,
    clients: list[dict[str, Any]],
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
    new_token_info: dict[str, str] | None = None,
    max_oauth_clients: int = 50,
    allowed_redirect_hosts: list[dict[str, Any]] | None = None,
    rejected_attempts: list[dict[str, Any]] | None = None,
    nav: dict[str, Any] | None = None,
) -> str:
    banners = _notices(notice, error)
    if new_token_info:
        banners += f"""
        <div class="notice notice-token"><div>
          New bearer token for <strong>{_e(new_token_info['client_name'])}</strong>. Copy it now — it won't be shown again.
          <div class="token-box">{_e(new_token_info['token'])}</div>
        </div></div>"""

    oauth_count = sum(1 for c in clients if c["kind"] == "oauth")
    rows = []
    for c in clients:
        cid = str(c["id"])
        trust = _badge("untrusted", "warn") if c["trust_ceiling"] == "untrusted" else _badge("owner", "muted")
        actions = []
        if c["kind"] == "bearer":
            actions.append(
                _post("/admin/clients/token", csrf_token, '<button type="submit" class="ghost sm">New token</button>', client_name=c["name"])
            )
            actions.append(
                _post("/admin/clients/disable", csrf_token, '<button type="submit" class="ghost sm">Disable</button>', client_id=cid)
            )
        else:
            actions.append(
                _post("/admin/clients/delete", csrf_token, '<button type="submit" class="ghost sm">Prune</button>', client_id=cid)
            )
        actions.append(_confirm("Revoke", "/admin/clients/revoke", csrf_token, "Revoke all tokens", client_id=cid))
        rows.append(
            "<tr>"
            f'<td class="lead" data-label="Client"><strong>{_e(c["name"])}</strong><span class="sub mono">{_e(c["oauth_client_id"] or cid)}</span></td>'
            f'<td data-label="Kind" class="muted">{_e(c["kind"])}</td>'
            f'<td data-label="Trust ceiling">{trust}</td>'
            f'<td data-label="Active tokens" class="num">{int(c["active_token_count"])}</td>'
            f'<td data-label="Last seen" class="num muted">{_ts(c["last_seen_at"])}</td>'
            f'<td class="actions"><div class="row">{"".join(actions)}</div></td>'
            "</tr>"
        )

    blocked = ""
    attempts = rejected_attempts or []
    if attempts:
        attempt_rows = []
        for att in attempts:
            att_host = str(att["host"])
            attempt_rows.append(
                "<tr>"
                f'<td class="lead" data-label="Attempt"><strong>{_e(att["client_name"])}</strong> tried to connect from <code>{_e(att_host)}</code></td>'
                f'<td data-label="When" class="num muted">{_ts(att["created_at"])}</td>'
                '<td class="actions"><div class="row">'
                + _post("/admin/clients/redirect-hosts/dismiss", csrf_token, '<button type="submit" class="ghost sm">Dismiss</button>', attempt_id=int(att["id"]))
                + _post("/admin/clients/redirect-hosts/allow", csrf_token, '<button type="submit" class="sm">Allow host</button>', host=att_host)
                + "</div></td></tr>"
            )
        blocked = _section(
            "Blocked connection attempts",
            _table(["Attempt", ("When", "num"), ("", "actions")], attempt_rows, empty=""),
            meta=str(len(attempts)),
            desc="OAuth clients whose redirect host isn't on your allowlist yet.",
        )

    host_rows = []
    for h in allowed_redirect_hosts or []:
        h_name = str(h["host"])
        host_rows.append(
            "<tr>"
            f'<td class="lead" data-label="Host"><code>{_e(h_name)}</code></td>'
            f'<td data-label="Source" class="muted">{_e(h.get("source", "manual"))}</td>'
            f'<td data-label="Added" class="num muted">{_ts(h.get("created_at"))}</td>'
            f'<td class="actions"><div class="row">{_confirm("Remove", "/admin/clients/redirect-hosts/remove", csrf_token, "Remove host", host=h_name)}</div></td>'
            "</tr>"
        )
    allow_form = _post(
        "/admin/clients/redirect-hosts/allow",
        csrf_token,
        '<input type="text" name="host" placeholder="chatgpt.com" required aria-label="Hostname">'
        '<button type="submit" class="secondary sm">Allow host</button>',
    )

    create_form = f"""
    <div class="panel"><div class="panel-body">
      <form class="form-row" method="post" action="/admin/clients/create">
        {_csrf(csrf_token)}
        <label class="field grow">Name
          <input type="text" name="name" placeholder="codex-cli" required>
        </label>
        <label class="field">Trust ceiling
          <select name="trust_ceiling">
            <option value="owner">owner</option>
            <option value="untrusted">untrusted</option>
          </select>
        </label>
        <button type="submit">Create &amp; mint token</button>
      </form>
    </div></div>"""

    body = (
        _head("Clients", "Agents and apps that can reach your MCP endpoint, and the redirect hosts OAuth clients may use.")
        + banners
        + blocked
        + _section(
            "Registered clients",
            _table(
                ["Client", "Kind", "Trust", ("Tokens", "num"), ("Last seen", "num"), ("", "actions")],
                rows,
                empty="No clients yet.",
            ),
            meta=f"OAuth: {oauth_count} / {int(max_oauth_clients)}",
        )
        + _section("New bearer client", create_form, desc="For CLIs and scripts. The token is shown once.")
        + _section(
            "OAuth redirect hosts",
            _table(["Host", "Source", ("Added", "num"), ("", "actions")], host_rows, empty="No external hosts allowed."),
            meta=str(len(host_rows)),
            desc="Loopback (<code>localhost</code>, <code>127.0.0.1</code>, <code>::1</code>) is always allowed. External HTTPS clients must match one of these.",
            aside=allow_form,
        )
    )
    return _layout(title="Clients", active="clients", csrf_token=csrf_token, body=body, nav=nav)


# ---------------------------------------------------------------------------
# Tools & policy
# ---------------------------------------------------------------------------

_DECISIONS = (("allow", "allow", "ok"), ("require_approval", "ask", "warn"), ("deny", "deny", "err"))
_TIERS = (("read", "read", "ok"), ("write", "write", "info"), ("external", "external", "warn"))


def _decision_badge(decision: str) -> str:
    kind = {"allow": "ok", "require_approval": "warn", "deny": "err"}.get(decision, "muted")
    return _badge(decision, kind)


def _decision_seg(*, csrf_token: str, pattern: str, client_id: str | None, effective: str | None, explicit: str | None) -> str:
    """One form, one click per decision: allow / ask / deny, plus reset when overridden."""
    btns = ""
    for val, label, kind in _DECISIONS:
        state = " pin" if explicit == val else (" on" if explicit is None and effective == val else "")
        btns += f'<button type="submit" name="decision" value="{val}" class="{kind}{state}" title="{val}">{label}</button>'
    if explicit is not None:
        btns += '<button type="submit" formaction="/admin/policy/clear" class="reset" title="Clear override" aria-label="Clear override">&times;</button>'
    else:
        btns += '<span class="reset-ph" aria-hidden="true">&times;</span>'
    return _post("/admin/policy/set", csrf_token, f'<div class="seg">{btns}</div>', cls="seg-form", client_id=client_id, pattern=pattern)


def render_policy_page(
    *,
    matrix: dict[str, Any],
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
    nav: dict[str, Any] | None = None,
) -> str:
    clients = matrix["clients"]
    tools = matrix["tools"]
    custom_rules = matrix["custom_rules"]

    client_headers = "".join(
        f'<th class="client">{_e(c["name"])}<span class="sub">{_e(c["trust_ceiling"])}</span></th>' for c in clients
    )

    tool_rows = []
    for t in tools:
        hints = [
            h
            for h, on in (("read-only", t["read_only_hint"]), ("destructive", t["destructive_hint"]), ("open-world", t["open_world_hint"]))
            if on
        ]
        tier_btns = "".join(
            f'<button type="submit" name="tier" value="{val}" class="{kind}{" pin" if t["tier"] == val else ""}">{label}</button>'
            for val, label, kind in _TIERS
        )
        tier_cell = _post("/admin/policy/tier", csrf_token, f'<div class="seg">{tier_btns}</div>', cls="seg-form", pattern=t["name"])
        global_cell = _decision_seg(
            csrf_token=csrf_token,
            pattern=t["name"],
            client_id=None,
            effective=None,
            explicit=t["global_override"],
        )
        client_cells = "".join(
            "<td>"
            + _decision_seg(
                csrf_token=csrf_token,
                pattern=t["name"],
                client_id=str(cell["client_id"]),
                effective=cell["effective"],
                explicit=cell["explicit"],
            )
            + "</td>"
            for cell in t["per_client"]
        )
        offline = "" if t["online"] else ' <span class="badge badge-muted">offline</span>'
        sub = " · ".join([_e(t["provider"]), *hints])
        tool_rows.append(
            f'<tr class="{"" if t["online"] else "offline"}">'
            f'<td><code>{_e(t["name"])}</code>{offline}<span class="sub">{sub}</span></td>'
            f"<td>{tier_cell}</td><td>{global_cell}</td>{client_cells}</tr>"
        )

    matrix_html = (
        f'<div class="panel"><div class="scroll-x"><table class="matrix"><thead><tr>'
        f'<th>Tool</th><th>Tier</th><th class="client">All clients<span class="sub">global</span></th>{client_headers}'
        f'</tr></thead><tbody>{"".join(tool_rows) or f"<tr><td colspan={3 + len(clients)} class=empty>No tools registered.</td></tr>"}</tbody></table></div></div>'
    )
    legend = (
        '<div class="legend">'
        '<span><span class="seg"><button type="button" class="ok on">allow</button></span> inherited</span>'
        '<span><span class="seg"><button type="button" class="ok pin">allow</button></span> override</span>'
        "</div>"
    )

    custom_rows = []
    for r in custom_rules:
        custom_rows.append(
            "<tr>"
            f'<td class="lead" data-label="Pattern"><code>{_e(r["pattern"])}</code></td>'
            f'<td data-label="Client">{_e(r["client_name"])}</td>'
            f'<td data-label="Decision">{_decision_badge(r["decision"])}</td>'
            '<td class="actions"><div class="row">'
            + _post(
                "/admin/policy/clear",
                csrf_token,
                '<button type="submit" class="ghost sm">Clear</button>',
                client_id=r["client_id"] or None,
                pattern=r["pattern"],
            )
            + "</div></td></tr>"
        )

    client_options = '<option value="">All clients</option>' + "".join(
        f'<option value="{_e(c["id"])}">{_e(c["name"])} ({_e(c["kind"])})</option>' for c in clients
    )
    pattern_form = f"""
    <div class="panel"><div class="panel-body">
      <form class="form-row" method="post" action="/admin/policy/set">
        {_csrf(csrf_token)}
        <label class="field grow">Tool pattern
          <input type="text" name="pattern" placeholder="mac_*  or  gmail_send" required>
        </label>
        <label class="field">Client
          <select name="client_id">{client_options}</select>
        </label>
        <label class="field">Decision
          <select name="decision">
            <option value="allow">allow</option>
            <option value="require_approval">ask (require approval)</option>
            <option value="deny">deny</option>
          </select>
        </label>
        <button type="submit">Add rule</button>
      </form>
    </div></div>"""

    rules_table = _table(["Pattern", "Client", "Decision", ("", "actions")], custom_rows, empty="") if custom_rows else ""
    body = (
        _head(
            "Tools &amp; policy",
            "Each tool has a tier. Override the decision for every client, or for one client. <em>ask</em> holds the call until you approve it.",
        )
        + _notices(notice, error)
        + _section("Tools", matrix_html, meta=str(len(tools)), aside=legend)
        + _section("Pattern rules", f'<div class="vstack">{rules_table}{pattern_form}</div>', meta=str(len(custom_rows)) if custom_rows else "",
                   desc="Glob patterns apply to tools that match, including ones that appear later.")
    )
    return _layout(title="Tools & policy", active="policy", csrf_token=csrf_token, body=body, nav=nav, wide=True)


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


def render_activity_page(
    *,
    data: dict[str, Any],
    filters: dict[str, str],
    csrf_token: str,
    nav: dict[str, Any] | None = None,
) -> str:
    def opts(all_label: str, items: list[tuple[str, str]], key: str) -> str:
        return f'<option value="">{all_label}</option>' + "".join(
            f'<option value="{_e(v)}"{" selected" if filters.get(key) == str(v) else ""}>{_e(label)}</option>'
            for v, label in items
        )

    client_opts = opts("All clients", [(str(c["id"]), c["name"]) for c in data["filter_clients"]], "client_id")
    prov_opts = opts("All providers", [(str(p["id"]), p["slug"]) for p in data["filter_providers"]], "provider_id")
    type_opts = opts("All types", [(t, t) for t in data["filter_types"]], "type")
    outcome_opts = opts("All outcomes", [(o, o) for o in ("ok", "denied", "error")], "outcome")

    rows = []
    for ev in data["events"]:
        kind = {"ok": "ok", "denied": "warn", "error": "err"}.get(ev["outcome"], "muted")
        digest_hex = ev["args_digest_hex"]
        digest = f'<code class="faint" title="{_e(digest_hex)}">{_e(digest_hex[:10])}</code>' if digest_hex else '<span class="faint">—</span>'
        tool = ev["tool"]
        what = (
            f'<code>{_e(tool)}</code><span class="sub">{_e(ev["provider_slug"] or "")}{" · " if ev["provider_slug"] else ""}{_e(ev["type"])}</span>'
            if tool
            else f'<span class="muted">{_e(ev["type"])}</span>'
        )
        rows.append(
            "<tr>"
            f'<td class="lead" data-label="Event">{what}</td>'
            f'<td data-label="Client">{_e(ev["client_name"] or "—")}</td>'
            f'<td data-label="Decision" class="muted">{_e(ev["decision"] or "—")}</td>'
            f'<td data-label="Outcome">{_badge(ev["outcome"], kind)}</td>'
            f'<td data-label="Args">{digest}</td>'
            f'<td data-label="Time" class="num muted">{_ts(ev["created_at"])}</td>'
            f'<td data-label="ID" class="num faint">{int(ev["id"])}</td>'
            "</tr>"
        )

    pager = ""
    if data["next_before_id"] is not None:
        qp = {k: v for k, v in filters.items() if v}
        qp["before_id"] = str(data["next_before_id"])
        pager = f'<a class="btn secondary sm" href="/admin/activity?{urlencode(qp)}">Older &rarr;</a>'

    filtered = any(v for v in filters.values())
    filter_bar = f"""
    <form class="form-row" method="get" action="/admin/activity" style="margin-bottom: 14px">
      <label class="field">Client<select name="client_id">{client_opts}</select></label>
      <label class="field">Provider<select name="provider_id">{prov_opts}</select></label>
      <label class="field">Type<select name="type">{type_opts}</select></label>
      <label class="field">Outcome<select name="outcome">{outcome_opts}</select></label>
      <button type="submit" class="secondary">Filter</button>
      {'<a class="btn ghost" href="/admin/activity">Reset</a>' if filtered else ""}
    </form>"""
    table = _table(
        ["Event", "Client", "Decision", "Outcome", "Args", ("Time", "num"), ("ID", "num")],
        rows,
        empty="No matching events.",
    )
    foot = f'<div class="panel-foot" style="border:0;padding:12px 2px"><span>Arguments are never stored — only SHA-256 digests.</span>{pager}</div>'
    body = _head("Activity", "Append-only audit trail of every call, decision and sign-in.") + filter_bar + table + foot
    return _layout(title="Activity", active="activity", csrf_token=csrf_token, body=body, nav=nav)


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


def _approval_status_badge(status: str) -> str:
    kind = {"pending": "warn", "approved": "ok", "consumed": "ok", "denied": "err", "expired": "muted"}.get(status, "muted")
    return _badge(status, kind)


def render_approvals_page(
    *,
    approvals: dict[str, list[dict[str, Any]]],
    csrf_token: str,
    notice: str | None = None,
    error: str | None = None,
    nav: dict[str, Any] | None = None,
) -> str:
    pending = approvals.get("pending") or []
    recent = approvals.get("recent") or []

    pending_rows = [
        "<tr>"
        f'<td class="lead" data-label="Tool"><code>{_e(item["tool"])}</code><span class="sub">from <strong>{_e(item["client_name"])}</strong></span></td>'
        f'<td data-label="Requested" class="num muted">{_ts(item["created_at"])}</td>'
        f'<td data-label="Expires" class="num muted">{_ts(item["expires_at"])}</td>'
        f'<td class="actions"><div class="row"><a class="btn sm" href="/admin/approvals/{_e(item["id"])}">Review</a></div></td>'
        "</tr>"
        for item in pending
    ]
    recent_rows = [
        "<tr>"
        f'<td class="lead" data-label="Tool"><a href="/admin/approvals/{_e(item["id"])}"><code>{_e(item["tool"])}</code></a>'
        f'<span class="sub">from {_e(item["client_name"])}</span></td>'
        f'<td data-label="Status">{_approval_status_badge(item["status"])}</td>'
        f'<td data-label="Requested" class="num muted">{_ts(item["created_at"])}</td>'
        f'<td data-label="Decided" class="num muted">{_ts(item["decided_at"] or item["expires_at"])}</td>'
        "</tr>"
        for item in recent
    ]

    body = (
        _head("Approvals", "Calls held for your decision, from clients that can't prompt you directly.")
        + _notices(notice, error)
        + _section(
            "Waiting",
            _table(["Tool", ("Requested", "num"), ("Expires", "num"), ("", "actions")], pending_rows, empty="Nothing waiting on you."),
            meta=str(len(pending)),
        )
        + _section(
            "Recent",
            _table(["Tool", "Status", ("Requested", "num"), ("Decided", "num")], recent_rows, empty="No decisions yet."),
        )
    )
    return _layout(
        title="Approvals",
        active="approvals",
        csrf_token=csrf_token,
        body=body,
        pending_approvals=len(pending),
        nav=nav,
    )


def render_approval_detail_page(
    *,
    approval: dict[str, Any] | None,
    csrf_token: str,
    pending_approvals: int = 0,
    notice: str | None = None,
    error: str | None = None,
    nav: dict[str, Any] | None = None,
) -> str:
    if approval is None:
        body = (
            _head("Approval not found", "This request doesn't exist or has been removed.", back=("/admin/approvals", "Approvals"))
            + _notices(notice, error)
        )
        return _layout(
            title="Approval not found",
            active="approvals",
            csrf_token=csrf_token,
            body=body,
            pending_approvals=pending_approvals,
            nav=nav,
        )

    aid = approval["id"]
    is_pending = approval["status"] == "pending"
    if is_pending and approval.get("args_pretty") is not None:
        args = f"""
        <pre class="approval-pre">{_e(approval["args_pretty"])}</pre>
        <div class="approval-actions">
          <form method="post" action="/admin/approvals/{_e(aid)}/deny">{_csrf(csrf_token)}<button type="submit" class="deny-btn">Deny</button></form>
          <form method="post" action="/admin/approvals/{_e(aid)}/approve">{_csrf(csrf_token)}<button type="submit" class="approve-btn">Approve</button></form>
        </div>"""
    else:
        args = '<div class="panel"><p class="empty">Arguments are purged once a request is decided or expires.</p></div>'

    digest = approval.get("args_digest_hex", "") or ""
    meta = f"""
    <div class="panel"><div class="panel-body"><div class="meta-grid">
      <div class="meta-item"><div class="label">Status</div><div class="val">{_approval_status_badge(approval["status"])}</div></div>
      <div class="meta-item"><div class="label">Client</div><div class="val"><strong>{_e(approval["client_name"])}</strong></div></div>
      <div class="meta-item"><div class="label">Requested</div><div class="val">{_ts(approval["created_at"])}</div></div>
      <div class="meta-item"><div class="label">Expires</div><div class="val">{_ts(approval["expires_at"])}</div></div>
      <div class="meta-item"><div class="label">Args SHA-256</div><div class="val"><code class="faint" title="{_e(digest)}">{_e(digest[:16])}</code></div></div>
    </div></div></div>"""

    body = (
        _head(f"<code>{_e(approval['tool'])}</code>", "Review the exact arguments before deciding." if is_pending else "", back=("/admin/approvals", "Approvals"))
        + _notices(notice, error)
        + meta
        + _section("Arguments", args)
    )
    return _layout(
        title=f"Approval — {approval['tool']}",
        active="approvals",
        csrf_token=csrf_token,
        body=body,
        pending_approvals=pending_approvals,
        nav=nav,
    )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


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
    nav: dict[str, Any] | None = None,
    devices: list[dict[str, Any]] | None = None,
    new_device_token: dict[str, str] | None = None,
) -> str:
    banners = _notices(notice, error)
    if new_device_token:
        d_slug = new_device_token["slug"]
        d_tok = new_device_token["token"]
        d_ws = new_device_token["ws_url"]
        one_liner = f"zos-device --url {d_ws} --token {d_tok} -- <command...>"
        toml_snippet = (
            f'url = "{d_ws}"\n'
            f'token = "{d_tok}"\n'
            'command = ["your-mcp-server"]'
        )
        banners += (
            '<div class="notice notice-token"><div>'
            f'Device token for <strong>{_e(d_slug)}</strong>. Copy it now — it won\'t be shown again.'
            f'<div class="token-box">{_e(d_tok)}</div>'
            '<p class="small" style="margin:0.75rem 0 0.25rem">Run any stdio MCP server as this device:</p>'
            f'<pre class="code">{_e(one_liner)}</pre>'
            '<p class="small" style="margin:0.75rem 0 0.25rem">Or save as <code>config.toml</code> and run <code>zos-device --config config.toml</code>:</p>'
            f'<pre class="code">{_e(toml_snippet)}</pre>'
            "</div></div>"
        )
    if instructions:
        banners += _section("Manual setup", f'<pre class="code">{_e(instructions)}</pre>', desc="Run this where the module lives, then attach it below.")

    installed_slugs = {str(m["slug"]) for m in installed}

    rows = []
    for m in installed:
        slug = str(m["slug"])
        status = str(m.get("status") or "installed")
        is_disabled = m.get("disabled_at") is not None or status == "disabled"
        if is_disabled:
            badge = _badge("disabled")
        elif status == "pending_attach":
            badge = _badge("pending attach", "warn")
        elif m.get("online"):
            badge = _badge("online", "ok")
        else:
            badge = _badge(str(m.get("upstream_status") or "offline"), "warn")
        tools = int(m.get("tool_count", 0) or 0)

        actions = []
        if m.get("ui_url") and not is_disabled:
            actions.append(f'<a class="btn secondary sm" href="/admin/m/{_e(slug)}/">Open</a>')
        actions.append(_post("/admin/modules/upgrade", csrf_token, '<button type="submit" class="ghost sm">Upgrade</button>', slug=slug))
        toggle = "enable" if is_disabled else "disable"
        actions.append(_post(f"/admin/modules/{toggle}", csrf_token, f'<button type="submit" class="ghost sm">{toggle.title()}</button>', slug=slug))
        uninstall = (
            '<details class="disclose"><summary class="btn ghost sm">Uninstall</summary><div class="disclose-body">'
            + _post("/admin/modules/uninstall", csrf_token, '<button type="submit" class="secondary sm">Keep data</button>', slug=slug)
            + _post("/admin/modules/uninstall", csrf_token, '<button type="submit" class="danger solid sm">Purge data</button>', slug=slug, purge="1")
            + "</div></details>"
        )
        actions.append(uninstall)
        attach = ""
        if status == "pending_attach":
            attach = '<div class="disclose-body">' + _post(
                "/admin/modules/attach",
                csrf_token,
                '<input type="text" name="mcp_url" placeholder="http://127.0.0.1:8081/mcp" aria-label="MCP URL" required>'
                '<input type="text" name="ui_url" placeholder="UI URL (optional)" aria-label="UI URL">'
                '<button type="submit" class="sm">Attach</button>',
                slug=slug,
            ) + "</div>"
        rows.append(
            "<tr>"
            f'<td class="lead" data-label="Module"><code>{_e(slug)}</code><span class="sub">v{_e(m.get("version", ""))}</span></td>'
            f'<td data-label="Host" class="muted">{_e(m.get("host", ""))}</td>'
            f'<td data-label="Status">{badge}</td>'
            f'<td data-label="Tools" class="num">{tools if tools else "<span class=faint>—</span>"}</td>'
            f'<td class="actions"><div class="row">{"".join(actions)}</div>{attach}</td>'
            "</tr>"
        )

    tiles = []
    for entry in catalog:
        already = entry.slug in installed_slugs
        traits = []
        if entry.storage.postgres:
            traits.append("postgres")
        traits.append("network egress" if entry.network.egress else "no egress")
        if entry.runtime.ui_port:
            traits.append("has UI")
        if already:
            action = ""
        else:
            fields = "".join(
                f'<label class="field"><span>{_e(f.name)}{"" if f.required else " <span class=hint>· optional</span>"}</span>'
                f'<input type="{"password" if f.secret else "text"}" name="cfg_{_e(f.name)}" '
                f'placeholder="{_e(f.help or f.name)}"{" required" if f.required else ""}></label>'
                for f in entry.config
            )
            action = _post("/admin/modules/install", csrf_token, f'{fields}<div><button type="submit" class="secondary sm">Install</button></div>', cls="install", slug=entry.slug)
        tiles.append(
            f'<div class="tile"><div class="tile-head"><span class="name">{_e(entry.name)}</span>'
            f'<span class="aside">{_badge("installed", "ok") if already else ""}</span></div>'
            f'<div class="desc">{_e(entry.description)}</div>'
            f'<div class="foot"><code class="faint small">{_e(entry.slug)} v{_e(entry.version)}</code>'
            f'<span class="faint small">{" · ".join(traits)}</span></div>'
            f'{f"<div class=foot>{action}</div>" if action else ""}</div>'
        )

    device_list = devices or []
    device_rows: list[str] = []
    for d in device_list:
        d_slug = str(d["slug"])
        is_enabled = bool(d.get("enabled", True))
        if not is_enabled:
            d_badge = _badge("disabled")
        elif d.get("online"):
            d_badge = _badge("online", "ok")
        else:
            d_badge = _badge("offline", "warn")
        d_tools = int(d.get("tool_count", 0) or 0)
        d_toggle = "disable" if is_enabled else "enable"
        d_actions = [
            _post(
                "/admin/devices/rotate",
                csrf_token,
                '<button type="submit" class="ghost sm">Rotate token</button>',
                slug=d_slug,
            ),
            _post(
                f"/admin/devices/{d_toggle}",
                csrf_token,
                f'<button type="submit" class="ghost sm">{d_toggle.title()}</button>',
                slug=d_slug,
            ),
            _confirm("Remove", "/admin/devices/remove", csrf_token, "Remove device", slug=d_slug),
        ]
        device_rows.append(
            "<tr>"
            f'<td class="lead" data-label="Device"><code>{_e(d_slug)}</code></td>'
            f'<td data-label="Status">{d_badge}</td>'
            f'<td data-label="Last seen" class="num muted">{_ts(d.get("last_seen_at"))}</td>'
            f'<td data-label="Tools" class="num">{d_tools if d_tools else "<span class=faint>—</span>"}</td>'
            f'<td class="actions"><div class="row">{"".join(d_actions)}</div></td>'
            "</tr>"
        )

    add_device_form = f"""
    <div class="panel" style="margin-top:0.75rem"><div class="panel-body">
      <form class="form-row" method="post" action="/admin/devices/add">
        {_csrf(csrf_token)}
        <label class="field grow">Device slug
          <input type="text" name="slug" placeholder="mac" pattern="[a-z][a-z0-9-]{{1,31}}" required>
        </label>
        <button type="submit">Add device</button>
      </form>
    </div></div>"""

    devices_section = _section(
        "Runs on your machine",
        _table(
            ["Device", "Status", ("Last seen", "num"), ("Tools", "num"), ("", "actions")],
            device_rows,
            empty="No devices registered yet. Add a device below to get a one-time token and zos-device command.",
        )
        + add_device_form,
        meta=str(len(device_list)),
        desc="Local stdio MCP servers that dial into zos via <code>zos-device</code>. Their tools appear while connected and default to <code>read</code> (if readOnlyHint) or <code>external</code> (approval required).",
    )

    body = (
        _head("Modules", f"Containers and local devices that add tools and apps to zos. Host: <code>{_e(host_kind)}</code>")
        + banners
        + _section(
            "Installed",
            _table(["Module", "Host", "Status", ("Tools", "num"), ("", "actions")], rows, empty="No modules installed yet."),
            meta=str(len(installed)),
        )
        + devices_section
        + _section(
            "Catalog",
            f'<div class="tiles">{"".join(tiles)}</div>' if tiles else '<div class="panel"><p class="empty">No module manifests found in <code>catalog/modules/</code>.</p></div>',
            meta=str(len(tiles)),
        )
    )
    return _layout(
        title="Modules",
        active="modules",
        csrf_token=csrf_token,
        body=body,
        pending_approvals=pending_approvals,
        ui_modules=ui_modules,
        nav=nav,
    )
