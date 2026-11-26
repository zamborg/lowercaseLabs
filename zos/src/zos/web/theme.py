"""Shared design system for every server-rendered ZOS page.

One stylesheet string, inlined into each page. Tokens live on ``:root`` and are
redefined for dark mode; components are plain classes. No external fonts, URLs,
images or JS (the admin CSP is ``default-src 'none'``), so icons are inline SVG
and the select caret is drawn with gradients.

Modules (``modules/*``) must not import ``zos``, so each module UI carries a
verbatim copy of ``MODULE_KIT_CSS`` and ``BRAND_MARK`` between generated markers.
After changing either, run ``uv run python -m zos.web.theme`` to rewrite the
copies; ``tests/unit/test_theme_sync.py`` fails if they drift.
"""

from __future__ import annotations

# Inline brand mark: a rounded square with a centre dot. Uses currentColor.
BRAND_MARK = (
    '<svg class="mark" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">'
    '<rect x="1.25" y="1.25" width="13.5" height="13.5" rx="4" fill="none" '
    'stroke="currentColor" stroke-width="1.5"/>'
    '<circle cx="8" cy="8" r="2.25" fill="currentColor"/></svg>'
)

# --- base: tokens, type, controls, tables, badges, notices (shared with modules) ---
BASE_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f7f7f5; --surface: #ffffff; --sunken: #f1f1ee; --hover: rgba(27,27,25,.035);
  --fg: #1b1b19; --fg-2: #3f3f3b; --muted: #6c6c66; --faint: #a2a29b;
  --border: #e6e5e1; --border-strong: #d4d3ce;
  --accent: #1b1b19; --accent-fg: #fafaf8; --focus: #3b6fd8;
  --ok: #26794a; --warn: #a35c00; --err: #c3372c; --info: #3b6fd8;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  --r-sm: 5px; --r: 8px; --r-lg: 12px;
  --ctl: 32px; --ctl-sm: 28px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e0e0d; --surface: #161615; --sunken: #1c1c1b; --hover: rgba(255,255,255,.035);
    --fg: #ecebe7; --fg-2: #cbcac4; --muted: #96958e; --faint: #63625d;
    --border: #262624; --border-strong: #353531;
    --accent: #ecebe7; --accent-fg: #121211; --focus: #7ea4f6;
    --ok: #5cbf83; --warn: #e2a13b; --err: #ef6b5f; --info: #7ea4f6;
  }
}
@media (pointer: coarse) { :root { --ctl: 44px; --ctl-sm: 40px; } }

*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 var(--sans); -webkit-font-smoothing: antialiased; overflow-wrap: anywhere; }
a { color: inherit; text-decoration-color: var(--border-strong); text-underline-offset: 3px; }
a:hover { text-decoration-color: currentColor; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: var(--r-sm); }
h1, h2, h3 { margin: 0; color: var(--fg); }
h1 { font-size: 22px; font-weight: 600; line-height: 1.25; letter-spacing: -0.02em; }
h2 { font-size: 14px; font-weight: 600; line-height: 1.4; letter-spacing: -0.005em; }
p { margin: 0 0 12px; }
strong { font-weight: 600; }
code, .mono { font-family: var(--mono); font-size: 0.9em; letter-spacing: -0.01em; }
code { background: var(--sunken); border-radius: 4px; padding: 0.1em 0.35em; }
td code, th code, .badge code, pre code, h1 code, .plain code, code.faint { background: none; padding: 0; }
.muted { color: var(--muted); }
.faint { color: var(--faint); }
.small { font-size: 12.5px; }
.err-text { color: var(--err); }
.nowrap { white-space: nowrap; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.spacer { flex: 1; }
.skip { position: absolute; left: -999px; top: 0; background: var(--accent); color: var(--accent-fg); padding: 8px 12px; z-index: 100; }
.skip:focus { left: 8px; top: 8px; }
.mark { display: block; flex: none; }
time { white-space: nowrap; font-variant-numeric: tabular-nums; }

/* Brand */
.brand { display: inline-flex; align-items: center; gap: 8px; text-decoration: none; font: 600 15px/1 var(--mono); letter-spacing: -0.03em; color: var(--fg); }

/* Page structure */
.page-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.page-head .lede { margin: 6px 0 0; }
.lede { color: var(--muted); max-width: 68ch; }
.back { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); text-decoration: none; font-size: 13px; margin-bottom: 14px; }
.back:hover { color: var(--fg); }
.section { margin-top: 36px; }
.page-head + .section, .notice + .section { margin-top: 0; }
.section-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; min-height: var(--ctl-sm); }
.section-head .meta { color: var(--faint); font-size: 13px; font-variant-numeric: tabular-nums; }
.section-head .aside { margin-left: auto; align-self: center; }
.section-desc { color: var(--muted); font-size: 13px; margin: -4px 0 12px; max-width: 72ch; }
.panel, .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); }
.card, .panel-body { padding: 16px; }
.card > :last-child, .panel-body > :last-child { margin-bottom: 0; }
.panel-foot { border-top: 1px solid var(--border); padding: 10px 16px; display: flex; gap: 8px; align-items: center; justify-content: space-between; color: var(--muted); font-size: 12.5px; }
.empty { color: var(--muted); text-align: center; padding: 28px 16px; }

/* Tables */
.scroll-x { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th { text-align: left; font-size: 12px; font-weight: 500; color: var(--muted); padding: 9px 16px; border-bottom: 1px solid var(--border); white-space: nowrap; }
td { padding: 11px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }
tbody tr:last-child td { border-bottom: 0; }
th.actions, td.actions { text-align: right; }
td.actions .row { justify-content: flex-end; flex-wrap: nowrap; gap: 4px; }
.sub { display: block; color: var(--muted); font-size: 12.5px; margin-top: 2px; }
.sub.mono { font-size: 12px; }
td.empty { text-align: center; }
@media (min-width: 721px) { td:not(.lead):not(.empty) { white-space: nowrap; } }
.tile code, .foot code { background: none; padding: 0; }

/* Badges: neutral chip, colour carried by the dot */
.badge { --c: var(--faint); display: inline-flex; align-items: center; gap: 6px; height: 22px; padding: 0 8px; border-radius: 999px; background: var(--sunken); color: var(--fg-2); font-size: 12px; font-weight: 500; line-height: 1; white-space: nowrap; vertical-align: middle; }
.badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--c); flex: none; }
.badge.plain::before { display: none; }
.badge-ok { --c: var(--ok); }
.badge-warn { --c: var(--warn); }
.badge-err { --c: var(--err); }
.badge-info { --c: var(--info); }
.badge-muted { --c: var(--faint); color: var(--muted); }
.badge-err, .badge-warn { background: color-mix(in srgb, var(--c) 11%, var(--surface)); }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--faint); flex: none; }
.dot.on { background: var(--ok); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 18%, transparent); }
.chip { display: inline-flex; align-items: center; height: 22px; padding: 0 8px; border-radius: 999px; background: var(--sunken); color: var(--fg-2); font: 500 12px/1 var(--mono); letter-spacing: -0.01em; white-space: nowrap; text-decoration: none; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }

/* Notices */
.notice { --c: var(--info); display: flex; gap: 10px; align-items: baseline; padding: 10px 14px; margin-bottom: 20px; border-radius: var(--r); border: 1px solid color-mix(in srgb, var(--c) 28%, var(--border)); background: color-mix(in srgb, var(--c) 7%, var(--surface)); color: var(--fg); font-size: 13.5px; }
.notice::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--c); flex: none; transform: translateY(-1px); }
.notice > div { min-width: 0; flex: 1; }
.notice-ok { --c: var(--ok); }
.notice-err { --c: var(--err); }
.notice-token, .notice-warn { --c: var(--warn); }
.token-box { margin-top: 8px; padding: 10px 12px; background: var(--surface); border: 1px solid var(--border-strong); border-radius: var(--r-sm); font: 13px/1.4 var(--mono); word-break: break-all; user-select: all; }

/* Controls */
form { margin: 0; }
form.inline { display: inline-flex; gap: 6px; align-items: center; flex-wrap: wrap; }
button, .btn { --b: var(--accent); display: inline-flex; align-items: center; justify-content: center; gap: 6px; height: var(--ctl); padding: 0 12px; border: 1px solid var(--b); border-radius: 6px; background: var(--b); color: var(--accent-fg); font: 500 13px/1 var(--sans); white-space: nowrap; text-decoration: none; cursor: pointer; }
button:hover, .btn:hover { background: color-mix(in srgb, var(--b) 85%, var(--bg)); }
button:disabled { opacity: .45; cursor: not-allowed; }
button.secondary, .btn.secondary { background: var(--surface); border-color: var(--border-strong); color: var(--fg); box-shadow: 0 1px 0 rgba(0,0,0,.03); }
button.secondary:hover, .btn.secondary:hover { background: var(--sunken); }
button.ghost, .btn.ghost { background: transparent; border-color: transparent; color: var(--fg-2); box-shadow: none; }
button.ghost:hover, .btn.ghost:hover { background: var(--hover); color: var(--fg); }
button.danger { background: transparent; border-color: transparent; color: var(--err); box-shadow: none; }
button.danger:hover { background: color-mix(in srgb, var(--err) 9%, transparent); }
button.danger.solid { --b: var(--err); background: var(--err); color: var(--bg); }
button.sm, .btn.sm { height: var(--ctl-sm); padding: 0 10px; font-size: 12.5px; }
button.block, .btn.block { width: 100%; }
input[type=text], input[type=password], input[type=search], select, textarea { width: 100%; min-width: 0; height: var(--ctl); padding: 0 10px; font: 13.5px/1.4 var(--sans); color: var(--fg); background: var(--surface); border: 1px solid var(--border-strong); border-radius: 6px; }
input::placeholder, textarea::placeholder { color: var(--faint); }
input:focus, select:focus, textarea:focus { outline: none; border-color: var(--focus); box-shadow: 0 0 0 3px color-mix(in srgb, var(--focus) 20%, transparent); }
select { appearance: none; -webkit-appearance: none; padding-right: 28px; cursor: pointer; background-image: linear-gradient(45deg, transparent 50%, var(--muted) 50%), linear-gradient(135deg, var(--muted) 50%, transparent 50%); background-position: calc(100% - 15px) 52%, calc(100% - 11px) 52%; background-size: 4px 4px; background-repeat: no-repeat; }
textarea { height: auto; min-height: 10rem; padding: 10px 12px; line-height: 1.55; resize: vertical; }
.field { display: flex; flex-direction: column; gap: 6px; font-size: 12.5px; font-weight: 500; color: var(--fg-2); min-width: 0; }
.field .hint { font-weight: 400; color: var(--faint); }
.form-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
.form-row > .field { flex: 1 1 10rem; }
.form-row > .grow { flex: 3 1 16rem; }
.vstack > * + * { margin-top: 14px; }

/* Disclosure: no-JS expanders and two-step confirms */
details > summary { list-style: none; cursor: pointer; }
details > summary::-webkit-details-marker { display: none; }
details.disclose { display: inline-block; position: relative; }
details.disclose[open] > summary { background: var(--sunken); color: var(--fg); }
.disclose-body { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; margin-top: 8px; }
details.disclose > .disclose-body { position: absolute; right: 0; top: 100%; z-index: 30; margin-top: 6px; padding: 8px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); box-shadow: 0 8px 28px -10px rgba(0,0,0,.18); }
.disclose-body form.inline { flex-wrap: nowrap; }
.disclose-body input[type=text], .disclose-body input[type=password], .section-head .aside input { width: 14rem; height: var(--ctl-sm); }
.section-head .aside form.inline { flex-wrap: nowrap; }
summary.btn { display: inline-flex; user-select: none; }

/* Code */
pre.code, .approval-pre { margin: 0; padding: 14px 16px; background: var(--sunken); border: 1px solid var(--border); border-radius: var(--r); font: 12.5px/1.6 var(--mono); white-space: pre-wrap; word-break: break-word; overflow-x: auto; color: var(--fg); }

/* Key/value list */
dl.kv { display: grid; grid-template-columns: minmax(7rem, max-content) 1fr; gap: 10px 20px; margin: 0; font-size: 13.5px; }
dl.kv dt { color: var(--muted); }
dl.kv dd { margin: 0; min-width: 0; word-break: break-word; }
dl.kv dd.mono, dl.kv dd code { font-size: 12.5px; }

/* Narrow screens: .stack tables become label/value cards */
@media (max-width: 720px) {
  input[type=text], input[type=password], input[type=search], select, textarea { font-size: 16px; }
  table.stack, table.stack tbody, table.stack tr, table.stack td { display: block; width: 100%; }
  table.stack thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
  table.stack tbody tr { padding: 12px 14px; border-bottom: 1px solid var(--border); }
  table.stack tbody tr:last-child { border-bottom: 0; }
  table.stack td { border: 0; padding: 3px 0; display: flex; justify-content: space-between; align-items: baseline; gap: 12px; text-align: right; }
  table.stack td::before { content: attr(data-label); color: var(--muted); font-size: 12.5px; text-align: left; flex: none; }
  table.stack td.lead { display: block; text-align: left; padding-bottom: 6px; }
  table.stack td.lead::before, table.stack td.actions::before, table.stack td[colspan]::before { display: none; }
  table.stack td.actions { padding-top: 8px; justify-content: flex-start; }
  table.stack td.actions .row { justify-content: flex-start; flex-wrap: wrap; }
  table.stack td[colspan] { display: block; text-align: center; }
  details.disclose { position: static; }
  details.disclose > .disclose-body { position: static; box-shadow: none; justify-content: flex-start; width: 100%; }
  .disclose-body form.inline { flex-wrap: wrap; }
  .disclose-body input[type=text], .disclose-body input[type=password], .section-head .aside input { width: auto; flex: 1 1 10rem; }
  .section-head .aside { margin-left: 0; width: 100%; }
  dl.kv { grid-template-columns: 1fr; gap: 2px; }
  dl.kv dd { margin-bottom: 10px; }
}
@media (prefers-reduced-motion: no-preference) {
  button, .btn, .nav a, input, select, textarea { transition: background-color .12s, color .12s, border-color .12s, box-shadow .12s; }
}

/* Standalone app bar (auth pages and module UIs) */
.appbar { position: sticky; top: 0; z-index: 20; background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: saturate(1.4) blur(10px); -webkit-backdrop-filter: saturate(1.4) blur(10px); border-bottom: 1px solid var(--border); }
.appbar-in { max-width: 880px; margin: 0 auto; padding: 0 20px; height: 52px; display: flex; align-items: center; gap: 10px; }
.crumb-sep { color: var(--faint); font-weight: 300; }
.crumb { font-weight: 600; text-decoration: none; letter-spacing: -0.01em; }
.page { max-width: 880px; margin: 0 auto; padding: 32px 20px 72px; }
.page.narrow { max-width: 440px; padding-top: 12vh; }
@media (max-width: 720px) { .appbar-in { padding: 0 16px; } .page { padding: 24px 16px 56px; } .page.narrow { padding-top: 32px; } }

/* Auth cards (sign-in, OAuth authorize, connect) */
.auth-brand { display: flex; justify-content: center; margin-bottom: 24px; }
.auth-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 28px; box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 24px -12px rgba(0,0,0,.08); }
.auth-card h1 { font-size: 19px; }
.auth-card .lede { margin: 6px 0 22px; }
.auth-card dl.kv { margin: 0 0 22px; padding: 14px 16px; background: var(--sunken); border-radius: var(--r); grid-template-columns: 1fr; gap: 2px; }
.auth-card dl.kv dt { font-size: 12px; }
.auth-card dl.kv dd { margin-bottom: 8px; font-family: var(--mono); font-size: 12.5px; }
.auth-card dl.kv dd:last-child { margin-bottom: 0; }
dd.host-callout { font: 600 15px/1.4 var(--mono) !important; color: var(--warn); }
.auth-form > * + * { margin-top: 16px; }
.auth-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.auth-foot { text-align: center; color: var(--faint); font-size: 12.5px; margin-top: 18px; }
@media (max-width: 720px) { .auth-card { padding: 22px 18px; } }
"""

# --- admin shell: sidebar navigation, stats strip, policy matrix, approvals ---
ADMIN_CSS = """
.shell { min-height: 100vh; }
.side { position: sticky; top: 0; z-index: 20; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; padding: 10px 16px 0; background: color-mix(in srgb, var(--bg) 90%, transparent); backdrop-filter: saturate(1.4) blur(10px); -webkit-backdrop-filter: saturate(1.4) blur(10px); border-bottom: 1px solid var(--border); }
.side .brand { padding: 4px 0; }
.nav { grid-column: 1 / -1; grid-row: 2; min-width: 0; display: flex; gap: 2px; overflow-x: auto; scrollbar-width: none; margin: 6px -16px 0; padding: 0 12px 8px; }
.nav::-webkit-scrollbar { display: none; }
.nav-label { display: none; }
.nav a { display: flex; align-items: center; gap: 8px; height: 32px; padding: 0 10px; border-radius: 6px; color: var(--muted); text-decoration: none; font-weight: 500; white-space: nowrap; }
.nav a:hover { color: var(--fg); background: var(--hover); }
.nav a.active { color: var(--fg); background: var(--surface); box-shadow: inset 0 0 0 1px var(--border), 0 1px 0 rgba(0,0,0,.03); }
.nav .count { margin-left: auto; min-width: 18px; height: 18px; padding: 0 5px; border-radius: 999px; display: inline-flex; align-items: center; justify-content: center; background: color-mix(in srgb, var(--warn) 16%, transparent); color: var(--warn); font-size: 11px; font-weight: 600; font-variant-numeric: tabular-nums; }
.nav .dot { margin-left: auto; width: 6px; height: 6px; box-shadow: none; }
.side-foot { grid-column: 2; grid-row: 1; }
.side-foot form { display: block; }
.side-foot button { color: var(--muted); }
.content { width: 100%; max-width: 1080px; margin: 0 auto; padding: 24px 16px 64px; min-width: 0; }
@media (pointer: coarse) { .nav a { height: 40px; } }
.content.wide { max-width: 1440px; }
@media (min-width: 960px) {
  .shell { display: grid; grid-template-columns: 232px minmax(0, 1fr); }
  .side { height: 100vh; display: flex; flex-direction: column; align-items: stretch; padding: 18px 12px 14px; border-bottom: 0; border-right: 1px solid var(--border); background: var(--bg); backdrop-filter: none; -webkit-backdrop-filter: none; }
  .side .brand { padding: 0 10px; margin-bottom: 22px; }
  .nav { flex-direction: column; overflow: visible; margin: 0; padding: 0; gap: 1px; }
  .nav-label { display: block; padding: 18px 10px 6px; color: var(--faint); font-size: 12px; font-weight: 500; }
  .side-foot { margin-top: auto; padding-top: 12px; }
  .side-foot button { width: 100%; justify-content: flex-start; padding: 0 10px; }
  .content { padding: 40px 48px 80px; }
}

/* Overview stats strip */
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); }
.stat { padding: 16px 18px; border-left: 1px solid var(--border); min-width: 0; }
.stat:first-child { border-left: 0; }
.stat .label { color: var(--muted); font-size: 12.5px; }
.stat .value { margin-top: 6px; font-size: 24px; font-weight: 600; letter-spacing: -0.025em; line-height: 1.15; font-variant-numeric: tabular-nums; }
.stat .value .of { color: var(--faint); font-weight: 500; }
.stat .value.warn { color: var(--warn); }
.stat .sub { font-size: 12.5px; }
.stat a { text-decoration: none; }
@media (max-width: 720px) {
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .stat:nth-child(odd) { border-left: 0; }
  .stat:nth-child(n+3) { border-top: 1px solid var(--border); }
  .stat .value { font-size: 20px; }
}

/* Tiles (apps, catalogs) */
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); gap: 10px; }
.tile { display: flex; flex-direction: column; gap: 6px; padding: 14px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); min-width: 0; }
a.tile { text-decoration: none; }
a.tile:hover { border-color: var(--border-strong); }
.tile-head { display: flex; align-items: center; gap: 8px; }
.tile-head .name { font-weight: 600; letter-spacing: -0.01em; }
.tile-head .aside { margin-left: auto; }
.tile .desc { color: var(--muted); font-size: 13px; line-height: 1.45; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.tile .foot { margin-top: auto; padding-top: 8px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.tile .foot .aside { margin-left: auto; }
.tile form.install { display: flex; flex-direction: column; gap: 10px; width: 100%; }

/* Segmented decision control (policy matrix) */
.seg { display: inline-flex; padding: 2px; gap: 2px; border-radius: 7px; background: var(--sunken); vertical-align: middle; }
.seg button { --c: var(--fg); height: 24px; padding: 0 8px; border: 0; border-radius: 5px; background: transparent; color: var(--muted); font-size: 12px; font-weight: 500; box-shadow: none; }
.seg button:hover { background: var(--hover); color: var(--fg); }
.seg button.on { color: var(--c); background: var(--surface); box-shadow: 0 0 0 1px var(--border); }
.seg button.pin { color: var(--c); background: color-mix(in srgb, var(--c) 14%, var(--surface)); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--c) 45%, transparent); font-weight: 600; }
.seg button.reset, .seg .reset-ph { color: var(--faint); padding: 0 6px; }
.seg .reset-ph { visibility: hidden; display: inline-flex; align-items: center; font-size: 12px; }
.legend .reset-ph { display: none; }
.seg button.reset:hover { color: var(--err); }
.seg .ok { --c: var(--ok); } .seg .warn { --c: var(--warn); } .seg .err { --c: var(--err); } .seg .info { --c: var(--info); }
@media (pointer: coarse) { .seg button { height: 34px; padding: 0 10px; } }
table.matrix th, table.matrix td { padding-left: 12px; padding-right: 12px; }
table.matrix th:first-child, table.matrix td:first-child { padding-left: 16px; }
table.matrix td { white-space: nowrap; }
table.matrix th.client { line-height: 1.3; }
table.matrix th.client .sub { font-weight: 400; color: var(--faint); font-size: 11.5px; }
table.matrix tr.offline td:first-child code { color: var(--muted); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 12.5px; }
.legend .seg { pointer-events: none; }

/* Approvals */
.approval-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }
.approval-actions button { width: 100%; height: 44px; font-size: 14px; }
button.approve-btn { --b: var(--ok); background: var(--ok); border-color: var(--ok); color: var(--bg); font-weight: 600; }
button.deny-btn { background: var(--surface); border-color: var(--border-strong); color: var(--err); }
button.deny-btn:hover { background: color-mix(in srgb, var(--err) 8%, var(--surface)); }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: 16px; }
.meta-item .label { color: var(--muted); font-size: 12.5px; }
.meta-item .val { margin-top: 4px; }
"""

# --- module kit: filters, lists, tabs, messages (module UIs only) ---
MODULE_CSS = """
.toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
.toolbar input[type=search], .toolbar input[type=text] { flex: 1 1 16rem; width: auto; }
.pills { display: flex; gap: 6px; overflow-x: auto; scrollbar-width: none; margin: 0 0 16px; padding-bottom: 2px; }
.pills::-webkit-scrollbar { display: none; }
.pill { display: inline-flex; align-items: center; gap: 6px; height: 28px; padding: 0 11px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface); color: var(--fg-2); font-size: 12.5px; font-weight: 500; text-decoration: none; white-space: nowrap; }
.pill:hover { border-color: var(--border-strong); color: var(--fg); }
.pill.active { background: var(--fg); border-color: var(--fg); color: var(--bg); }
.pill .n { color: var(--faint); font-variant-numeric: tabular-nums; }
.pill.active .n { color: inherit; opacity: .6; }
.pill .count { min-width: 18px; height: 18px; padding: 0 5px; border-radius: 999px; display: inline-flex; align-items: center; justify-content: center; background: var(--info); color: var(--bg); font-size: 11px; font-weight: 600; }
.tabs { display: inline-flex; gap: 2px; padding: 2px; border-radius: 7px; background: var(--sunken); margin-bottom: 14px; }
.tabs a { height: 26px; padding: 0 10px; display: inline-flex; align-items: center; border-radius: 5px; color: var(--muted); text-decoration: none; font-size: 12.5px; font-weight: 500; }
.tabs a:hover { color: var(--fg); }
.tabs a.active { background: var(--surface); color: var(--fg); box-shadow: 0 0 0 1px var(--border); }
@media (pointer: coarse) { .pill { height: 36px; } .tabs a { height: 36px; padding: 0 14px; } }
.list { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; }
.item { display: block; padding: 13px 16px; border-bottom: 1px solid var(--border); text-decoration: none; color: inherit; }
.item:last-child { border-bottom: 0; }
a.item:hover { background: var(--hover); }
.item .top { display: flex; align-items: baseline; gap: 12px; min-width: 0; }
.item .title { font-weight: 500; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.item .when { margin-left: auto; color: var(--faint); font-size: 12.5px; flex: none; font-variant-numeric: tabular-nums; }
.item .meta { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 4px; color: var(--muted); font-size: 12.5px; }
.item .snippet { margin-top: 3px; color: var(--muted); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.item.unread .title { font-weight: 600; }
.item.unread .title::before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--info); margin-right: 8px; vertical-align: 1px; }
.prose { white-space: pre-wrap; word-break: break-word; font-size: 14px; line-height: 1.65; }
.msg { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); margin-bottom: 10px; }
.msg-head { display: flex; gap: 12px; align-items: center; padding: 11px 16px; border-bottom: 1px solid var(--border); font-size: 13px; }
.msg-head .who { min-width: 0; }
.msg-head .aside { margin-left: auto; display: flex; gap: 8px; align-items: center; }
.msg-body { padding: 14px 16px; }
"""

CSS = BASE_CSS + ADMIN_CSS
MODULE_KIT_CSS = BASE_CSS + MODULE_CSS

_BEGIN = "# --- BEGIN zos theme (generated by `python -m zos.web.theme`; do not edit) ---"
_END = "# --- END zos theme ---"


def module_theme_block() -> str:
    """The exact Python block each module UI embeds in place of importing this file."""
    return f'{_BEGIN}\nBRAND_MARK = {BRAND_MARK!r}\n_THEME_CSS = """{MODULE_KIT_CSS}"""\n{_END}'


def module_ui_files() -> list[str]:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "modules"
    return sorted(str(p) for p in root.glob("*/src/*/ui.py"))


def sync_modules() -> list[str]:
    """Rewrite the generated theme block in every module UI; return files changed."""
    import re
    from pathlib import Path

    changed = []
    pattern = re.compile(re.escape(_BEGIN) + r".*?" + re.escape(_END), re.S)
    for path in module_ui_files():
        p = Path(path)
        src = p.read_text()
        new = pattern.sub(lambda _m: module_theme_block(), src)
        if new != src:
            p.write_text(new)
            changed.append(path)
    return changed


if __name__ == "__main__":
    for f in sync_modules():
        print(f"synced {f}")
