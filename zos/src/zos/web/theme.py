"""Shared design system (single inline stylesheet) for every server-rendered ZOS page.

One stylesheet string, inlined into each page: CSS custom properties for color,
spacing, radius and type scale; light and dark via ``prefers-color-scheme``;
mobile-first layout with 44px tap targets. No external fonts, URLs, or JS.

The inbox module (``modules/inbox``) keeps an inline copy of the token and
component block below because modules must not import ``zos``. Keep them in sync.
"""

from __future__ import annotations

CSS = """
:root {
  color-scheme: light dark;
  --bg: #f5f5f6; --surface: #ffffff; --surface-2: #fafafa; --sunken: #f0f0f2;
  --fg: #18181b; --muted: #5a5a66; --border: #e4e4e7; --border-strong: #cfcfd6;
  --accent: #18181b; --accent-fg: #ffffff; --focus: #2563eb;
  --ok: #166534; --ok-bg: #dcfce7; --ok-bd: #86efac;
  --warn: #92400e; --warn-bg: #fef3c7; --warn-bd: #fcd34d;
  --err: #991b1b; --err-bg: #fee2e2; --err-bd: #fca5a5;
  --info: #1e40af; --info-bg: #dbeafe;
  --code-bg: #18181b; --code-fg: #f4f4f5;
  --s1: 0.25rem; --s2: 0.5rem; --s3: 0.75rem; --s4: 1rem; --s5: 1.5rem; --s6: 2.25rem;
  --r-sm: 6px; --r: 10px; --r-lg: 14px;
  --fs-xs: 0.75rem; --fs-sm: 0.8125rem; --fs: 0.9375rem; --fs-lg: 1.0625rem; --fs-xl: 1.375rem;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --shadow: 0 1px 2px rgba(0,0,0,.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e0e10; --surface: #17171a; --surface-2: #1c1c20; --sunken: #121215;
    --fg: #f4f4f5; --muted: #a1a1ab; --border: #2a2a30; --border-strong: #3d3d46;
    --accent: #f4f4f5; --accent-fg: #18181b; --focus: #60a5fa;
    --ok: #86efac; --ok-bg: #052e16; --ok-bd: #166534;
    --warn: #fcd34d; --warn-bg: #3a2105; --warn-bd: #92400e;
    --err: #fca5a5; --err-bg: #3f0d0d; --err-bd: #991b1b;
    --info: #93c5fd; --info-bg: #172554;
    --code-bg: #0a0a0c; --code-fg: #f4f4f5;
    --shadow: none;
  }
}
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; font-family: var(--sans); font-size: var(--fs); line-height: 1.5; background: var(--bg); color: var(--fg); overflow-wrap: anywhere; }
a { color: inherit; text-underline-offset: 2px; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: var(--r-sm); }
h1 { font-size: var(--fs-xl); line-height: 1.25; margin: 0 0 var(--s4); letter-spacing: -0.01em; }
h2 { font-size: var(--fs-lg); line-height: 1.3; margin: 0 0 var(--s3); display: flex; align-items: center; gap: var(--s2); flex-wrap: wrap; }
p { margin: 0 0 var(--s3); }
code, .mono { font-family: var(--mono); font-size: 0.85em; }
code { background: var(--sunken); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 0.05em 0.35em; }
td code, th code, .badge code, pre code, .token-box code { background: none; border: 0; padding: 0; }
.skip { position: absolute; left: -999px; top: 0; background: var(--accent); color: var(--accent-fg); padding: var(--s2) var(--s3); z-index: 100; }
.skip:focus { left: var(--s2); top: var(--s2); }

/* Shell: header + scrollable nav */
.topbar { background: var(--surface); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 20; }
.topbar-in { max-width: 1180px; margin: 0 auto; padding: var(--s2) var(--s4) 0; display: flex; align-items: center; justify-content: space-between; gap: var(--s3); flex-wrap: wrap; }
.brand { font-weight: 700; letter-spacing: 0.01em; text-decoration: none; padding: var(--s2) 0; display: inline-flex; align-items: center; gap: var(--s2); }
.brand::before { content: ""; width: 0.65rem; height: 0.65rem; border-radius: 3px; background: var(--accent); }
nav.main-nav { display: flex; gap: var(--s1); overflow-x: auto; scrollbar-width: none; -webkit-overflow-scrolling: touch; max-width: 1180px; margin: 0 auto; padding: 0 var(--s3); }
nav.main-nav::-webkit-scrollbar { display: none; }
nav.main-nav a { text-decoration: none; white-space: nowrap; color: var(--muted); padding: 0 var(--s3); min-height: 44px; display: inline-flex; align-items: center; gap: var(--s2); border-bottom: 2px solid transparent; font-size: var(--fs); font-weight: 500; }
nav.main-nav a:hover { color: var(--fg); }
nav.main-nav a.active { color: var(--fg); border-bottom-color: var(--accent); font-weight: 600; }
main { max-width: 1180px; margin: 0 auto; padding: var(--s4) var(--s4) var(--s6); }
main.narrow { max-width: 30rem; padding-top: var(--s5); }
.page-head { display: flex; align-items: center; justify-content: space-between; gap: var(--s3); flex-wrap: wrap; margin-bottom: var(--s4); }
.page-head h1 { margin: 0; }
.back { margin: 0 0 var(--s3); }
.pager { margin: var(--s4) 0 0; }

/* Surfaces */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: var(--s4); margin-bottom: var(--s4); box-shadow: var(--shadow); }
.card > :last-child { margin-bottom: 0; }
.stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s3); margin-bottom: var(--s4); }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: var(--s3) var(--s4); box-shadow: var(--shadow); min-width: 0; }
.stat .label, .meta-item .label { font-size: var(--fs-xs); color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
.stat .value { font-size: 1.5rem; font-weight: 700; margin-top: var(--s1); line-height: 1.2; }
.stat .value .sub { display: block; font-size: var(--fs-sm); font-weight: 400; color: var(--muted); margin-top: var(--s1); }
.muted { color: var(--muted); font-size: var(--fs-sm); }
.muted.block { display: block; margin-top: var(--s1); }
.err-text { color: var(--err); }
.empty { text-align: center; color: var(--muted); padding: var(--s5) var(--s3); font-size: var(--fs); }
.row { display: flex; gap: var(--s2); flex-wrap: wrap; align-items: center; }
.row.gap-lg { gap: var(--s3); }
.stack-sm > * + * { margin-top: var(--s2); }

/* Tables: scroll inside container; .stack tables become cards on narrow screens */
.scroll-x { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 0 calc(-1 * var(--s1)); padding: 0 var(--s1); }
table { width: 100%; border-collapse: collapse; font-size: var(--fs-sm); }
th, td { text-align: left; padding: var(--s2) var(--s3); border-bottom: 1px solid var(--border); vertical-align: middle; }
th { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.nowrap { white-space: nowrap; }
.digest { font-family: var(--mono); font-size: var(--fs-xs); color: var(--muted); }

/* Status pills */
.badge { display: inline-flex; align-items: center; gap: 0.4em; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: var(--fs-xs); font-weight: 600; line-height: 1.4; white-space: nowrap; border: 1px solid transparent; }
.badge::before { content: ""; width: 0.45em; height: 0.45em; border-radius: 50%; background: currentColor; opacity: 0.85; }
.badge-ok { background: var(--ok-bg); color: var(--ok); border-color: var(--ok-bd); }
.badge-warn { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-bd); }
.badge-err { background: var(--err-bg); color: var(--err); border-color: var(--err-bd); }
.badge-muted { background: var(--sunken); color: var(--muted); border-color: var(--border); }
.badge-info { background: var(--info-bg); color: var(--info); }
.badge.plain::before { display: none; }

/* Notices */
.notice { padding: var(--s3) var(--s4); border-radius: var(--r); margin-bottom: var(--s4); border: 1px solid; font-size: var(--fs); }
.notice-ok { background: var(--ok-bg); border-color: var(--ok-bd); color: var(--ok); }
.notice-err { background: var(--err-bg); border-color: var(--err-bd); color: var(--err); }
.notice-token { background: var(--warn-bg); border-color: var(--warn-bd); color: var(--warn); }
.token-box { margin-top: var(--s2); padding: var(--s3); background: var(--surface); color: var(--fg); border: 1px dashed var(--warn-bd); border-radius: var(--r-sm); font-family: var(--mono); word-break: break-all; font-size: var(--fs-sm); user-select: all; }

/* Forms and buttons */
form { margin: 0; }
form.inline { display: inline-flex; gap: var(--s2); align-items: center; flex-wrap: wrap; }
.form-row { display: flex; gap: var(--s3); flex-wrap: wrap; align-items: flex-end; }
.form-row > .grow { flex: 1 1 14rem; }
label.field { display: flex; flex-direction: column; gap: var(--s1); font-size: var(--fs-sm); font-weight: 600; min-width: 0; }
label.field.block { margin-bottom: var(--s4); }
input[type=text], input[type=password], select, textarea { font: inherit; font-size: var(--fs); color: var(--fg); background: var(--surface); padding: 0 var(--s3); min-height: 44px; border: 1px solid var(--border-strong); border-radius: var(--r-sm); max-width: 100%; }
input[type=text], input[type=password], textarea { width: 100%; }
td input[type=text], td input[type=password], .row input[type=text], .row input[type=password], form.inline input[type=text], form.inline input[type=password] { width: 11rem; }
input::placeholder, textarea::placeholder { color: var(--muted); opacity: 0.8; }
textarea { padding: var(--s3); min-height: 9rem; resize: vertical; line-height: 1.5; }
select { padding-right: var(--s2); }
button, a.btn { display: inline-flex; align-items: center; justify-content: center; gap: var(--s2); text-decoration: none; font: inherit; font-size: var(--fs); font-weight: 600; min-height: 44px; padding: 0 var(--s4); border-radius: var(--r-sm); border: 1px solid var(--accent); background: var(--accent); color: var(--accent-fg); cursor: pointer; white-space: nowrap; }
button:hover, a.btn:hover { opacity: 0.88; }
button.secondary, a.btn.secondary { background: var(--surface); color: var(--fg); border-color: var(--border-strong); }
button.secondary:hover, a.btn.secondary:hover { background: var(--sunken); opacity: 1; }
button.danger { background: var(--surface); color: var(--err); border-color: var(--err-bd); }
button.danger:hover { background: var(--err-bg); opacity: 1; }
button.sm, a.btn.sm { min-height: 44px; padding: 0 var(--s3); font-size: var(--fs-sm); }
button.block, .block-btn { width: 100%; }
@media (min-width: 721px) and (pointer: fine) {
  button.sm, a.btn.sm { min-height: 34px; padding: 0 var(--s3); }
  input[type=text], input[type=password], select { min-height: 38px; }
  button, a.btn { min-height: 38px; }
  nav.main-nav a { min-height: 44px; }
}
.cell-controls { display: flex; flex-direction: column; gap: var(--s2); align-items: flex-start; }

/* Approval detail */
.approval-pre, pre.code { background: var(--code-bg); color: var(--code-fg); padding: var(--s4); border-radius: var(--r-sm); font-family: var(--mono); font-size: var(--fs-sm); line-height: 1.55; white-space: pre-wrap; word-break: break-word; overflow-x: auto; margin: var(--s2) 0 0; border: 1px solid var(--border); }
.approval-actions { display: grid; grid-template-columns: 1fr 1fr; gap: var(--s3); margin-top: var(--s4); }
.approval-actions button { width: 100%; min-height: 52px; font-size: var(--fs-lg); border-radius: var(--r); }
button.approve-btn { background: var(--ok); border-color: var(--ok); color: var(--surface); }
button.deny-btn { background: var(--err); border-color: var(--err); color: var(--surface); }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr)); gap: var(--s4) var(--s3); }
.meta-item .val { margin-top: var(--s1); word-break: break-word; }

/* Key/value list (authorize + connect pages) */
dl.kv { margin: 0 0 var(--s4); }
dl.kv dt { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; margin-top: var(--s3); }
dl.kv dd { margin: var(--s1) 0 0; font-family: var(--mono); font-size: var(--fs-sm); word-break: break-all; }
dd.host-callout { background: var(--warn-bg); border: 1px solid var(--warn-bd); color: var(--warn); border-radius: var(--r-sm); padding: var(--s2) var(--s3); font-size: var(--fs); font-weight: 700; }
.meta-box { background: var(--sunken); border: 1px solid var(--border); border-radius: var(--r-sm); padding: var(--s3); font-family: var(--mono); font-size: var(--fs-sm); word-break: break-all; margin: var(--s3) 0 var(--s4); }
.auth-actions { display: grid; grid-template-columns: 1fr 1fr; gap: var(--s3); }
.auth-form > * + * { margin-top: var(--s4); }

/* Narrow screens: stacked table rows */
@media (max-width: 720px) {
  main { padding: var(--s3) var(--s3) var(--s5); }
  .topbar-in { padding: 0 var(--s3); }
  nav.main-nav { padding: 0 var(--s2); }
  .card { padding: var(--s3) var(--s3); }
  input[type=text], input[type=password], select, textarea { font-size: 16px; }
  table.stack, table.stack tbody, table.stack tr, table.stack td { display: block; width: 100%; }
  table.stack thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
  table.stack tr { border: 1px solid var(--border); border-radius: var(--r); padding: var(--s2) var(--s3); margin-bottom: var(--s3); background: var(--surface-2); }
  table.stack td { border: 0; padding: var(--s1) 0; display: flex; gap: var(--s3); justify-content: space-between; align-items: center; text-align: right; }
  table.stack td::before { content: attr(data-label); color: var(--muted); font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; text-align: left; flex: none; }
  table.stack td.cell-wide { display: block; text-align: left; }
  table.stack td.cell-wide::before { display: block; margin-bottom: var(--s1); }
  table.stack td[colspan] { display: block; text-align: center; }
  table.stack td[colspan]::before { display: none; }
  form.inline { width: 100%; }
  form.inline input[type=text], form.inline input[type=password], td input[type=text], td input[type=password] { flex: 1 1 8rem; width: auto; min-width: 0; }
  .form-row > label.field, .form-row > button, .form-row > a.btn { flex: 1 1 100%; }
  .stat .value { font-size: 1.3rem; }
}
@media (min-width: 721px) {
  .stats { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  main { padding: var(--s5) var(--s5) var(--s6); }
  .topbar-in { padding: var(--s2) var(--s5) 0; }
}
@media (prefers-reduced-motion: no-preference) {
  button, a.btn, nav.main-nav a { transition: background-color .12s, opacity .12s, color .12s; }
}
"""
