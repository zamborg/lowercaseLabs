"""Offline HTML experiment report; raw replay links are analyst-only artifacts."""
from __future__ import annotations
import html
import json
from collections import defaultdict
from urllib.parse import quote
from pathlib import Path
from .ratings import analyze


def write_report(experiment_dir, records: list[dict], manifest: dict, bootstrap: int = 2000) -> Path:
    root = Path(experiment_dir)
    root.mkdir(parents=True, exist_ok=True)
    analysis = analyze(records, bootstrap=bootstrap)
    (root/'ratings.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    esc = lambda v: html.escape(str(v), quote=True)
    def number(v):
        return 'unavailable' if v is None else f'{v:.2f}'
    def interval(v):
        return 'unavailable' if v is None else f'[{v[0]:.1f}, {v[1]:.1f}]'
    out = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Social Arena experiment</title>',
           '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#18202a}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}.notice{background:#fff1c2;padding:16px}</style>',
           '<h1>Social Arena experiment</h1>', '<p class="notice">Exploratory estimates. Ratings describe whole policies or declared team compositions within one frozen game pool. Separate pools are not comparable.</p>',
           f'<p>Games: {len(records)} · Complete: {analysis["counts"].get("complete",0)} · Eligible: {analysis["eligible_games"]} · Cost USD: {number(analysis["cost_usd"])} · Completion rate: {number(analysis["completion_rate"])} · Invalid action rate: {number(analysis["invalid_action_rate"])} · Cost per finished game: {number(analysis["cost_per_finished_game"])}</p>']
    # Whitelist policy identity fields; environment credentials and work paths never render.
    metadata = {k:manifest[k] for k in ('experiment_id','game','engine','protocol','protocol_version','engine_version','policy_ids','seed','schema','budget','max_decisions','probes') if k in manifest}
    out.append('<h2>Experiment definition</h2><pre>'+esc(json.dumps(metadata,indent=2))+'</pre>')
    out.append('<h3>Frozen policy identities</h3><table><tr><th>Name</th><th>Immutable ID</th><th>Definition</th></tr>')
    for name, policy in manifest.get('policies',{}).items():
        safe = {k:policy[k] for k in ('kind','provider','model','prompt_hash','reasoning','memory','context','max_context_chars','max_output_tokens','service_tier','transport_retries','input_usd_per_million','output_usd_per_million','price_source') if k in policy}
        out.append('<tr><td>'+esc(name)+'</td><td>'+esc(policy.get('id','unavailable'))+'</td><td><pre>'+esc(json.dumps(safe,indent=2))+'</pre></td></tr>')
    out.append('</table>')
    ledger_path = root/'ledger.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {}
    ledger_rows = list(ledger.values())
    known = sum(v.get('actual_usd',0) or 0 for v in ledger_rows)
    uncertain = sum(v.get('reserved_usd',0) for v in ledger_rows if 'actual_usd' not in v)
    failed = [v for v in ledger_rows if v.get('error')]
    out.append('<h2>Cost ledger</h2><p>Known charges USD: '+number(known)+' · Uncertain/pending reserved USD: '+number(uncertain)+' · Failed calls: '+str(len(failed))+' · Calls: '+str(len(ledger_rows))+'</p>')
    out.append('<table><tr><th>Request</th><th>Game / policy / kind</th><th>Status / failure</th><th>Reserved USD</th><th>Known actual USD</th></tr>')
    for request, entry in sorted(ledger.items()):
        out.append('<tr><td>'+esc(request)+'</td><td>'+esc(' / '.join(str(entry.get(k,'')) for k in ('game_id','policy','kind')))+'</td><td>'+esc(str(entry.get('status',''))+' / '+str(entry.get('error','')))+'</td><td>'+number(entry.get('reserved_usd'))+'</td><td>'+number(entry.get('actual_usd'))+'</td></tr>')
    out.append('</table>')
    out.append('<h2>Diagnostics</h2><table><tr><th>Policy</th><th>Games</th><th>Complete</th><th>Policy failure</th><th>Incomplete</th><th>Excluded</th></tr>')
    for policy, row in sorted(analysis['policy_reliability'].items()):
        out.append('<tr><td>'+esc(policy)+'</td>'+''.join('<td>'+str(row[k])+'</td>' for k in ('games','complete','policy_failure','incomplete','excluded'))+'</tr>')
    out.append('</table>')
    for game,pool in analysis['pools'].items():
        out += [f'<h2>{esc(game)}</h2>',f'<p>Independent scenarios: {pool["independent_scenarios"]} · Eligible games: {pool["eligible_games"]} · Good / first-player advantage: {number(pool["side_advantage_elo"])} Elo {interval(pool["side_advantage_ci95"])}</p>']
        out.extend('<p class="notice">'+esc(w)+'</p>' for w in pool['warnings'])
        out.append('<table><tr><th>Policy</th><th>Rating [95% interval]</th><th>Good / first W/L/D</th><th>Evil / second W/L/D</th></tr>')
        for policy, row in sorted(pool['ratings'].items(),key=lambda kv:-kv[1]['rating']):
            raw=row['raw']
            def wld(side):
                return '/'.join(str(raw[side][k]) for k in ('wins','losses','draws'))
            out.append(f'<tr><td>{esc(policy)}</td><td>{number(row["rating"])} {interval(row["rating_ci95"])}</td><td>{wld("first_or_good")}</td><td>{wld("second_or_evil")}</td></tr>')
        out.append('</table><h3>Matchups</h3><table><tr><th>A vs B</th><th>Games</th><th>Elo gap [95%]</th><th>Side-averaged expected score [95%]</th></tr>')
        for m in pool['matchups']:
            out.append(f'<tr><td>{esc(m["a"])} vs {esc(m["b"])}</td><td>{m["games"]}</td><td>{number(m["rating_gap"])} {interval(m["gap_ci95"])}</td><td>{number(m["side_averaged_expected_score"])} {interval(m["expected_score_ci95"])}</td></tr>')
        out += ['</table><details><summary>Strength prior sensitivity (SD 1 and 3 logits)</summary><pre>',esc(json.dumps(pool['prior_sensitivity'],indent=2)),'</pre></details>']
    out += ['<h2>Exclusions and reliability</h2><p>Primary ratings include complete swapped pairs and explicitly marked self-play only. Infrastructure interruptions remain missing outcomes.</p><pre>',esc(json.dumps({'excluded_games': analysis['excluded_games'], 'policy_reliability': analysis['policy_reliability'], 'failure_as_loss_sensitivity': {g:p['failure_as_loss_sensitivity'] for g,p in analysis['pools'].items()}},indent=2)),'</pre><h2>Game records and replays</h2><p class="notice">Analyst-only post-game evidence. Omniscient event logs may contain all private roles, cards, hands, and submitted actions. Do not provide these logs to gameplay policies. Recipient visibility is recorded on individual events.</p>']
    all_probes = []
    for r in records:
        gid = str(r.get('game_id',''))
        # Never let identifiers traverse outside the report root.
        if Path(gid).name != gid or gid in ('','.','..'):
            continue
        out.append(f'<details><summary>{esc(gid)} — {esc(r.get("status"))}</summary>')
        directory = root/'games'/gid
        if (directory/'events.jsonl').is_file():
            _, probes = _write_replay(directory, len(r.get('policies',[])) or (5 if r.get('game','avalon')=='avalon' else 2))
            all_probes.extend(probes)
            out.append('<p><a href="games/'+quote(gid,safe='')+'/replay.html">Open decision replay — public / seat / omniscient</a></p>')
        for filename in ('summary.json','events.jsonl'):
            path=root/'games'/gid/filename
            if path.is_file():
                href= 'games/'+quote(gid,safe='')+'/'+filename
                out.append(f'<p><a href="{esc(href)}">{esc(filename)} — analyst-only post-game</a></p>')
        out.append('</details>')
    grouped = defaultdict(list)
    for probe in all_probes:
        if isinstance(probe.get('brier'), (int,float)):
            grouped[(probe.get('role','unknown'), probe.get('phase','unknown'))].append(probe['brier'])
    out.append('<h2>Detached belief probes</h2><p>Mean Brier score across eligible unknown targets per probe; lower is better. Mechanically known labels are excluded by the runner. Probes use detached seat-visible context and do not write to gameplay history.</p><table><tr><th>Role</th><th>Phase</th><th>Probes</th><th>Mean Brier</th></tr>')
    for (role,phase), values in sorted(grouped.items()):
        out.append('<tr><td>'+esc(role)+'</td><td>'+esc(phase)+'</td><td>'+str(len(values))+'</td><td>'+number(sum(values)/len(values))+'</td></tr>')
    if not grouped:
        out.append('<tr><td colspan="4">No scored probes recorded.</td></tr>')
    out.append('</table>')
    out.append('<p>Method: Gaussian strength SD 1.5 logits, side SD 3; sum-zero strengths; 1500 reference mean; 95% whole-scenario bootstrap intervals. Dominion uses Davidson ties and expected score (win + half draw). Bootstrap resamples: '+str(bootstrap)+'.</p></html>')
    target=root/'report.html'
    target.write_text('\n'.join(out),encoding='utf-8')
    return target


def _write_replay(directory: Path, seats: int) -> tuple[Path, list[dict]]:
    from .runner import export_view
    public = export_view(directory)
    omni = export_view(directory, omniscient=True)
    views = {'public': public}
    views.update({f'seat-{seat}': export_view(directory, seat=seat) for seat in range(seats)})
    views['omniscient'] = omni
    # A script data element must not contain literal HTML terminators or separators.
    payload = json.dumps(views, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    options = '<option value="public">PUBLIC — shared events only</option>' + ''.join(f'<option value="seat-{s}">SEAT {s} — permitted private evidence</option>' for s in range(seats)) + '<option value="omniscient">OMNISCIENT — analyst-only post-game</option>'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Post-game replay</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:16px}.warning{background:#fff1c2;padding:16px}select,button{padding:8px;margin:4px}</style>
<h1>Post-game replay</h1><p><a href="../../report.html">Experiment report</a></p>
<p class="warning">Analyst-only post-game review. Seat and omniscient views include private evidence and must never be supplied to gameplay policies. PUBLIC shows only public events; SEAT includes that seat's attempts, observations, action specifications, provider results, and detached probes. OMNISCIENT shows the authoritative commit.</p>
<label>Evidence view <select id="view">OPTIONS</select></label><label>Decision <select id="decision"></select></label><button id="previous">Previous</button><button id="next">Next</button><p id="label"></p><main id="evidence"></main>
<script id="data" type="application/json">PAYLOAD</script>
<script>
const views=JSON.parse(document.getElementById('data').textContent),view=document.getElementById('view'),decision=document.getElementById('decision'),evidence=document.getElementById('evidence');
const count=views.public.length;
for(let i=0;i<count;i++){const o=document.createElement('option');o.value=i;o.textContent=String(views.public[i].decision_id);decision.append(o);}
function section(title,value){const h=document.createElement('h2');h.textContent=title;const pre=document.createElement('pre');pre.textContent=JSON.stringify(value,null,2);evidence.append(h,pre);}
function render(){evidence.replaceChildren();document.getElementById('label').textContent=view.value==='omniscient'?'OMNISCIENT — ANALYST-ONLY POST-GAME':view.value==='public'?'PUBLIC EVIDENCE':'PRIVATE SEAT EVIDENCE — '+view.value;const c=(views[view.value]||[])[Number(decision.value)];if(!c){section('No committed decisions',[]);return;}section('Decision',c.decision_id);section('Visible events and resolution',c.events||[]);section('Action attempts: seat observation, specification, proposed action, provider result',c.attempts||[]);section('Detached belief probes',c.probes||[]);if(view.value==='omniscient'){section('Authoritative commit, actions and replay hashes',c);}}
view.onchange=render;decision.onchange=render;document.getElementById('previous').onclick=()=>{decision.selectedIndex=Math.max(0,decision.selectedIndex-1);render();};document.getElementById('next').onclick=()=>{decision.selectedIndex=Math.min(count-1,decision.selectedIndex+1);render();};render();
</script></html>'''.replace('OPTIONS',options).replace('PAYLOAD',payload)
    path=directory/'replay.html'
    path.write_text(page, encoding='utf-8')
    return path, [p for commit in omni for p in commit.get('probes', [])]
