"""Batch, scenario-block ratings. Logit strengths are centered within each game pool."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

ELO = 400 / math.log(10)


def _entrants(r):
    if r.get('game', 'avalon') == 'avalon':
        return r.get('good_policy'), r.get('evil_policy')
    p = r.get('policies', [])
    return r.get('first_policy', p[0] if p else None), r.get('second_policy', p[1] if len(p) > 1 else None)


def _result(r):
    w = r.get('outcome', {}).get('winner')
    if r.get('game', 'avalon') == 'avalon':
        return 0 if w == 'good' else 1 if w == 'evil' else None
    return 2 if w is None else w if w in (0, 1) else None


def _solve(a, b):
    a = [list(row) + [v] for row, v in zip(a, b)]
    for k in range(len(b)):
        pivot = max(range(k, len(b)), key=lambda i: abs(a[i][k]))
        a[k], a[pivot] = a[pivot], a[k]
        v = a[k][k]
        a[k] = [x / v for x in a[k]]
        for i in range(len(b)):
            if i != k:
                v = a[i][k]
                a[i] = [x - v*y for x, y in zip(a[i], a[k])]
    return [row[-1] for row in a]


def _fit(rows, names, game, sd=1.5):
    n = len(names)
    size = n + 1 + (game == 'dominion')
    index = {name: i for i, name in enumerate(names)}
    observations = []
    for r in rows:
        a, b = _entrants(r)
        x = [0.] * size
        x[index[a]] += 1
        x[index[b]] -= 1
        x[n] = 1
        # Davidson: p(win A), p(win B), p(tie) proportional to
        # exp(d/2), exp(-d/2), exp(log tie propensity).
        if game == 'dominion':
            tie = [0.] * size
            tie[-1] = 1
            features = [[v / 2 for v in x], [-v / 2 for v in x], tie]
        else:
            features = [x, [0.] * size]
        observations.append((features, _result(r)))
    precision = [1 / sd**2] * n + [1 / 9] + ([1 / 9] if game == 'dominion' else [])
    theta = [0.] * size
    for _ in range(35):
        g = [p*t for p, t in zip(precision, theta)]
        h = [[precision[i] if i == j else 0. for j in range(size)] for i in range(size)]
        for features, y in observations:
            logits = [sum(t*x for t, x in zip(theta, f)) for f in features]
            m = max(logits)
            weights = [math.exp(v-m) for v in logits]
            probs = [v/sum(weights) for v in weights]
            mu = [sum(p*f[i] for p, f in zip(probs, features)) for i in range(size)]
            for i in range(size):
                g[i] += mu[i] - features[y][i]
                for j in range(size):
                    h[i][j] += sum(p*f[i]*f[j] for p, f in zip(probs, features)) - mu[i]*mu[j]
        delta = _solve(h, g)
        scale = min(1., 2 / max(2., max(abs(v) for v in delta)))
        theta = [t-scale*d for t, d in zip(theta, delta)]
        if max(abs(v) for v in delta) < 1e-7:
            break
    mean = sum(theta[:n])/n
    return [t-mean for t in theta[:n]] + theta[n:]


def _prob(theta, a, b, n, game):
    d = theta[a] - theta[b]
    def p(side):
        z = max(-700, min(700, d + side*theta[n]))
        if game == 'avalon':
            return 1 / (1 + math.exp(-z))
        # Expected score, retaining draws in the likelihood and prediction.
        vals = [z/2, -z/2, theta[-1]]
        m = max(vals)
        w = [math.exp(v-m) for v in vals]
        return (w[0] + .5*w[2])/sum(w)
    return (p(1) + p(-1))/2


def _interval(values):
    if not values:
        return None
    v = sorted(values)
    def q(p):
        x = (len(v)-1)*p
        i = int(x)
        return v[i] + (v[min(i+1, len(v)-1)]-v[i])*(x-i)
    return [q(.025), q(.975)]


def analyze(records: list[dict], bootstrap: int = 2000, seed: int = 0) -> dict:
    """Fit separate game pools; exclude incomplete pairs, resample entire scenarios.

    Unidentified compositions require explicit good_policy/evil_policy IDs. Failures
    never enter the primary fit. Bootstrap zero disables uncertainty estimation.
    """
    if bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    records = sorted(records, key=lambda r: (str(r.get('game')), str(r.get('scenario_id')), str(r.get('pair_id')), str(r.get('game_id'))))
    groups = defaultdict(list)
    for r in records:
        groups[(r.get('game', 'avalon'), r.get('scenario_id'), r.get('pair_id'))].append(r)
    eligible, exclusions = [], []
    for key, group in groups.items():
        valid = all(r.get('status') == 'complete' and all(_entrants(r)) and _result(r) is not None for r in group)
        paired = len(group) == 2 and _entrants(group[0]) == tuple(reversed(_entrants(group[1])))
        selfplay = len(group) == 1 and group[0].get('self_play') and _entrants(group[0])[0] == _entrants(group[0])[1]
        valid = valid and key[1] is not None and key[2] is not None and (paired or selfplay)
        if valid:
            eligible.extend(group)
        else:
            for r in group:
                reason = ('status_' + str(r.get('status'))) if r.get('status') != 'complete' else 'incomplete_or_invalid_pair'
                exclusions.append({'game_id': r.get('game_id'), 'policies': list(_entrants(r)), 'reason': reason})
    result = {'method': 'regularized Bradley–Terry / Davidson; scenario bootstrap', 'bootstrap': bootstrap, 'seed': seed,
              'counts': dict(Counter(r.get('status', 'unknown') for r in records)), 'total_games': len(records),
              'eligible_games': len(eligible), 'excluded_games': exclusions, 'pools': {},
              'cost_usd': sum(r.get('costs_usd', 0) or 0 for r in records),
              'invalid_actions': sum(r.get('invalid_actions', 0) or 0 for r in records),
              'decisions': sum(r.get('decisions', 0) or 0 for r in records)}
    reliability = defaultdict(lambda: {'games': 0, 'complete': 0, 'policy_failure': 0, 'incomplete': 0, 'excluded': 0})
    excluded_ids = {r['game_id'] for r in exclusions}
    for r in records:
        for policy in sorted(set(p for p in _entrants(r) if p)):
            reliability[policy]['games'] += 1
            status = r.get('status')
            if status in ('complete', 'policy_failure', 'incomplete'):
                reliability[policy][status] += 1
            if r.get('game_id') in excluded_ids:
                reliability[policy]['excluded'] += 1
    result['policy_reliability'] = dict(reliability)
    result['invalid_action_rate'] = result['invalid_actions']/result['decisions'] if result['decisions'] else None
    completed = result['counts'].get('complete', 0)
    result['completion_rate'] = completed/len(records) if records else None
    result['cost_per_finished_game'] = result['cost_usd']/completed if completed else None
    for game in sorted({r.get('game', 'avalon') for r in eligible}):
        rows = [r for r in eligible if r.get('game', 'avalon') == game]
        names = sorted({p for r in rows for p in _entrants(r)})
        n = len(names)
        blocks = defaultdict(list)
        raw = {p: {'first_or_good': {'wins': 0, 'losses': 0, 'draws': 0}, 'second_or_evil': {'wins': 0, 'losses': 0, 'draws': 0}} for p in names}
        adjacency = {p: set() for p in names}
        match_counts = Counter()
        for r in rows:
            blocks[str(r['scenario_id'])].append(r)
            a, b = _entrants(r)
            adjacency[a].add(b); adjacency[b].add(a)
            match_counts[tuple(sorted((a,b)))] += 1
            y = _result(r)
            raw[a]['first_or_good']['draws' if y == 2 else 'wins' if y == 0 else 'losses'] += 1
            raw[b]['second_or_evil']['draws' if y == 2 else 'wins' if y == 1 else 'losses'] += 1
        seen, pending = set(), [names[0]]
        while pending:
            p = pending.pop()
            if p not in seen:
                seen.add(p); pending.extend(adjacency[p]-seen)
        warnings = ['Exploratory estimates; scenario counts and model assumptions limit interpretation.']
        if len(seen) != n:
            warnings.append('Disconnected matchup graph: relative strengths across components are prior-driven and unidentified by outcomes.')
        if len(blocks) < 20:
            warnings.append('Few independent scenario blocks; bootstrap intervals may be unstable.')
        if any(r.get('status') == 'policy_failure' for r in records if r.get('game', 'avalon') == game):
            warnings.append('Policy failures can bias the capability ranking; inspect policy reliability and failure-as-loss sensitivity.')
        theta = _fit(rows, names, game)
        rng = random.Random(seed)
        block_list = [blocks[k] for k in sorted(blocks)]
        samples = []
        for _ in range(bootstrap):
            sample = [r for block in rng.choices(block_list, k=len(block_list)) for r in block]
            samples.append(_fit(sample, names, game))
        ratings = {p: {'strength': theta[i], 'rating': 1500 + ELO*theta[i],
                        'rating_ci95': _interval([1500 + ELO*s[i] for s in samples]),
                        'strength_ci95': _interval([s[i] for s in samples]), 'raw': raw[p]} for i,p in enumerate(names)}
        matchups = []
        for i,a in enumerate(names):
            for j in range(i+1,n):
                b = names[j]
                matchups.append({'a': a, 'b': b, 'games': match_counts[tuple(sorted((a,b)))], 'rating_gap': ELO*(theta[i]-theta[j]),
                                 'gap_ci95': _interval([ELO*(s[i]-s[j]) for s in samples]),
                                 'side_averaged_expected_score': _prob(theta,i,j,n,game),
                                 'expected_score_ci95': _interval([_prob(s,i,j,n,game) for s in samples])})
        sensitivity = {str(sd): {p: 1500+ELO*t for p,t in zip(names,_fit(rows,names,game,sd))} for sd in (1.,3.)}
        failure_rows = []
        unresolved_failures = []
        for r in records:
            if r.get('game', 'avalon') != game or r.get('status') != 'policy_failure':
                continue
            a, b = _entrants(r)
            failed = r.get('failing_policy', r.get('failed_policy'))
            failed_side = r.get('failed_side')
            if a not in names or b not in names or a == b or (failed not in (a, b) and failed_side not in ('good', 'evil', 0, 1)):
                unresolved_failures.append(r.get('game_id'))
                continue
            first_failed = failed == a if failed in (a, b) else failed_side in ('good', 0)
            winner = ('evil' if first_failed else 'good') if game == 'avalon' else (1 if first_failed else 0)
            failure_rows.append(dict(r, outcome={'winner': winner}))
        failure_sensitivity = {'added_policy_failures': len(failure_rows), 'unattributed_or_unidentified_failures': unresolved_failures,
                               'ratings': {p: 1500+ELO*t for p,t in zip(names, _fit(rows+failure_rows,names,game))} if failure_rows else None,
                               'description': 'Eligible primary games plus attributable policy failures treated as losses; infrastructure interruptions remain missing. This sensitivity is unpaired.'}
        result['pools'][game] = {'ratings': ratings, 'matchups': matchups, 'independent_scenarios': len(blocks),
                               'eligible_games': len(rows), 'side_advantage_elo': ELO*theta[n],
                               'side_advantage_ci95': _interval([ELO*s[n] for s in samples]),
                               'tie_propensity': math.exp(theta[-1]) if game == 'dominion' else None,
                               'prior_sensitivity': sensitivity, 'failure_as_loss_sensitivity': failure_sensitivity, 'warnings': warnings}
    return result
