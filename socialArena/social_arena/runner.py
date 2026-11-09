"""Decision-boundary journal, response caching, frozen batches, resume and replay."""
import ast
import copy
import fcntl
import json
import math
import random
import urllib.error
from pathlib import Path
from .avalon import Avalon, EVIL
from .budget import Ledger, BudgetExceeded
from .common import append_json, atomic_json, canonical, digest, read_json, read_stream
from .policies import definition, local_act, request_payload, openai_call


class TransportFailure(RuntimeError):
    pass


def composition_id(policies, roles, evil):
    members = sorted((role, policies[i]) for i,role in enumerate(roles) if (role in EVIL) == evil)
    unique = {policy for role, policy in members}
    return next(iter(unique)) if len(unique) == 1 else digest(['composition-v1', members])


def adapter(name):
    if name == 'avalon':
        return Avalon()
    if name == 'dominion':
        from .dominion import Dominion
        return Dominion()
    raise ValueError('unknown game')


def code_identity():
    return {p.name: digest(p.read_text()) for p in sorted(Path(__file__).parent.glob('*.py'))}


def freeze(config):
    policies = {name: definition(p) for name, p in config.get('policies', {'random': {'kind': 'random'}, 'scripted': {'kind': 'scripted'}}).items()}
    names = list(policies)
    game = config.get('game', 'avalon')
    a = adapter(game)
    if len(names) < 1:
        raise ValueError('at least one policy')
    scenarios = int(config.get('scenarios', 50))
    if scenarios < 1:
        raise ValueError('scenarios must be positive')
    schedule = []
    # Explicit schedule supports interventions and independently registered holdouts.
    if 'schedule' in config:
        schedule = copy.deepcopy(config['schedule'])
    else:
        for i in range(scenarios):
            first, second = names[i % len(names)], names[(i + 1) % len(names)]
            seed = int(config.get('seed', 1)) + i
            for swap in range(2):
                x, y = (second, first) if swap else (first, second)
                setup = copy.deepcopy(config.get('game_config', {}))
                if game == 'avalon':
                    setup['leader'] = i % 5
                    state = a.reset(setup, seed)
                    seats = [y if role in EVIL else x for role in state['roles']]
                else:
                    setup['starting_player'] = swap
                    # Reverse entrant-to-first/second assignment while retaining seat shuffle streams.
                    seats = [first, second]
                schedule.append(dict(game_id=f'g{i:04d}-{swap}', scenario_id=f's{i:04d}', pair_id=f'p{i:04d}', seed=seed,
                                     config=setup, policies=seats, self_play=first == second))
    for item in schedule:
        item.setdefault('self_play', len(set(item['policies'])) == 1)
        item.setdefault('config', copy.deepcopy(config.get('game_config', {})))
        if any(name not in policies for name in item['policies']):
            raise ValueError('schedule refers to undefined policy')
        if len(item['policies']) != (5 if game == 'avalon' else 2):
            raise ValueError('incorrect seat count')
    def protocol_config(setup):
        defaults = {'discussion_rounds':2} if game == 'avalon' else {'max_turns':200}
        defaults.update({k:v for k,v in setup.items() if k not in ('leader','roles','starting_player')})
        return defaults
    protocols = {canonical(protocol_config(x['config'])) for x in schedule}
    if len(protocols) > 1:
        raise ValueError('mixed protocol configurations require separate experiments/pools')
    if len({x['game_id'] for x in schedule}) != len(schedule):
        raise ValueError('duplicate game IDs')
    for x in schedule:
        if not all(str(x[k]).replace('-', '').replace('_', '').isalnum() for k in ('game_id','scenario_id','pair_id')):
            raise ValueError('IDs must be simple alphanumeric, hyphen or underscore')
    manifest = dict(schema='social-arena-v1', game=game, engine=a.version, code=code_identity(), policies=policies,
                    schedule=schedule, budget=dict(max_usd=float(config.get('max_usd', 0)), allow_paid=bool(config.get('allow_paid', False))),
                    max_decisions=int(config.get('max_decisions', 10000)), max_total_decisions=int(config.get('max_total_decisions', 1000000)), probes=bool(config.get('probes', False)),
                    protocol=json.loads(next(iter(protocols))), registration=config.get('registration', {}))
    if not math.isfinite(manifest['budget']['max_usd']) or manifest['budget']['max_usd'] < 0 or manifest['max_decisions'] < 1 or manifest['max_total_decisions'] < 1:
        raise ValueError('invalid budget/decision cap')
    manifest['experiment_id'] = digest(manifest)
    return manifest


class Experiment:
    def __init__(self, directory, config=None):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / 'manifest.json'
        proposed = freeze(config) if config is not None else None
        if path.exists():
            self.manifest = read_json(path)
            if proposed is not None and proposed != self.manifest:
                raise ValueError('immutable experiment differs; choose a new output directory')
        elif proposed:
            atomic_json(path, proposed)
            self.manifest = proposed
        else:
            raise ValueError('manifest missing')
        if self.manifest['code'] != code_identity():
            raise ValueError('engine code differs from pinned manifest; replay/resume requires exact code')
        self._total_decisions = sum(len(read_stream(p)) for p in (self.root / 'games').glob('*/events.jsonl'))
        self.game = adapter(self.manifest['game'])
        self.ledger = Ledger(self.root / 'ledger.json', **self.manifest['budget'])

    def _call(self, policy, obs, spec, request_id, kind='action', repair=None):
        cache = self.root / 'requests' / (request_id + '.json')
        if cache.exists():
            return read_json(cache)
        if policy['kind'] != 'openai':
            result = dict(action=local_act(obs, spec, policy, request_id), usage=None, cost_usd=0)
            atomic_json(cache, result)
            return result
        for price in ('input_usd_per_million', 'output_usd_per_million'):
            if not isinstance(policy.get(price), (int, float)) or policy[price] < 0 or not policy.get('price_source'):
                raise BudgetExceeded('unknown model price/provenance blocks paid dispatch')
        payload = request_payload(obs, spec, policy, repair)
        # UTF-8 bytes upper bound ordinary tokenizer tokens, plus generous framing reserve.
        upper_input = len(canonical(payload).encode()) + 4096
        reserve = (upper_input * policy['input_usd_per_million'] + policy['max_output_tokens'] * policy['output_usd_per_million']) / 1e6
        last = None
        for retry in range(policy['transport_retries'] + 1):
            rid = request_id + '-t' + str(retry)
            self.ledger.reserve(rid, reserve, dict(kind=kind, policy=policy['id'], request_id=request_id, game_id=getattr(self, '_active_game', None)))
            try:
                result = openai_call(payload, rid)
            except Exception as error:
                # Network and server failures can have uncertain charges; retain reservation.
                self.ledger.settle(rid, error=type(error).__name__, http_status=getattr(error, 'code', None), provider_detail=getattr(error, 'provider_detail', None))
                last = error
                if isinstance(error, urllib.error.HTTPError) and error.code in (408, 409, 429, 500, 502, 503, 504):
                    continue
                raise TransportFailure(type(error).__name__ + ': ' + str(getattr(error, 'code', '')) + ' ' + canonical(getattr(error, 'provider_detail', {}))) from error
            usage = result.get('usage')
            actual = None if not usage else (usage['input_tokens'] * policy['input_usd_per_million'] + usage['output_tokens'] * policy['output_usd_per_million']) / 1e6
            self.ledger.settle(rid, actual, usage=usage, response_id=result.get('response_id'), model=result.get('model'), service_tier=result.get('service_tier'))
            try:
                action = json.loads(result['text'])['action']
            except (ValueError, KeyError, TypeError):
                action = None
            result.update(action=action, cost_usd=actual, request_id=rid)
            atomic_json(cache, result)
            return result
        raise TransportFailure('transport retries exhausted') from last

    def run_game(self, item, crash_after=None):
        directory = self.root / 'games' / item['game_id']
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._run_locked(directory, item, crash_after)

    def _run_locked(self, directory, item, crash_after):
        summary_path = directory / 'summary.json'
        if summary_path.exists() and read_json(summary_path)['status'] in ('complete', 'policy_failure'):
            return read_json(summary_path)
        self._active_game = item['game_id']
        state = self.game.reset(item['config'], item['seed'])
        initial = dict(state=state, state_hash=digest(state), seed=item['seed'], visibility='omniscient')
        if (directory / 'initial.json').exists():
            if read_json(directory / 'initial.json') != initial:
                raise ValueError('setup changed')
        else:
            atomic_json(directory / 'initial.json', initial)
        rng = random.Random(item['seed'])
        commits = read_stream(directory / 'events.jsonl')
        for commit in commits:
            if digest(state) != commit['before_hash']:
                raise ValueError('replay prestate mismatch')
            state, events = self.game.resolve(state, {int(k): v for k,v in commit['actions'].items()}, rng)
            if digest(state) != commit['state_hash'] or events != commit['events'] or repr(rng.getstate()) != commit['rng_after']:
                raise ValueError('replay mismatch')
        status, reason, failing_policy = 'complete', None, None
        invalid = sum(x.get('invalid_actions', 0) for x in commits)
        count = len(commits)
        while self.game.outcome(state) is None:
            total = self._total_decisions
            if count >= self.manifest['max_decisions'] or total >= self.manifest['max_total_decisions']:
                status, reason = 'incomplete', 'decision_cap'
                break
            batch = self.game.pending_decision(state)
            # Build every observation/spec before any call, including for simultaneous batches.
            views = {seat: copy.deepcopy(self.game.observe(state, seat)) for seat in batch['seats']}
            specs = {seat: copy.deepcopy(self.game.action_spec(state, seat)) for seat in batch['seats']}
            actions, attempts = {}, []
            before_invalid = invalid
            try:
                for seat in batch['seats']:
                    policy = self.manifest['policies'][item['policies'][seat]]
                    failing_policy = policy['id']
                    repair = None
                    for attempt in range(2):
                        rid = digest([self.manifest['experiment_id'], item['game_id'], batch['id'], seat, attempt])
                        result = self._call(policy, views[seat], specs[seat], rid, repair=repair)
                        action = result['action']
                        entry = dict(seat=seat, request_id=rid, policy=policy['id'], observation_hash=digest(views[seat]), observation=views[seat], spec=specs[seat], action=action, visibility=[seat], result=result)
                        attempts.append(entry)
                        try:
                            self.game.validate(state, seat, action)
                            actions[seat] = action
                            break
                        except (ValueError, TypeError, KeyError) as error:
                            invalid += 1
                            repair = str(error)
                            append_json(directory / 'attempts.jsonl', dict(**entry, error=repair))
                    else:
                        raise ValueError('schema repair exhausted')
                failing_policy = None
            except BudgetExceeded as error:
                status, reason = 'incomplete', str(error)
                break
            except TransportFailure as error:
                status, reason = 'incomplete', 'infrastructure: ' + str(error)
                append_json(directory / 'failures.jsonl', dict(decision_id=batch['id'], policy=failing_policy, error=reason, attempts=attempts))
                break
            except Exception as error:
                status, reason = 'policy_failure', type(error).__name__ + ': ' + str(error)
                append_json(directory / 'failures.jsonl', dict(decision_id=batch['id'], policy=failing_policy, error=reason, attempts=attempts))
                break
            probes = []
            if self.manifest['probes'] and self.game.name == 'avalon' and state['phase'] in ('vote', 'assassinate'):
                try:
                    for seat in range(5):
                        if state['roles'][seat] not in ('Servant', 'Merlin'):
                            continue
                        obs = copy.deepcopy(self.game.observe(state, seat))
                        targets = [i for i in range(5) if i != seat and i not in obs['known_evil']]
                        if state['roles'][seat] == 'Merlin':
                            targets = []  # all labels mechanically known for this fixed role set
                        if not targets:
                            continue
                        pid = digest([self.manifest['experiment_id'], item['game_id'], batch['id'], seat, 'belief'])
                        policy = self.manifest['policies'][item['policies'][seat]]
                        result = self._call(policy, obs, {'kind':'belief', 'targets':targets, 'instruction':'Return JSON probabilities mapping target seat to P(evil), each in [0,1].'}, pid, kind='probe')
                        probabilities = result['action'].get('probabilities', {})
                        if set(probabilities) != {str(i) for i in targets} or any(type(x) not in (int,float) or not 0 <= x <= 1 for x in probabilities.values()):
                            raise ValueError('invalid detached belief probabilities')
                        score = sum((probabilities[str(i)] - (state['roles'][i] in EVIL)) ** 2 for i in targets) / len(targets)
                        probes.append(dict(seat=seat, role=state['roles'][seat], phase=state['phase'], request_id=pid, result=result, brier=score, visibility=[seat]))
                except BudgetExceeded as error:
                    status, reason = 'incomplete', str(error)
                    break
                except Exception as error:
                    append_json(directory / 'probe_failures.jsonl', dict(decision_id=batch['id'], error=type(error).__name__))
            before = digest(state)
            next_state, events = self.game.resolve(state, actions, rng)
            commit = dict(decision_id=batch['id'], batch=batch, before_hash=before, actions=actions,
                          attempts=attempts, probes=probes, events=events, state_hash=digest(next_state), rng_after=repr(rng.getstate()),
                          invalid_actions=invalid-before_invalid, visibility='omniscient')
            append_json(directory / 'events.jsonl', commit)
            state = next_state
            count += 1
            self._total_decisions += 1
            atomic_json(directory / 'checkpoint.json', dict(state=state, state_hash=digest(state), decisions=count, rng_state=repr(rng.getstate())))
            if crash_after is not None and count == crash_after:
                raise RuntimeError('injected crash')
        outcome = self.game.outcome(state)
        if outcome and outcome.get('complete') is False:
            status, reason = 'incomplete', outcome.get('reason')
        entries = self.ledger.entries()
        request_ids = {x['request_id'] for c in read_stream(directory / 'events.jsonl') for x in c['attempts']}
        costs = sum(v.get('actual_usd', v['reserved_usd']) for v in entries.values() if v.get('game_id') == item['game_id'])
        summary = dict(**{k:item[k] for k in ('game_id','scenario_id','pair_id','self_play')}, game=self.game.name,
                       status=status, reason=reason, outcome=outcome, policies=[self.manifest['policies'][x]['id'] for x in item['policies']],
                       decisions=count, invalid_actions=invalid, costs_usd=costs, failing_policy=failing_policy, final_state_hash=digest(state))
        if self.game.name == 'avalon':
            summary['good_policy'] = composition_id(summary['policies'], state['roles'], False)
            summary['evil_policy'] = composition_id(summary['policies'], state['roles'], True)
        else:
            start = item['config'].get('starting_player', 0)
            summary.update(first_policy=summary['policies'][start], second_policy=summary['policies'][1-start], starting_player=start)
            if outcome and type(outcome.get('winner')) is int:
                summary['outcome'] = dict(outcome, winner=0 if outcome['winner'] == start else 1)
        atomic_json(summary_path, summary)
        return summary

    def run(self):
        with (self.root / 'experiment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._total_decisions = sum(len(read_stream(p)) for p in (self.root / 'games').glob('*/events.jsonl'))
            return self._run_all()

    def _run_all(self):
        records = [self.run_game(item) for item in self.manifest['schedule']]
        atomic_json(self.root / 'results.json', records)
        return records


def replay(directory):
    directory = Path(directory)
    experiment = Experiment(directory.parent.parent)
    item = next(x for x in experiment.manifest['schedule'] if x['game_id'] == directory.name)
    state = experiment.game.reset(item['config'], item['seed'])
    rng = random.Random(item['seed'])
    for c in read_stream(directory / 'events.jsonl'):
        if digest(state) != c['before_hash']:
            raise ValueError('replay before hash differs')
        state, events = experiment.game.resolve(state, {int(k):v for k,v in c['actions'].items()}, rng)
        if digest(state) != c['state_hash'] or events != c['events'] or repr(rng.getstate()) != c['rng_after']:
            raise ValueError('replay differs')
    return dict(state_hash=digest(state), outcome=experiment.game.outcome(state))


def export_view(directory, seat=None, omniscient=False):
    """Only post-game analysts may request omniscient; never pass raw journals to policies."""
    result = []
    for c in read_stream(Path(directory) / 'events.jsonl'):
        if omniscient:
            result.append(c)
            continue
        attempts = [x for x in c['attempts'] if seat is not None and x['seat'] == seat]
        events = [x for x in c['events'] if x.get('visibility') == 'public' or seat is not None and seat in x.get('visibility', [])]
        result.append(dict(decision_id=c['decision_id'], events=events, attempts=attempts, probes=[x for x in c.get('probes', []) if seat is not None and x['seat'] == seat]))
    return result
