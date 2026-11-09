"""Five-seat Avalon, fixed benchmark discussion protocol (not optional tabletop roles)."""
import copy
import random

ROLES = ['Merlin', 'Servant', 'Servant', 'Assassin', 'Minion']
EVIL = {'Assassin', 'Minion'}
SIZES = [2, 3, 2, 3, 3]


class Avalon:
    name = 'avalon'
    version = 'avalon-v1'

    def reset(self, config, seed):
        rng = random.Random(seed)
        roles = list(config.get('roles', ROLES))
        if sorted(roles) != sorted(ROLES):
            raise ValueError('fixed five-seat role set required')
        if 'roles' not in config:
            rng.shuffle(roles)
        rounds = config.get('discussion_rounds', 2)
        if rounds not in (0, 2):
            raise ValueError('discussion_rounds must be 0 or 2; creates distinct pool')
        return dict(roles=roles, leader=config.get('leader', seed % 5) % 5,
                    speaking_start=seed % 5, discussion_rounds=rounds, phase='propose',
                    step=0, quest=0, successes=0, failures=0, rejections=0,
                    team=[], speech_index=0, public=[], winner=None)

    def pending_decision(self, s):
        phase = s['phase']
        if phase == 'terminal':
            return dict(id=str(s['step']), seats=[], simultaneous=False, visibility='public')
        if phase == 'propose':
            seats = [s['leader']]
        elif phase in ('discuss', 'final_discuss'):
            seats = [(s['speaking_start'] + s['speech_index']) % 5]
        elif phase == 'vote':
            seats = list(range(5))
        elif phase == 'quest':
            seats = list(s['team'])
        else:
            seats = [s['roles'].index('Assassin')]
        return dict(id=str(s['step']), seats=seats, simultaneous=phase in ('vote', 'quest'),
                    visibility='private' if phase == 'quest' else 'public')

    def observe(self, s, seat):
        role = s['roles'][seat]
        known = [i for i, r in enumerate(s['roles']) if r in EVIL] if role in EVIL | {'Merlin'} else []
        return dict(game=self.name, protocol=self.version, rules='Good wins three quests and survives assassination; evil wins three failed quests, five rejected teams, or Assassin identifying Merlin after three successes. Quest sizes 2/3/2/3/3; one failure fails a quest. Good must succeed; evil may fail. Team approval requires 3 of 5 votes. Merlin knows evil; evil know each other. Public speech is limited to 100 words.', seat=seat, role=role,
                    known_evil=known, phase=s['phase'], quest=s['quest'],
                    quest_size=SIZES[min(s['quest'], 4)], leader=s['leader'],
                    successes=s['successes'], failures=s['failures'], rejections=s['rejections'],
                    team=list(s['team']), public=copy.deepcopy(s['public']))

    def action_spec(self, s, seat):
        phase = s['phase']
        if phase == 'propose':
            n = SIZES[s['quest']]
            return dict(kind='subset', options=list(range(5)), min=n, max=n, field='team')
        if phase in ('discuss', 'final_discuss'):
            return dict(kind='speech', field='text', max_words=100, example={'text':'Public message'})
        if phase == 'vote':
            return dict(kind='choice', choices=[{'approve': True}, {'approve': False}])
        if phase == 'quest':
            return dict(kind='choice', choices=[{'success': True}] + ([{'success': False}] if s['roles'][seat] in EVIL else []))
        if phase == 'assassinate':
            return dict(kind='choice', choices=[{'target': i} for i in range(5) if s['roles'][i] not in EVIL])
        raise ValueError('no action at terminal')

    def validate(self, s, seat, action):
        if seat not in self.pending_decision(s)['seats'] or not isinstance(action, dict):
            raise ValueError('unexpected actor/action')
        spec = self.action_spec(s, seat)
        if spec['kind'] == 'choice':
            # JSON identity distinguishes booleans from integers.
            from .common import canonical
            if canonical(action) not in [canonical(x) for x in spec['choices']]:
                raise ValueError('action not in legal choices')
        elif spec['kind'] == 'speech':
            if set(action) != {'text'} or not isinstance(action['text'], str) or len(action['text'].split()) > 100 or len(action['text']) > 8000:
                raise ValueError('speech must be exactly a JSON object with one text string field: {"text":"message"}, at most 100 words and 8000 characters')
        else:
            team = action.get('team')
            if set(action) != {'team'} or not isinstance(team, list) or any(type(i) is not int or i not in range(5) for i in team) or len(set(team)) != spec['min'] or len(team) != spec['min']:
                raise ValueError('wrong quest team')

    def resolve(self, state, actions, rng):
        batch = self.pending_decision(state)
        if set(actions) != set(batch['seats']):
            raise ValueError('exact decision batch required')
        for seat, action in actions.items():
            self.validate(state, seat, action)
        s = copy.deepcopy(state)
        events = []
        phase = s['phase']
        if phase == 'propose':
            s['team'] = sorted(actions[s['leader']]['team'])
            s['speech_index'] = 0
            s['phase'] = 'discuss' if s['discussion_rounds'] else 'vote'
            events.append(dict(type='proposal', leader=s['leader'], team=s['team']))
        elif phase in ('discuss', 'final_discuss'):
            seat = batch['seats'][0]
            events.append(dict(type='speech', seat=seat, text=actions[seat]['text']))
            s['speech_index'] += 1
            if s['speech_index'] == (5 if phase == 'final_discuss' else 5 * s['discussion_rounds']):
                s['phase'] = 'assassinate' if phase == 'final_discuss' else 'vote'
        elif phase == 'vote':
            votes = {str(k): v['approve'] for k, v in actions.items()}
            approved = sum(votes.values()) > 2
            events.append(dict(type='votes', votes=votes, approved=approved))
            s['leader'] = (s['leader'] + 1) % 5
            if approved:
                s['rejections'] = 0
                s['phase'] = 'quest'
            else:
                s['rejections'] += 1
                s['phase'] = 'propose'
                if s['rejections'] == 5:
                    s['winner'], s['phase'] = 'evil', 'terminal'
        elif phase == 'quest':
            failures = sum(not v['success'] for v in actions.values())
            s['failures' if failures else 'successes'] += 1
            events.append(dict(type='quest', quest=s['quest'], failures=failures, success=not failures))
            s['quest'] += 1
            if s['failures'] == 3:
                s['winner'], s['phase'] = 'evil', 'terminal'
            elif s['successes'] == 3:
                s['phase'], s['speech_index'] = ('final_discuss' if s['discussion_rounds'] else 'assassinate'), 0
            else:
                s['phase'] = 'propose'
        else:
            target = next(iter(actions.values()))['target']
            s['winner'] = 'evil' if s['roles'][target] == 'Merlin' else 'good'
            s['phase'] = 'terminal'
            events.append(dict(type='assassination', target=target, winner=s['winner']))
        for event in events:
            event['visibility'] = 'public'
            s['public'].append(copy.deepcopy(event))
        s['step'] += 1
        return s, events

    def outcome(self, s):
        return dict(winner=s['winner'], reason='assassination' if s['successes'] == 3 else 'rejections' if s['rejections'] == 5 else 'quests') if s['winner'] else None
