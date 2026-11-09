"""Bounded two-player Dominion, second edition First Game kingdom.

Rules reference: Rio Grande Games Dominion second edition rulebook. This
independent adapter implements only the fixed kingdom, without vendored assets.
"""
from collections import Counter
from copy import deepcopy
import random

KINGDOM = ['Cellar', 'Market', 'Merchant', 'Militia', 'Mine', 'Moat', 'Remodel', 'Smithy', 'Village', 'Workshop']
COST = dict(zip(['Copper','Silver','Gold','Estate','Duchy','Province','Curse'] + KINGDOM,
                [0,3,6,2,5,8,0,2,5,3,4,5,2,4,4,3,3]))
TREASURE = {'Copper': 1, 'Silver': 2, 'Gold': 3}
POINTS = {'Estate': 1, 'Duchy': 3, 'Province': 6, 'Curse': -1}


class Dominion:
    name = 'dominion'
    version = 'dominion-v1'

    def reset(self, config, seed):
        start = config.get('starting_player', 0)
        if start not in (0, 1):
            raise ValueError('starting_player must be 0 or 1')
        if not isinstance(config.get('max_turns', 200), int) or config.get('max_turns', 200) < 1:
            raise ValueError('max_turns must be positive')
        supply = {c: 10 for c in KINGDOM}
        supply.update(Copper=46, Silver=40, Gold=30, Estate=8, Duchy=8, Province=8, Curse=10)
        s = dict(seed=seed, supply=supply, trash=[], players=[], active=start,
                 phase='action', counter=0, actions=1, buys=1, coins=0,
                 merchants=0, silver_played=False, turns=[0, 0],
                 max_turns=config.get('max_turns', 200), terminal=None)
        for seat in range(2):
            s['players'].append(dict(deck=[], hand=[], discard=['Copper']*7 + ['Estate']*3,
                                     played=[], shuffles=0))
            self._draw(s, seat, 5, [])
        return s

    def _draw(self, s, seat, count, events):
        p = s['players'][seat]
        drawn = []
        for _ in range(count):
            if not p['deck'] and p['discard']:
                p['deck'], p['discard'] = p['discard'], []
                random.Random(f"{s['seed']}:{seat}:{p['shuffles']}").shuffle(p['deck'])
                p['shuffles'] += 1
                events.append(dict(type='shuffle', seat=seat, visibility=[seat], order=p['deck'][:]))
            if p['deck']:
                card = p['deck'].pop()
                p['hand'].append(card)
                drawn.append(card)
        if drawn:
            events.append(dict(type='draw', seat=seat, cards=drawn, visibility=[seat]))

    def pending_decision(self, s):
        seats = [] if s['terminal'] else [1-s['active'] if s['phase'] in ('reaction','militia_discard') else s['active']]
        return dict(id=f"dominion:{s['counter']}", seats=seats, simultaneous=False, visibility='private')

    def observe(self, s, seat):
        if seat not in (0, 1):
            raise ValueError('invalid seat')
        result = {k: deepcopy(s[k]) for k in ('supply','trash','active','phase','counter','actions','buys','coins','turns','terminal')}
        result['rules'] = 'Score Estate=1, Duchy=3, Province=6, Curse=-1. End after Province pile empty or three supply piles empty, at end of turn. Ties favor fewer turns. Each turn: 1 action, play treasures, 1 buy, cleanup draw 5; discard shuffles into deck when needed. Copper/Silver/Gold worth 1/2/3 coins. Cellar +1 action discard any draw same; Market +1 card/action/buy/coin; Merchant +1 card/action, first Silver +1 coin; Militia +2 coins, opponent discards to 3 unless Moat; Mine optional trash treasure gain treasure up to +3 cost to hand; Moat +2 cards or reveal to block attack; Remodel trash card gain up to +2 cost; Smithy +3 cards; Village +1 card/+2 actions; Workshop gain up to cost4. Gains to discard unless Mine.'
        result['seat'] = seat
        result['players'] = [dict(hand=p['hand'][:] if i == seat else len(p['hand']),
                                  deck_size=len(p['deck']), discard=p['discard'][:], played=p['played'][:])
                             for i,p in enumerate(s['players'])]
        return result

    def action_spec(self, s, seat):
        if seat not in self.pending_decision(s)['seats']:
            raise ValueError('seat has no pending decision')
        hand = s['players'][seat]['hand']
        phase = s['phase']
        if phase in ('cellar','militia_discard'):
            n = max(0, len(hand)-3) if phase == 'militia_discard' else 0
            return dict(kind='subset', options=list(range(len(hand))), min=n, max=n if phase == 'militia_discard' else len(hand))
        choices = []
        if phase == 'action':
            choices = [dict(type='end_actions')]
            if s['actions']:
                choices += [dict(type='play', card=c) for c in sorted(set(hand)) if c in KINGDOM]
        elif phase == 'buy':
            choices = [dict(type='end_turn')]
            if not s.get('bought', False):
                choices += [dict(type='treasure', card=c) for c in sorted(set(hand)) if c in TREASURE]
            if s['buys']:
                choices += [dict(type='buy', card=c) for c in sorted(COST) if s['supply'][c] and COST[c] <= s['coins']]
        elif phase == 'reaction':
            choices = [dict(type='allow')]
            if 'Moat' in hand:
                choices += [dict(type='reveal_moat')]
        elif phase in ('mine','remodel'):
            choices = [dict(type='trash', card=c) for c in sorted(set(hand)) if phase == 'remodel' or c in TREASURE]
            if phase == 'mine' or not choices:
                choices += [dict(type='skip')]
        elif phase == 'gain':
            choices = [dict(type='gain', card=c) for c in sorted(COST) if s['supply'][c] and COST[c] <= s['gain_limit'] and (not s['gain_treasure'] or c in TREASURE)]
            if not choices:
                choices = [dict(type='skip')]
        return dict(kind='choice', choices=choices)

    def validate(self, s, seat, action):
        spec = self.action_spec(s, seat)
        if not isinstance(action, dict):
            raise ValueError('action must be an object')
        if spec['kind'] == 'choice':
            if action not in spec['choices']:
                raise ValueError('illegal choice')
        else:
            indices = action.get('indices')
            if set(action) != {'indices'} or not isinstance(indices,list) or any(type(i) is not int for i in indices) or len(set(indices)) != len(indices) or not spec['min'] <= len(indices) <= spec['max'] or any(i not in spec['options'] for i in indices):
                raise ValueError('illegal discard subset')

    def resolve(self, state, actions, rng):
        seats = self.pending_decision(state)['seats']
        if set(actions) != set(seats) or not seats:
            raise ValueError('incorrect decision seats')
        seat = seats[0]
        self.validate(state, seat, actions[seat])
        s = deepcopy(state)
        a = actions[seat]
        phase = s['phase']
        p = s['players'][seat]
        events = [dict(type='action', seat=seat, action=deepcopy(a), visibility=[seat])]
        s['counter'] += 1
        kind = a.get('type')
        card = a.get('card')
        if phase in ('cellar','militia_discard'):
            discarded = [p['hand'][i] for i in sorted(a['indices'])]
            for i in sorted(a['indices'], reverse=True):
                p['hand'].pop(i)
            p['discard'].extend(discarded)
            events.append(dict(type='discard', seat=seat, cards=discarded, visibility='public'))
            if phase == 'cellar':
                self._draw(s,seat,len(discarded),events)
            s['phase'] = 'action'
        elif kind == 'end_actions':
            s['phase'] = 'buy'
        elif kind == 'end_turn':
            self._finish_turn(s, events)
        elif kind in ('treasure','play'):
            p['hand'].remove(card)
            p['played'].append(card)
            events.append(dict(type='play', seat=seat, card=card, visibility='public'))
            if kind == 'treasure':
                s['coins'] += TREASURE[card]
                if card == 'Silver' and not s['silver_played']:
                    s['coins'] += s['merchants']
                    s['silver_played'] = True
            else:
                s['actions'] -= 1
                self._play(s,seat,card,events)
        elif kind == 'buy':
            s['buys'] -= 1
            s['coins'] -= COST[card]
            s['bought'] = True
            self._gain(s,seat,card,'discard',events)
        elif kind in ('allow','reveal_moat'):
            if kind == 'reveal_moat':
                events.append(dict(type='reaction',seat=seat,card='Moat',visibility='public'))
            s['phase'] = 'militia_discard' if kind == 'allow' and len(p['hand']) > 3 else 'action'
        elif kind == 'trash':
            p['hand'].remove(card)
            s['trash'].append(card)
            events.append(dict(type='trash',seat=seat,card=card,visibility='public'))
            s.update(phase='gain', gain_limit=COST[card]+(3 if phase=='mine' else 2), gain_treasure=phase=='mine', gain_destination='hand' if phase=='mine' else 'discard')
        elif kind == 'gain':
            self._gain(s,seat,card,s['gain_destination'],events)
            s['phase'] = 'action'
        elif kind == 'skip':
            s['phase'] = 'action'
        self.check_invariants(s)
        return s, events

    def _gain(self,s,seat,card,destination,events):
        s['supply'][card] -= 1
        s['players'][seat][destination].append(card)
        events.append(dict(type='gain',seat=seat,card=card,destination=destination,visibility='public'))

    def _play(self,s,seat,card,events):
        draws = {'Market':1,'Merchant':1,'Moat':2,'Smithy':3,'Village':1}
        if card in draws:
            self._draw(s,seat,draws[card],events)
        s['actions'] += {'Cellar':1,'Market':1,'Merchant':1,'Village':2}.get(card,0)
        if card == 'Market':
            s['buys'] += 1
            s['coins'] += 1
        elif card == 'Merchant':
            s['merchants'] += 1
        elif card == 'Militia':
            s['coins'] += 2
            s['phase'] = 'reaction'
        elif card in ('Cellar','Mine','Remodel'):
            s['phase'] = card.lower()
        elif card == 'Workshop':
            s.update(phase='gain',gain_limit=4,gain_treasure=False,gain_destination='discard')

    def _finish_turn(self,s,events):
        seat = s['active']
        p = s['players'][seat]
        p['discard'].extend(p['hand'] + p['played'])
        p['hand'], p['played'] = [], []
        s['turns'][seat] += 1
        self._draw(s,seat,5,events)
        if s['supply']['Province'] == 0 or sum(v==0 for v in s['supply'].values()) >= 3:
            scores = [sum(POINTS.get(c,0) for zone in ('hand','deck','discard','played') for c in player[zone]) for player in s['players']]
            best = max((scores[i],-s['turns'][i]) for i in range(2))
            winners = [i for i in range(2) if (scores[i],-s['turns'][i]) == best]
            s['terminal'] = dict(winner=winners[0] if len(winners)==1 else None, winners=winners, scores=scores, turns=s['turns'][:], complete=True)
        elif sum(s['turns']) >= s['max_turns']:
            s['terminal'] = dict(winner=None,complete=False,reason='max_turns',turns=s['turns'][:])
        s.update(active=1-seat,phase='action',actions=1,buys=1,coins=0,merchants=0,silver_played=False,bought=False)

    def outcome(self,s):
        return deepcopy(s['terminal'])

    def check_invariants(self,s):
        total = Counter(s['supply']) + Counter(s['trash'])
        for p in s['players']:
            for zone in ('hand','deck','discard','played'):
                total.update(p[zone])
        expected = Counter({c:10 for c in KINGDOM})
        expected.update(Copper=60,Silver=40,Gold=30,Estate=14,Duchy=8,Province=8,Curse=10)
        if total != expected or any(v<0 for v in s['supply'].values()):
            raise ValueError('card conservation violated')
        if s['actions'] < 0 or s['buys'] < 0 or s['coins'] < 0:
            raise ValueError('negative turn resources')
