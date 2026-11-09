import json
import random
import unittest
from copy import deepcopy
from social_arena.dominion import Dominion, KINGDOM


class DominionTests(unittest.TestCase):
    def setUp(self):
        self.game = Dominion()
        self.state = self.game.reset({}, 17)

    def move_to_hand(self, card, seat=0):
        s = self.state
        s['supply'][card] -= 1
        s['players'][seat]['hand'].append(card)

    def act(self, action):
        seat = self.game.pending_decision(self.state)['seats'][0]
        self.state, events = self.game.resolve(self.state, {seat: action}, random.Random(0))
        return events

    def test_deterministic_setup_and_privacy(self):
        self.assertEqual(self.state,self.game.reset({},17))
        json.dumps(self.state)
        other = deepcopy(self.state)
        other['players'][1]['hand'].reverse()
        other['players'][0]['deck'].reverse()
        self.assertEqual(self.game.observe(self.state,0),self.game.observe(other,0))
        self.assertEqual(self.game.action_spec(self.state,0),self.game.action_spec(other,0))

    def test_invalid_does_not_mutate(self):
        before = deepcopy(self.state)
        with self.assertRaises(ValueError):
            self.act({'type':'buy','card':'Province'})
        self.assertEqual(before,self.state)

    def test_all_kingdom_cards_resolve(self):
        for card in KINGDOM:
            self.state = self.game.reset({},17)
            self.move_to_hand(card)
            self.act({'type':'play','card':card})
            self.game.check_invariants(self.state)
            json.dumps(self.state)

    def test_moat_and_discard(self):
        self.move_to_hand('Militia')
        self.move_to_hand('Moat',1)
        self.act({'type':'play','card':'Militia'})
        self.assertEqual(self.state['phase'],'reaction')
        count = len(self.state['players'][1]['hand'])
        self.act({'type':'reveal_moat'})
        self.assertEqual(self.state['phase'],'action')
        self.assertEqual(len(self.state['players'][1]['hand']),count)
        self.state['actions'] = 1
        self.move_to_hand('Militia')
        self.act({'type':'play','card':'Militia'})
        self.act({'type':'allow'})
        count = len(self.state['players'][1]['hand']) - 3
        self.act({'indices':list(range(count))})
        self.assertEqual(len(self.state['players'][1]['hand']),3)

    def test_mine_nested_gain_to_hand(self):
        self.move_to_hand('Mine')
        self.move_to_hand('Copper')
        self.act({'type':'play','card':'Mine'})
        self.act({'type':'trash','card':'Copper'})
        self.act({'type':'gain','card':'Silver'})
        self.assertIn('Silver',self.state['players'][0]['hand'])
        self.assertIn('Copper',self.state['trash'])

    def test_merchant_stacks_first_silver_only(self):
        for card in ('Merchant','Merchant','Silver','Silver'):
            self.move_to_hand(card)
        self.act({'type':'play','card':'Merchant'})
        self.act({'type':'play','card':'Merchant'})
        self.act({'type':'end_actions'})
        self.act({'type':'treasure','card':'Silver'})
        self.assertEqual(self.state['coins'],4)
        self.act({'type':'treasure','card':'Silver'})
        self.assertEqual(self.state['coins'],6)

    def test_random_play_replay_conservation_and_cap(self):
        for seed in range(10):
            s = self.game.reset({'max_turns':30},seed)
            r = random.Random(seed)
            for _ in range(5000):
                if self.game.outcome(s):
                    break
                seat = self.game.pending_decision(s)['seats'][0]
                spec = self.game.action_spec(s,seat)
                if spec['kind']=='choice':
                    a = r.choice(spec['choices'])
                else:
                    a = {'indices':r.sample(spec['options'],r.randint(spec['min'],spec['max']))}
                before = deepcopy(s)
                s, events = self.game.resolve(s,{seat:a},r)
                replay, replay_events = self.game.resolve(before,{seat:a},random.Random(999))
                self.assertEqual((s,events),(replay,replay_events))
                self.game.check_invariants(s)
            self.assertIsNotNone(self.game.outcome(s))

    def test_end_tiebreak_and_cap(self):
        self.state['supply']['Province'] = 0
        self.state['trash'] += ['Province']*8
        self.act({'type':'end_actions'})
        self.act({'type':'end_turn'})
        self.assertEqual(self.game.outcome(self.state)['winner'],1)
        self.state = self.game.reset({'max_turns':1},1)
        self.act({'type':'end_actions'})
        self.act({'type':'end_turn'})
        self.assertFalse(self.game.outcome(self.state)['complete'])


if __name__ == '__main__':
    unittest.main()
