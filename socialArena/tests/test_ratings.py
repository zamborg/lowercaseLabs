import unittest
from social_arena.ratings import analyze


def fixture(game='avalon',count=12,mode='strong'):
    records=[]
    for i in range(count):
        for swap in range(2):
            a,b=('a','b') if swap==0 else ('b','a')
            winner = ('good' if a=='a' else 'evil') if mode=='strong' else 'good'
            if game=='dominion':
                winner = None if mode=='draw' else 0 if a=='a' else 1
            records.append(dict(game_id=f'{i}-{swap}',game=game,scenario_id=str(i),pair_id=str(i),status='complete',good_policy=a,evil_policy=b,first_policy=a,second_policy=b,outcome={'winner':winner}))
    return records


class RatingsTests(unittest.TestCase):
    def test_ordering_and_reordering(self):
        rows=fixture()
        a=analyze(rows,bootstrap=12)
        self.assertEqual(a,analyze(list(reversed(rows)),bootstrap=12))
        pool=a['pools']['avalon']
        self.assertGreater(pool['ratings']['a']['rating'],pool['ratings']['b']['rating'])
        self.assertAlmostEqual(pool['side_advantage_elo'],0,places=6)
        self.assertEqual(pool['independent_scenarios'],12)

    def test_equal_policies_and_side_bias(self):
        p=analyze(fixture(mode='side'),bootstrap=0)['pools']['avalon']
        self.assertAlmostEqual(p['ratings']['a']['rating'],1500,places=6)
        self.assertGreater(p['side_advantage_elo'],0)

    def test_dominion_draws(self):
        p=analyze(fixture(game='dominion',mode='draw'),bootstrap=3)['pools']['dominion']
        self.assertAlmostEqual(p['ratings']['a']['rating'],1500)
        self.assertGreater(p['tie_propensity'],1)
        self.assertEqual(p['ratings']['a']['raw']['first_or_good']['draws'],12)
        self.assertAlmostEqual(p['matchups'][0]['side_averaged_expected_score'],.5)

    def test_exclude_partial_pair(self):
        rows=fixture(count=1)
        rows[0]['status']='policy_failure'
        p=analyze(rows,bootstrap=0)
        self.assertEqual(p['eligible_games'],0)
        self.assertEqual(len(p['excluded_games']),2)

    def test_failure_loss_sensitivity(self):
        rows=fixture(count=2)
        failed=dict(rows[0],game_id='failed',pair_id='failed',scenario_id='failed',status='policy_failure',failing_policy='a')
        rows.append(failed)
        p=analyze(rows,bootstrap=0)['pools']['avalon']
        self.assertEqual(p['failure_as_loss_sensitivity']['added_policy_failures'],1)
        self.assertLess(p['failure_as_loss_sensitivity']['ratings']['a'],p['ratings']['a']['rating'])

    def test_disconnected_warning(self):
        rows=fixture(count=1)
        extra=fixture(count=1)
        for r in extra:
            r.update(game_id='x'+r['game_id'],scenario_id='x',pair_id='x')
            r['good_policy']='c' if r['good_policy']=='a' else 'd'
            r['evil_policy']='c' if r['evil_policy']=='a' else 'd'
        p=analyze(rows+extra,bootstrap=0)['pools']['avalon']
        self.assertTrue(any('Disconnected' in warning for warning in p['warnings']))

    def test_selfplay(self):
        r=fixture(count=1)[0]
        r.update(good_policy='a',evil_policy='a',self_play=True)
        p=analyze([r],bootstrap=2)['pools']['avalon']
        self.assertEqual(p['ratings']['a']['rating'],1500)

if __name__=='__main__':
    unittest.main()
