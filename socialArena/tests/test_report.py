import json
import re
import tempfile
import unittest
from pathlib import Path
from social_arena.report import write_report


class ReportTests(unittest.TestCase):
    def test_manifest_ledger_filtered_replay_and_probe_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            directory=root/'games'/'g1'
            directory.mkdir(parents=True)
            injected='</script><script>alert(1)</script>'
            commit=dict(decision_id='1',attempts=[dict(seat=0,observation={'secret':injected},spec={'kind':'speech'},action={'text':injected},result={'text':injected})],events=[dict(type='speech',text=injected,visibility='public'),dict(type='secret',visibility=[0],value='private')],probes=[dict(seat=0,role='Servant',phase='vote',brier=.25,request_id='probe-1',result={'action':{'probabilities':{'1':.5}}})],actions={'0':{'text':injected}},rng_after='hidden')
            (directory/'events.jsonl').write_text(json.dumps(commit)+'\n')
            (directory/'summary.json').write_text('{}')
            (root/'ledger.json').write_text(json.dumps({'request-1':dict(game_id='g1',policy='p1',kind='probe',reserved_usd=.1,status='uncertain',error='TimeoutError')}))
            manifest=dict(experiment_id='experiment',engine='avalon-v1',protocol={'speech_rounds':2},policies={'friendly':{'id':'p1','provider':'local','prompt_hash':'hash','api_key':'must-not-render'}})
            records=[dict(game_id='g1',scenario_id='s1',pair_id='p1',game='avalon',status='complete',self_play=True,policies=['p1']*5,good_policy='p1',evil_policy='p1',outcome={'winner':'good'},costs_usd=.1,decisions=1,invalid_actions=0)]
            report=write_report(root,records,manifest,bootstrap=0).read_text()
            self.assertIn('avalon-v1',report)
            self.assertIn('friendly',report)
            self.assertNotIn('must-not-render',report)
            self.assertIn('TimeoutError',report)
            self.assertIn('Servant',report)
            self.assertIn('0.25',report)
            self.assertIn('games/g1/replay.html',report)
            replay=(directory/'replay.html').read_text()
            self.assertNotIn(injected,replay)
            payload=re.search(r'<script id="data" type="application/json">(.*?)</script>',replay,re.S).group(1)
            views=json.loads(payload)
            self.assertEqual(len(views['public'][0]['events']),1)
            self.assertEqual(views['public'][0]['attempts'],[])
            self.assertEqual(len(views['seat-0'][0]['events']),2)
            self.assertEqual(views['seat-1'][0]['attempts'],[])
            self.assertEqual(views['omniscient'][0]['rng_after'],'hidden')
            self.assertEqual(views['seat-0'][0]['attempts'][0]['observation']['secret'],injected)

    def test_empty_report_has_no_invented_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            report=write_report(temp,[],{},bootstrap=0).read_text()
            self.assertIn('Games: 0',report)
            self.assertIn('No scored probes recorded.',report)
            self.assertIn('unavailable',report)

if __name__=='__main__':
    unittest.main()
