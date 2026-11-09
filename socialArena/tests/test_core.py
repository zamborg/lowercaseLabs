import concurrent.futures
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from social_arena.avalon import Avalon, EVIL
from social_arena.budget import Ledger, BudgetExceeded
from social_arena.common import digest, read_json, read_stream
from social_arena.policies import definition
from social_arena.runner import Experiment, replay, export_view, freeze, composition_id


class AvalonTests(unittest.TestCase):
    def setUp(self):
        self.g = Avalon()
        self.s = self.g.reset({'roles':['Merlin','Servant','Servant','Assassin','Minion'], 'discussion_rounds':0}, 0)

    def resolve(self, s, actions):
        return self.g.resolve(s, actions, random.Random(1))[0]

    def test_roles_and_inaccessible_secrets(self):
        self.assertEqual(self.g.observe(self.s,0)['known_evil'],[3,4])
        self.assertEqual(self.g.observe(self.s,1)['known_evil'],[])
        changed = copy.deepcopy(self.s); changed['roles'][0],changed['roles'][2] = changed['roles'][2],changed['roles'][0]
        self.assertEqual(self.g.observe(self.s,1),self.g.observe(changed,1))
        self.assertEqual(self.g.action_spec(self.s,1),self.g.action_spec(changed,1))

    def test_ties_leader_five_rejections(self):
        s=self.s
        for i in range(5):
            leader=s['leader']; s=self.resolve(s,{leader:{'team':[0,1]}})
            s=self.resolve(s,{j:{'approve':j<2} for j in range(5)})
            self.assertEqual(s['leader'],(leader+1)%5)
        self.assertEqual(self.g.outcome(s)['winner'],'evil')
        self.assertEqual(self.g.outcome(s)['reason'],'rejections')

    def test_quest_privacy_and_all_terminal_paths(self):
        for fail, assassin_target, winner in [(True,1,'evil'),(False,0,'evil'),(False,1,'good')]:
            s=copy.deepcopy(self.s)
            for q in range(3):
                n=[2,3,2][q]; team=[3,0,1][:n] if fail else [0,1,2][:n]
                s=self.resolve(s,{s['leader']:{'team':team}})
                s=self.resolve(s,{i:{'approve':True} for i in range(5)})
                actions={i:{'success':not(fail and i==3)} for i in team}
                s,events=self.g.resolve(s,actions,random.Random(1))
                self.assertNotIn('actions',events[0]); self.assertNotIn('roles',events[0])
                self.assertEqual(events[0]['failures'],int(fail))
            if not fail:
                self.assertEqual(s['phase'],'assassinate')
                self.assertNotIn({'target':4},self.g.action_spec(s,3)['choices'])
                s=self.resolve(s,{3:{'target':assassin_target}})
            self.assertEqual(self.g.outcome(s)['winner'],winner)

    def test_protocol_rounds_and_invalid_atomic(self):
        s=self.g.reset({},2); before=copy.deepcopy(s)
        with self.assertRaises(ValueError): self.resolve(s,{s['leader']:{'team':[0,0]}})
        self.assertEqual(s,before)
        s=self.resolve(s,{s['leader']:{'team':[0,1]}})
        speaking=[]
        for _ in range(10):
            seat=self.g.pending_decision(s)['seats'][0]; speaking.append(seat)
            s=self.resolve(s,{seat:{'text':'test'}})
        self.assertEqual(speaking,[2,3,4,0,1]*2)
        self.assertEqual(s['phase'],'vote')


class CoreTests(unittest.TestCase):
    def config(self):
        return dict(scenarios=1, seed=14, policies={'a':{'kind':'scripted'},'b':{'kind':'random'}}, probes=True)

    def test_crash_resume_and_exact_replay(self):
        with tempfile.TemporaryDirectory() as d:
            a=Experiment(Path(d)/'a',self.config()); item=a.manifest['schedule'][0]
            with self.assertRaisesRegex(RuntimeError,'injected'):
                a.run_game(item,crash_after=4)
            resumed=Experiment(Path(d)/'a').run_game(item)
            b=Experiment(Path(d)/'b',self.config()); straight=b.run_game(item)
            self.assertEqual(resumed,straight)
            self.assertEqual(replay(Path(d)/'a'/'games'/item['game_id'])['state_hash'],straight['final_state_hash'])
            self.assertEqual(read_stream(Path(d)/'a'/'games'/item['game_id']/'events.jsonl'),read_stream(Path(d)/'b'/'games'/item['game_id']/'events.jsonl'))
            public=export_view(Path(d)/'a'/'games'/item['game_id'])
            self.assertTrue(all(not c['attempts'] and not c['probes'] for c in public))
            for c in public:
                self.assertTrue(all(e['visibility']=='public' for e in c['events']))

    def test_snapshot_frozen_and_probe_detached(self):
        from social_arena import runner
        views=[]
        def act(obs,spec,policy,rid):
            views.append(copy.deepcopy(obs))
            obs['public'].append({'injected':'should not propagate'})
            from social_arena.policies import local_act
            return local_act(obs,spec,policy,rid)
        with tempfile.TemporaryDirectory() as d, patch.object(runner,'local_act',side_effect=act):
            Experiment(d,self.config()).run()
        self.assertTrue(all(not any('injected' in x for x in o['public']) for o in views))

    def test_budget_concurrent_reservations_and_uncertain(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger.json',1,True)
            def reserve(i):
                try: ledger.reserve(str(i),.3); return True
                except BudgetExceeded: return False
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(sum(pool.map(reserve,range(10))),3)
            ids=sorted(ledger.entries())
            ledger.settle(ids[0],error='timeout')
            self.assertEqual(ledger.entries()[ids[0]]['status'],'uncertain')
            with self.assertRaises(BudgetExceeded): ledger.reserve(ids[0],.3)
            ledger.settle(ids[1],.1)
            ledger.reserve('next',.3)

    def test_identity_price_and_default_no_network(self):
        self.assertNotEqual(definition({'kind':'scripted'})['id'],definition({'kind':'scripted','prompt':'different'})['id'])
        for bad in (float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError): definition({'input_usd_per_million':bad})
            with self.assertRaises(ValueError): freeze({'max_usd':bad})
        with tempfile.TemporaryDirectory() as d:
            config=dict(scenarios=1,policies={'p':{'kind':'openai','model':'not-priced'}})
            with patch('social_arena.runner.openai_call',side_effect=AssertionError('network forbidden')):
                records=Experiment(d,config).run()
            self.assertTrue(all(r['status']=='incomplete' for r in records))

    def test_infrastructure_failure_missing(self):
        policy={'kind':'openai','model':'test','input_usd_per_million':1,'output_usd_per_million':1,'price_source':'fixture'}
        with tempfile.TemporaryDirectory() as d, patch('social_arena.runner.openai_call',side_effect=TimeoutError):
            records=Experiment(d,dict(scenarios=1,policies={'p':policy},allow_paid=True,max_usd=1)).run()
            self.assertTrue(all(r['status']=='incomplete' for r in records))
            self.assertTrue(all(r['outcome'] is None for r in records))
            self.assertGreater(sum(r['costs_usd'] for r in records),0)

    def test_provider_json_mode_input_and_http_audit(self):
        from social_arena.policies import request_payload
        payload=request_payload({'seat':0},{'kind':'choice','choices':[{'a':True}]},definition({'kind':'openai','model':'fixture'}))
        self.assertIn('JSON',payload['input'])
        self.assertEqual(payload['text']['format']['type'],'json_schema')
        self.assertTrue(payload['text']['format']['strict'])
        self.assertEqual(payload['service_tier'],'default')
        self.assertFalse(payload['store'])

    def test_protocol_pool_rejects_mixing(self):
        schedule=[dict(game_id='a',scenario_id='a',pair_id='a',seed=1,config={'discussion_rounds':0},policies=['p']*5),dict(game_id='b',scenario_id='b',pair_id='b',seed=2,config={'discussion_rounds':2},policies=['p']*5)]
        with self.assertRaisesRegex(ValueError,'protocol'):
            freeze(dict(policies={'p':{}},schedule=schedule))

    def test_composition_ignores_opponent(self):
        roles=['Merlin','Servant','Servant','Assassin','Minion']
        self.assertEqual(composition_id(['a','b','b','x','x'],roles,False),composition_id(['a','b','b','y','z'],roles,False))

if __name__=='__main__': unittest.main()
