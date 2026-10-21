"""Synthetic navigation contracts, not simulated robot/teleop evidence."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from punktnull.inspection import LioOccupancyGrid
from punktnull.mission_skills import CrawlInspectionExplorer,CrawlMissionMixin,entry_errors,observed_exit
from navigation import GridConfig


class MissionSkillGates(unittest.TestCase):
    def setUp(self):
        self.grid=LioOccupancyGrid(GridConfig());self.grid.last_update=1.
        shape=self.grid.layers()['safe'].shape
        self.grid.layers=lambda:{'safe':np.ones(shape,bool),'clearance':np.ones(shape)*2}
        self.e=CrawlInspectionExplorer(self.grid,[0.,0.])

    def room(self,index):
        xy=self.e.inspection_plan['rooms'][index]['observation_xy_m'];self.e.last_position=None
        self.assertEqual(self.e.step(xy,0,1.,pose_time=1.).status,'SCAN_REQUIRED')
        self.e.finish_scan()

    def reach_ingress(self):
        self.e.finish_scan();self.room(0);self.room(1)
        self.e.last_position=None;gate=self.e.pending_skill
        self.assertEqual(self.e.step(gate['approach_xy_m'],np.pi/2,1.,pose_time=1.).status,'CRAWL_REQUIRED')
        return gate

    def test_room_scan_cannot_replace_physical_skill(self):
        gate=self.reach_ingress()
        self.assertEqual([r['id'] for r in self.e.completed_rooms],['west','east'])
        with self.assertRaises(RuntimeError):self.e.finish_scan()
        with self.assertRaises(RuntimeError):self.e.finish_skill([0,3.75],1.,{'gate_id':gate['id'],'observed_exit':False})
        self.assertIsNotNone(self.e.pending_skill)

    def test_two_traversals_required_and_start_is_preserved(self):
        gate=self.reach_ingress()
        self.e.finish_skill([0,3.75],1.,{'id':gate['id'],'gate_id':gate['id'],'observed_exit':True})
        self.room(2)
        self.assertEqual(self.e.step([0,3.75],0,1.).status,'EXPLORATION_EXHAUSTED')
        self.e.return_to_start();gate=self.e.pending_skill;self.e.last_position=None
        self.assertEqual(gate['direction'],'egress')
        self.assertEqual(self.e.step(gate['approach_xy_m'],0,1.).status,'CRAWL_REQUIRED')
        self.e.finish_skill([0,.65],1.,{'id':gate['id'],'gate_id':gate['id'],'observed_exit':True})
        self.e.last_position=None
        self.assertEqual(self.e.step([0,0],0,1.).status,'RETURNED')
        self.assertEqual(len(self.e.completed_skills),2);self.assertEqual(len(self.e.completed_rooms),3)

    def test_stale_map_or_wrong_gate_cannot_admit_resume(self):
        gate=self.reach_ingress()
        with self.assertRaises(RuntimeError):self.e.finish_skill([0,3.75],1.,{'gate_id':'wrong','observed_exit':True})
        with self.assertRaises(RuntimeError):self.e.finish_skill([0,3.75],3.,{'gate_id':gate['id'],'observed_exit':True})
        self.assertEqual(self.e.completed_skills,[])

    def test_operator_review_does_not_skip_ingress(self):
        self.e.inject_review_after_room='east';self.reach_ingress_for_review()
        self.assertEqual(self.e.step([2.,-.1],0,1.).status,'OPERATOR_REVIEW_REQUIRED')
        self.assertTrue(self.e.reanchor_from_handover([2.,-.1],1.))
        self.assertEqual(self.e.pending_skill['direction'],'ingress')
        np.testing.assert_allclose(self.e.target,self.e.pending_skill['approach_xy_m'])

    def reach_ingress_for_review(self):
        self.e.finish_scan();self.room(0);self.room(1)

    def test_entry_and_exit_are_actual_bounded_states(self):
        gate=self.e.skill_contract['ingress']
        state={'base':[*gate['approach_xy_m'],.8],'yaw_deg':gate['heading_deg']}
        self.assertTrue(entry_errors(state,gate)['accepted']);self.assertFalse(observed_exit(state,gate))
        state['base'][0]+=.5
        self.assertFalse(entry_errors(state,gate)['accepted'])
        state={'base':[0,3.75,.8],'yaw_deg':gate['heading_deg']}
        self.assertTrue(observed_exit(state,gate))

    def test_actual_patched_launcher_uses_selected_mission_backend(self):
        # Import the executable in a fresh process, as a real launch does.
        # Other tests import their own launcher and mutate the cached retained
        # module; importing a second wrapper there would double-patch it.
        import subprocess,sys
        from punktnull.retained import ROOT
        code='''import run_mission as m
mission=object.__new__(m.LioMission)
launcher=m._launch_namespace['launch']
assert 'make_explorer' in launcher.__code__.co_names
assert 'Explorer' not in launcher.__code__.co_names
for selected,expected in ((True,m.CrawlInspectionExplorer),(False,m.InspectionExplorer)):
    mission.crawl_enabled=selected
    assert type(mission.make_explorer(m.LioOccupancyGrid(),[0.,0.])) is expected
'''
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_long_operator_interventions_restart_entry_check_and_restore_phase(self):
        gate=self.e.skill_contract['ingress'];clock=[10.]
        class Harness(CrawlMissionMixin):
            def __init__(self):
                self.commands=['takeover'];self.calls=0;self.ticks=0;self.events=[]
                self.command_stabilizer=SimpleNamespace(reset=lambda:None)
                self.client=SimpleNamespace(hold=lambda yaw:None)
                self.state=self.truth_state={'base':[*gate['approach_xy_m'],.8],'yaw_deg':gate['heading_deg']}
            def transition(self,phase,*args):self.phase=phase
            def takeover(self):self.calls+=1;clock[0]+=45.;self.phase='RETURN_CHECK'
            def tick(self):
                clock[0]+=.2;self.ticks+=1
                if self.ticks==2 and self.phase=='SKILL_APPROACH':self.commands.append('takeover')
            def event(self,event,**kw):self.events.append(event)
        harness=Harness()
        with patch('punktnull.mission_skills.time.monotonic',side_effect=lambda:clock[0]),patch('punktnull.skills.session.measured_error',return_value=None):
            harness._align_skill_entry(gate)
        self.assertEqual(harness.calls,2)
        self.assertEqual(harness.phase,'SKILL_APPROACH')
        self.assertIn('skill_entry_admitted',harness.events)


class StableFootprintEpisode(unittest.TestCase):
    def test_checked_new_episode_keeps_history_and_never_fills_between_endpoints(self):
        grid=LioOccupancyGrid(GridConfig())
        self.assertTrue(grid.record_stable_pose([.08,.45],1.,stable=True,support_free=True))
        original=grid.proprioceptive.copy();seen=grid.seen.copy();odds=grid.logodds.copy()
        self.assertFalse(grid.record_stable_pose([0.,3.4],5.,stable=True,support_free=True))
        proof=dict(accepted=True,inside_exit_region=True,minimum_nonfloor_aabb_clearance_m=.1)
        evidence=grid.begin_stable_episode([0.,3.4],5.,gate_id='ingress',exit_geometry=proof,stable_seconds=.71,support_free=True)
        self.assertFalse(evidence['interpolated_path']);self.assertEqual(len(grid.proprioceptive_poses),2)
        self.assertTrue(np.all(grid.proprioceptive[original]))
        mid=grid.world_to_cell([0.,1.8]);self.assertFalse(grid.proprioceptive[mid[1],mid[0]])
        np.testing.assert_array_equal(grid.seen,seen);np.testing.assert_array_equal(grid.logodds,odds)
        self.assertTrue(grid.record_stable_pose([.01,3.4],5.1,stable=True,support_free=True))
        with self.assertRaises(ValueError):grid.begin_stable_episode([0,3.4],6.,gate_id='ingress',exit_geometry=proof,stable_seconds=.71,support_free=True)
        self.assertEqual(len(grid.proprioceptive_evidence()['separate_stable_episodes']),1)

    def test_missing_exit_or_unstable_state_cannot_create_footprint_episode(self):
        grid=LioOccupancyGrid(GridConfig());proof=dict(accepted=True,inside_exit_region=True,minimum_nonfloor_aabb_clearance_m=.1)
        for changes in ({'stable_seconds':.69},{'support_free':False},{'exit_geometry':{}},{'exit_geometry':dict(proof,minimum_nonfloor_aabb_clearance_m=.01)}):
            args=dict(gate_id='ingress',exit_geometry=proof,stable_seconds=.7,support_free=True);args.update(changes)
            with self.assertRaises(ValueError):grid.begin_stable_episode([0,3.4],5.,**args)
        self.assertFalse(grid.proprioceptive.any());self.assertEqual(grid.proprioceptive_poses,[])


if __name__=='__main__':unittest.main()
