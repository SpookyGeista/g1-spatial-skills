"""CPU fixtures for v2 actual-state acceptance; never simulation validation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from punktnull.skills.legacy import WORKSPACE
import numpy as np
from punktnull.skills.limits import load_joint_limits,POSITION_TOLERANCE_RAD
from punktnull.skills.guard import GroundGuard
from punktnull.skills.session import SkillSession,REGISTRY,measured_error,ClearanceEvidence
from punktnull.skills.reference import ReferenceClip,ReferencePoseStream
from punktnull.skills.evaluate import evaluate

class ActualLimitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=load_joint_limits();cls.clip=ReferenceClip.load(REGISTRY['crawl'].source)

    def state(self,now=10):
        return dict(monotonic=now,support=False,base=[0,0,.78,1,0,0,0,0,0,0,0,0,0],q=[0.]*29,dq=[0.]*29,tau=[0.]*29,stable_full_rate_sim_seconds=1.)

    def test_actual_config_and_wrist_limit_are_not_relaxed(self):
        l=self.limits
        self.assertEqual(l.effort[l.names.index('right_wrist_pitch_joint')],5.)
        self.assertEqual(l.effort[l.names.index('left_wrist_yaw_joint')],5.)
        self.assertEqual(l.effort[l.names.index('right_hip_pitch_joint')],139.)
        self.assertEqual(l.urdf_effort[l.names.index('right_hip_pitch_joint')],88.)
        self.assertEqual(l.velocity[l.names.index('left_hip_pitch_joint')],20.)
        self.assertEqual(l.velocity[l.names.index('right_ankle_pitch_joint')],35.)

    def test_actual_ankle_overshoot_rejected_despite_small_reference_error(self):
        q=np.zeros(29);i=self.limits.names.index('right_ankle_pitch_joint')
        q[i]=self.limits.lower[i]-.003
        self.assertEqual(self.limits.check(q,np.zeros(29),np.zeros(29)),'joint_position_limit')
        q[i]=self.limits.lower[i]-.5*POSITION_TOLERANCE_RAD
        self.assertIsNone(self.limits.check(q,np.zeros(29),np.zeros(29)))

    def test_per_joint_speed_and_applied_effort(self):
        q=np.zeros(29);dq=q.copy();tau=q.copy();dq[0]=20.001
        self.assertEqual(self.limits.check(q,dq,tau),'joint_velocity_limit')
        dq[0]=0;tau[20]=5.001
        self.assertEqual(self.limits.check(q,dq,tau),'joint_effort_limit')
        tau[20]=5.
        self.assertIsNone(self.limits.check(q,dq,tau))

    def test_missing_malformed_nonfinite_effort_fail_closed(self):
        q=np.zeros(29)
        self.assertEqual(self.limits.check(q,q),'missing_applied_effort')
        self.assertEqual(self.limits.check(q,q,np.zeros(28)),'invalid_applied_effort')
        tau=q.copy();tau[0]=np.nan
        self.assertEqual(self.limits.check(q,q,tau),'invalid_applied_effort')
        self.assertEqual(self.limits.check(q[:28],q,tau),'invalid_joint_state_shape')

    def test_session_requires_fresh_actual_state_and_effort(self):
        s=self.state();self.assertIsNone(measured_error(s,10,upright=True))
        self.assertEqual(measured_error(s,10.2),'stale_measured_state')
        del s['tau'];self.assertEqual(measured_error(s,10),'missing_applied_effort')
        s['tau']=[0.]*29;s['q'][10]=-.875
        self.assertEqual(measured_error(s,10),'joint_position_limit')

    def guard_fixture(self,folder,orientation=0):
        g=GroundGuard(self.clip.source,Path(folder)/'controller')
        g.reference.errors=lambda *args:(2.,orientation,3.)
        g.reference.permission=np.ones(self.clip.frames,bool)
        g.feedback.poll=lambda now:dict(monotonic=now,motion='streamed',encoder_mode=0,playing=True,index=1)
        def assess(now,**changes):
            auth=dict(owner='skill',phase='GROUND',source_digest=self.clip.digest,source=str(self.clip.source),owner_token='one-owner',
                monotonic=now,clearance_ok=True,submitted_frame=500,heading_alignment_rad=0)
            auth.update(changes);(Path(folder)/'skill_stream.json').write_text(json.dumps(auth))
            return g.assess(now,[0,0,.3],[2**-.5,2**-.5,0,0],[0.]*29,[0.]*29,efforts=[0.]*29,state_monotonic=now)
        return g,assess

    def test_large_height_joint_reference_error_remains_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:
            g,assess=self.guard_fixture(d)
            self.assertIsNone(assess(10));self.assertIsNone(assess(10.2))
            self.assertEqual(g.last[2:],[2.,0,3.])

    def test_orientation_phase_guard_still_aborts(self):
        with tempfile.TemporaryDirectory() as d:
            _,assess=self.guard_fixture(d,orientation=46)
            self.assertIsNone(assess(10))
            self.assertEqual(assess(10.2),'sustained_stream_reference_orientation_error')

    def test_changed_owner_and_unverified_clearance_reject(self):
        with tempfile.TemporaryDirectory() as d:
            _,assess=self.guard_fixture(d)
            self.assertIsNone(assess(10))
            self.assertEqual(assess(10.2,owner_token='other'),'skill_owner_changed_mid_execution')
        with tempfile.TemporaryDirectory() as d:
            _,assess=self.guard_fixture(d)
            self.assertEqual(assess(10,clearance_ok=False),'skill_clearance_not_confirmed')

    def test_owner_retirement_needs_bound_release_and_fresh_native_planner_ack(self):
        with tempfile.TemporaryDirectory() as d:
            g=GroundGuard(self.clip.source,Path(d)/'controller');s=self.state()
            native=dict(motion='streamed',encoder_mode=0,playing=True,index=1)
            g.feedback.poll=lambda now:dict(native,monotonic=now)
            def assess(now,**changes):
                a=dict(owner='skill',owner_token='first',phase='RETURN_READY',monotonic=now,
                    source=str(self.clip.source),source_digest=self.clip.digest,clearance_ok=True,
                    submitted_frame=0,heading_alignment_rad=0.)
                a.update(changes);(Path(d)/'skill_stream.json').write_text(json.dumps(a))
                return g.assess(now,s['base'][:3],s['base'][3:7],s['q'],s['dq'],efforts=s['tau'],state_monotonic=now)
            g.reference.errors=lambda *args:(0.,0.,0.)
            self.assertIsNone(assess(10))
            self.assertIsNone(assess(10.1,owner='none',owner_token=None,phase='COMPLETE',released_owner_token='first'))
            self.assertEqual(g._token,'first')
            self.assertEqual(assess(10.12),'retired_skill_owner_token')
            self.assertEqual(assess(10.14,owner_token='second',phase='PRIMING'),'skill_owner_changed_mid_execution')
            native['motion']='planner_motion'
            self.assertIsNone(assess(10.2,owner='none',owner_token=None,phase='COMPLETE',released_owner_token='first'))
            self.assertIsNone(g._token)
            self.assertEqual(assess(10.22),'retired_skill_owner_token')
            self.assertIsNone(assess(10.24,owner_token='second',phase='PRIMING'))
            self.assertEqual(g._token,'second')
        with tempfile.TemporaryDirectory() as d:
            g,assess=self.guard_fixture(d)
            self.assertIsNone(assess(10,phase='RETURN_READY'))
            self.assertEqual(assess(10.1,owner='none',owner_token=None,phase='COMPLETE',released_owner_token='wrong'),'ground_without_skill_owner')
            self.assertEqual(assess(10.2,owner_token='second',phase='PRIMING'),'skill_owner_changed_mid_execution')

    def test_supported_initialization_is_tiny_bounded_and_cannot_reopen(self):
        waiting=dict(mode='waiting',accepted_count=0,rejected_count=0)
        active=dict(mode='active',accepted_count=1,rejected_count=0)
        with tempfile.TemporaryDirectory() as d:
            g=GroundGuard(self.clip.source,Path(d)/'controller');s=self.state();s['q'][11]=.261802
            def assess(t,status=waiting,support=True):
                return g.assess(t,s['base'][:3],s['base'][3:7],s['q'],s['dq'],support=support,
                    efforts=s['tau'],state_monotonic=t,command_status=status)
            self.assertIsNone(assess(10));self.assertEqual(g.startup_position_diagnostic,'joint_position_limit')
            self.assertIsNone(assess(11,active))
            s['tau'][20]=5.01
            self.assertEqual(assess(12,active),'joint_effort_limit');s['tau'][20]=0
            s['q'][11]=.261901
            self.assertEqual(assess(13,active),'joint_position_limit');s['q'][11]=.261802
            self.assertEqual(assess(130.01,active),'joint_position_limit')
            self.assertEqual(assess(131,active,support=False),'joint_position_limit')
            self.assertEqual(assess(132,active),'joint_position_limit')
        with tempfile.TemporaryDirectory() as d:
            g=GroundGuard(self.clip.source,Path(d)/'controller')
            self.assertEqual(g.assess(10,s['base'][:3],s['base'][3:7],s['q'],s['dq'],support=True,
                efforts=s['tau'],state_monotonic=10),'joint_position_limit')
            self.assertIsNone(g.assess(11,s['base'][:3],s['base'][3:7],s['q'],s['dq'],support=True,
                efforts=s['tau'],state_monotonic=11,command_status=active))
            self.assertEqual(g.assess(12,s['base'][:3],s['base'][3:7],s['q'],s['dq'],support=False,
                efforts=s['tau'],state_monotonic=12,command_status=active),'joint_position_limit')
            self.assertEqual(g.assess(13,s['base'][:3],s['base'][3:7],s['q'],s['dq'],support=True,
                efforts=s['tau'],state_monotonic=13,command_status=active),'joint_position_limit')

    def test_guard_stale_measurement_not_refreshed_by_call_time(self):
        with tempfile.TemporaryDirectory() as d:
            g=GroundGuard(self.clip.source,Path(d)/'controller')
            s=self.state()
            self.assertEqual(g.assess(10.3,s['base'][:3],s['base'][3:7],s['q'],s['dq'],efforts=s['tau'],state_monotonic=10),'stale_or_missing_measured_state')

    def test_final_return_uses_actual_upright_not_reference_joint_match(self):
        session=SkillSession(self.clip,experimental=True);s=self.state()
        token=session.acquire('autonomy',s,ClearanceEvidence(10,'disclosed_simulation_fixture',True,True,.1,'test'),10)
        session.phase='EXITING';session.native_ack_time=10.;session.ground_observed=True;session.ground_displacement=.6
        for t in [11,11.8]:
            s=self.state(t)
            result=session.update(token,s,ClearanceEvidence(t,'disclosed_simulation_fixture',True,True,.1,'test'),t,
                native_stream_active=True,submitted_frame=1023,inferred_frame=1023,tracking_error=[2.,0,3.],finished_sending=True)
        self.assertEqual(session.phase,'RETURN_READY');self.assertFalse(result['return_owner'])
        with self.assertRaises(RuntimeError):session.release(token,fresh_map_safe=True,explicit_release=False)

    def test_no_final_return_without_full_rate_stability(self):
        session=SkillSession(self.clip,experimental=True);s=self.state();token=session.acquire('autonomy',s,ClearanceEvidence(10,'disclosed_simulation_fixture',True,True,.1,'test'),10)
        session.phase='EXITING';session.native_ack_time=10.;session.ground_observed=True;session.ground_displacement=.6
        for t in [11,11.8]:
            s=self.state(t);s['stable_full_rate_sim_seconds']=.1
            session.update(token,s,ClearanceEvidence(t,'disclosed_simulation_fixture',True,True,.1,'test'),t,
                native_stream_active=True,submitted_frame=1023,inferred_frame=1023,tracking_error=[0,0,0],finished_sending=True)
        self.assertEqual(session.phase,'SETTLING')

    def test_generated_crawl_requires_explicit_opt_in_and_complete_root_sources(self):
        # Constructor provenance fixture only; the real loader/exporter is separately tested.
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'evidence') as d:
            source=Path(d);derived=replace(self.clip,source=source)
            metadata=dict(kind='generated_g1_skill_reference',skill='crawl',physical_execution=False,
                sources_sha256={str(self.clip.source/name):sha for name,sha in self.clip.hashes.items()})
            (source/'generated_reference.json').write_text(json.dumps(metadata))
            def load(path):return self.clip if Path(path)==self.clip.source else derived
            with patch('punktnull.skills.session.ReferenceClip.load',side_effect=load):
                with self.assertRaisesRegex(ValueError,'differs'):SkillSession(derived,experimental=True)
                self.assertIsNotNone(SkillSession(derived,experimental=True,allow_generated=True).generated_reference)
                metadata['sources_sha256'].pop(next(iter(metadata['sources_sha256'])))
                (source/'generated_reference.json').write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError,'complete registered source'):SkillSession(derived,experimental=True,allow_generated=True)

    def test_retrospective_contract_cannot_accept_missing_actual_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('punktnull.skills.evaluate._evaluate_v1',return_value={'accepted':False,'checks':{'bounded_inferred_phase_tracking':False},'limitations':[]}):
                result=evaluate(d)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['acceptance_contract']['version'],2)
            self.assertNotIn('bounded_inferred_phase_tracking',result['checks'])
            self.assertIn('complete_actual_limits_evidence',result['failed_checks'])

if __name__=='__main__':unittest.main()
