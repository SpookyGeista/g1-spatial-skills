"""CPU component checks; synthetic states are not simulation acceptance."""
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from punktnull.skills.reference import ReferenceClip,ReferencePoseStream,validate_reference_pose,HEADER_SIZE,attach_reference_validation
from punktnull.skills.session import REGISTRY,SkillSession,ClearanceEvidence,measured_error
from punktnull.skills.legacy import ground_module,planner_module
from punktnull.skills.guard import GroundGuard


class SkillsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip=ReferenceClip.load(REGISTRY['crawl'].source)

    def state(self,frame=0,now=10):
        g=ground_module();c=self.clip
        return {'monotonic':now,'support':False,'base':[*c.root[frame],*c.quaternion[frame],0,0,0,0,0,0],
                'q':c.data['joint_pos'][frame,g.MJ_TO_IL].tolist(),'dq':[0.]*29,'tau':[0.]*29,'stable_full_rate_sim_seconds':1.}

    def clear(self,now=10):return ClearanceEvidence(now,'disclosed_simulation_fixture',True,True,.08,'test_fixture')

    def align(self,s,token):
        for t in [10,10.01,10.8]:
            s.update(token,self.state(0,t),self.clear(t),t,native_stream_active=True,submitted_frame=0)
        self.assertEqual(s.phase,'ENTERING')

    def test_actual_reference_integrity(self):
        self.assertEqual(self.clip.frames,1024)
        self.assertEqual(self.clip.digest,'7b068d903e6049d16a956b19b867bb4d59ca4775fe2c32dc3e23445113b27191')

    def test_joint_protocol_not_smpl(self):
        p=ReferencePoseStream(self.clip).packet(10)
        header=json.loads(p[4:4+HEADER_SIZE].split(b'\0',1)[0])
        self.assertEqual(header['v'],1)
        f,n,t=validate_reference_pose(p,now=10)
        self.assertEqual(f['joint_pos'].shape,(51,29));self.assertNotIn('smpl_pose',f)
        np.testing.assert_allclose(f['joint_pos'][0],self.clip.data['joint_pos'][0],atol=1e-7)

    def test_replay_stale_and_protocol_rejected(self):
        p=ReferencePoseStream(self.clip).packet(10)
        with self.assertRaises(ValueError):validate_reference_pose(p,now=10,last_frame=50)
        with self.assertRaises(ValueError):validate_reference_pose(p,now=11)
        with self.assertRaises(ValueError):validate_reference_pose(b'evil'+p[4:],now=10)
        raw=bytearray(p);raw[4+HEADER_SIZE-1]=1
        with self.assertRaises(ValueError):validate_reference_pose(bytes(raw),now=10)

    def test_no_source_frame_skips_and_final_hold(self):
        s=ReferencePoseStream(self.clip);last=None
        for i in range(self.clip.frames+2):
            p=s.packet(10+i*.02,advance=True);_,last,_=validate_reference_pose(p,now=10+i*.02,last_frame=last)
        self.assertTrue(s.finished_sending);self.assertEqual(s.source_frame,1023)
        f,_,_=validate_reference_pose(s.packet(10+(self.clip.frames+2)*.02,advance=False),now=10+(self.clip.frames+2)*.02)
        np.testing.assert_allclose(f['joint_pos'][-1],self.clip.data['joint_pos'][-1],atol=1e-7)
        self.assertFalse(f['joint_vel'].any())

    def test_late_stream_fails_instead_of_catchup(self):
        s=ReferencePoseStream(self.clip);s.packet(10,advance=True)
        with self.assertRaises(TimeoutError):s.packet(10.2,advance=True)
        self.assertEqual(s.source_frame,1)

    def test_clearance_unknown_or_stale_rejects(self):
        for c in [ClearanceEvidence(10,'unidentified',True,True,.1,'a'),self.clear(9),
                  ClearanceEvidence(10,'measured_sensor_geometry',True,False,.1,'a')]:
            session=SkillSession(self.clip,experimental=True)
            with self.assertRaises(ValueError):session.acquire('autonomy',self.state(),c,10)

    def test_stairs_cannot_gain_executable_status(self):
        with self.assertRaises(ValueError):SkillSession(self.clip,skill='stairs',experimental=True)
        with self.assertRaises(ValueError):SkillSession(self.clip)

    def test_no_ownership_or_stale_state_means_pause_not_idle(self):
        s=SkillSession(self.clip,experimental=True);token=s.acquire('autonomy',self.state(),self.clear(),10)
        a=s.update('wrong',self.state(),self.clear(),10,native_stream_active=True,submitted_frame=0)
        self.assertEqual(a['action'],'PAUSE_SIM_AND_STOP_NATIVE');self.assertFalse(a['return_owner'])

    def test_sent_completion_without_ground_is_failure(self):
        s=SkillSession(self.clip,experimental=True);token=s.acquire('autonomy',self.state(),self.clear(),10)
        self.align(s,token)
        a=s.update(token,self.state(-1,11),self.clear(11),11,native_stream_active=True,
                   submitted_frame=1023,inferred_frame=1023,tracking_error=[0,0,0],finished_sending=True)
        self.assertEqual(a['action'],'PAUSE_SIM_AND_STOP_NATIVE')

    def test_rejoin_requires_actual_ground_travel_and_upright_window(self):
        s=SkillSession(self.clip,experimental=True);token=s.acquire('autonomy',self.state(),self.clear(),10)
        self.align(s,token)
        frames=np.flatnonzero(s.reference.ground & (s.reference.tilt>45) & (s.reference.xyz[:,2]<.45))
        for i in [frames[0],frames[-1]]:
            s.update(token,self.state(i,11),self.clear(11),11,native_stream_active=True,
                     submitted_frame=int(i),inferred_frame=int(i),tracking_error=[0,0,0])
        self.assertTrue(s.ground_observed);self.assertGreater(s.ground_displacement,.5)
        for t in [12,12.8]:
            s.update(token,self.state(1023,t),self.clear(t),t,native_stream_active=True,
                     submitted_frame=1023,inferred_frame=1023,tracking_error=[0,0,0],finished_sending=True)
        self.assertEqual(s.phase,'RETURN_READY')
        with self.assertRaises(RuntimeError):s.release(token,fresh_map_safe=False,explicit_release=True)
        self.assertEqual(s.release(token,fresh_map_safe=True,explicit_release=True),'autonomy')

    def test_ack_alone_does_not_advance_reference_before_alignment(self):
        s=SkillSession(self.clip,experimental=True);token=s.acquire('autonomy',self.state(),self.clear(),10)
        result=s.update(token,self.state(),self.clear(),10,native_stream_active=True,submitted_frame=0)
        self.assertEqual(result['action'],'PRIME_START_POSE');self.assertEqual(s.phase,'ALIGNING')
        result=s.update(token,self.state(0,10.02),self.clear(10.02),10.02,native_stream_active=True,submitted_frame=1)
        self.assertEqual(result['reason'],'reference_advanced_before_measured_alignment')

    def test_guard_missing_owner_never_authorizes_ground(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=GroundGuard(self.clip.source,Path(tmp)/'controller');i=500;s=self.state(i)
            self.assertEqual(g.assess(10,s['base'][:3],s['base'][3:7],s['q'],s['dq'],efforts=s['tau'],state_monotonic=10), 'ground_without_skill_owner')

    def test_native_ack_polled_while_startup_support_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'controller';folder.mkdir()
            for name,column,value in [('motion_name','motion_name','planner_motion'),('motion_playing','playing_0','1'),('encoder_mode','encoder_mode_0','0')]:
                (folder/(name+'.csv')).write_text('index,time_monotonic_ms,'+column+'\n0,10000,'+value+'\n')
            g=GroundGuard(self.clip.source,folder);s=self.state()
            self.assertIsNone(g.assess(10,s['base'][:3],s['base'][3:7],s['q'],s['dq'],support=True))
            self.assertEqual(g.native['motion'],'planner_motion')

    def test_guard_fake_cursor_without_native_ack_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=GroundGuard(self.clip.source,Path(tmp)/'controller');s=SkillSession(self.clip,experimental=True)
            s.acquire('autonomy',self.state(),self.clear(),10);s.phase='GROUND'
            stream=ReferencePoseStream(self.clip);stream.source_frame=500
            Path(tmp,'skill_stream.json').write_text(json.dumps(s.authorization(10,stream)))
            st=self.state(500)
            self.assertEqual(g.assess(10,st['base'][:3],st['base'][3:7],st['q'],st['dq'],efforts=st['tau'],state_monotonic=10),'stale_native_skill_feedback')

    def test_retained_transport_accepts_v1_only_after_requested_owner(self):
        client=planner_module().PlannerClient(simulation_only=True,clock=lambda:10.01)
        attach_reference_validation(client)
        p=ReferencePoseStream(self.clip).packet(10.01)
        with self.assertRaises(RuntimeError):client.offer_pose(p)
        client.set_anchor(0);client.start_control();client.acknowledge_started();client.request_pose()
        # New packet must postdate the explicit ownership request.
        client.clock=lambda:10.03
        client.offer_pose(ReferencePoseStream(self.clip).packet(10.03))
        self.assertEqual(client.status()['requested_mode'],'POSE')
        packets=client._packets(10.3)
        self.assertEqual(client.status()['requested_mode'],'STOPPED')
        self.assertIn('freshness timeout',client.status()['fault'])


if __name__=='__main__':unittest.main()
