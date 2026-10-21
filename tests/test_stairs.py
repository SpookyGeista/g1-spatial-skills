"""CPU tests; never labelled native/physics acceptance."""
import hashlib,json,tempfile,unittest,shutil
from pathlib import Path
import numpy as np
from punktnull.skills.reference import ReferenceClip,WIDTHS
from punktnull.skills.stairs import stable_landing,StairSession,TOP
from punktnull.skills.legacy import WORKSPACE
from tools.stair_trial import install_reused_run

class StairTests(unittest.TestCase):
    def state(self):
        return dict(base=[.8,0,TOP+.78,1,0,0,0,0,0,0,0,0,0],dq=[0]*29,q=[0]*29,tau=[0]*29,
            support=False,command_mode='active',tilt_deg=0,
            stair_contacts=dict(fresh=True,both_on_landing=True))
    def test_landing_needs_actual_both_contact(self):
        s=self.state();self.assertTrue(stable_landing(s));s['stair_contacts']['both_on_landing']=False
        self.assertFalse(stable_landing(s))
    def test_landing_rejects_stale_support_motion_and_tilt(self):
        for key,value in [('support',True),('tilt_deg',13),('command_mode','inactive')]:
            s=self.state();s[key]=value;self.assertFalse(stable_landing(s))
        s=self.state();s['stair_contacts']['fresh']=False;self.assertFalse(stable_landing(s))
        s=self.state();s['base'][7]=.11;self.assertFalse(stable_landing(s))
    def test_landing_height_relative_to_real_platform(self):
        s=self.state();s['base'][2]=.78;self.assertFalse(stable_landing(s))
    def test_generated_data_checked_and_endpoint_stationary(self):
        c=ReferenceClip.load(WORKSPACE/'assets/stairs_s01_reference_003')
        self.assertGreater(c.root[-1,2]-c.root[0,2],.20)
        self.assertTrue(np.allclose(c.data['joint_vel'][-45:],0))
        self.assertTrue(np.allclose(c.root[-45:],c.root[-1]))
    def test_corrupt_local_channel_rejected(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'evidence') as d:
            p=Path(d)/'reference';shutil.copytree(WORKSPACE/'assets/stairs_s01_reference_003',p)
            f=p/'joint_pos.csv';f.write_text(f.read_text().replace('0.000000000','0.001000000',1))
            # Some source channels contain no exactzero; append a byte if needed.
            f.write_bytes(f.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):ReferenceClip.load(p)
    def test_cursor_finished_is_not_completed_without_landing(self):
        c=ReferenceClip.load(WORKSPACE/'assets/stairs_s01_reference_003');session=StairSession(c)
        session.token='owner';session.phase='SETTLING';session.entered=10;session.initial_xy=np.zeros(2)
        s=self.state();s.update(monotonic=11,sim_time=1,yaw_deg=0,landing_stable_full_rate_seconds=0)
        a=session.update('owner',s,None,11,native_stream_active=True,submitted_frame=c.frames-1,inferred_frame=c.frames-1,tracking_error=[0,0,0],finished_sending=True)
        self.assertEqual(session.phase,'SETTLING');self.assertEqual(a['action'],'HOLD_REFERENCE')
    def test_native_owner_loss_aborts(self):
        c=ReferenceClip.load(WORKSPACE/'assets/stairs_s01_reference_003');session=StairSession(c)
        session.token='owner';session.phase='ENTERING';session.entered=10;session.initial_xy=np.zeros(2)
        s=self.state();s.update(monotonic=11,sim_time=1,yaw_deg=0)
        a=session.update('owner',s,None,11,native_stream_active=False,submitted_frame=5,inferred_frame=5,tracking_error=[0,0,0],finished_sending=False)
        self.assertEqual(a['action'],'PAUSE_SIM_AND_STOP_NATIVE')
    def test_cursor_waits_for_real_first_riser_support(self):
        c=ReferenceClip.load(WORKSPACE/'assets/stairs_s01_reference_003');session=StairSession(c)
        session.token='owner';session.phase='ENTERING';session.entered=10;session.initial_xy=np.zeros(2)
        s=self.state();s.update(monotonic=11,sim_time=1,yaw_deg=0)
        kwargs=dict(native_stream_active=True,submitted_frame=55,inferred_frame=55,tracking_error=[0,0,0],finished_sending=False)
        a=session.update('owner',s,None,11,**kwargs);self.assertEqual(a['action'],'HOLD_REFERENCE')
        s['first_riser_contact_seconds']=.11
        a=session.update('owner',s,None,11,**kwargs);self.assertEqual(a['action'],'ADVANCE_REFERENCE')
        self.assertTrue(session.first_support_confirmed)
    def test_retained_supervisor_substitutions_still_exact(self):install_reused_run()
if __name__=='__main__':unittest.main()
