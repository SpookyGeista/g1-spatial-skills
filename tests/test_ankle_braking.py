"""CPU calculation/transport invariants, not simulated containment proof."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from punktnull.skills.ankle_braking import assistance,AnkleBraking
from punktnull.skills.target_limits import AnkleTargetLimiter
from punktnull.skills.legacy import LEGACY,load
bridge=load('braking_test_bridge',LEGACY/'bridge.py')

class AnkleBrakingTests(unittest.TestCase):
    def command(self):return bridge.CommandSnapshot(q=[0.]*29,dq=[0.]*29,kp=[10.]*29,kd=[2.]*29,tau=[0.]*29,received_at=10.)
    def call(self,obj,q=None,dq=None,effort=None,**changes):
        q=np.zeros(29) if q is None else q;dq=np.zeros(29) if dq is None else dq;effort=np.zeros(29) if effort is None else effort
        options=dict(wall=10.01,sim=1.,support=False,status=dict(mode='active',last_rejection=None),snapshot=self.command(),command_index=0)
        options.update(changes);return obj.apply(effort,q,dq,**options)
    def test_symmetric_lower_upper_and_cap(self):
        lower,upper=-.87267,.5236;q=lower+.05
        left=assistance(q,-4,lower,upper);right=assistance(lower+upper-q,4,lower,upper)
        self.assertEqual(left['signed_assist'],-right['signed_assist']);self.assertEqual(left['capped_assist_lower'],20.)
    def test_inward_motion_has_no_outward_damping_but_keeps_restoring_term(self):
        t=assistance(-.25,.1,-.2618,.2618)
        self.assertEqual(t['outward_speed_lower'],0.)
        self.assertAlmostEqual(t['raw_assist_lower'],10*t['weight_lower'])
        self.assertGreater(t['signed_assist'],0.)
        self.assertEqual(assistance(0,0,-.2618,.2618)['signed_assist'],0.)
        self.assertGreater(assistance(-.1,-4,-.2618,.2618)['signed_assist'],0.)
    def test_other_joints_disabled_motors_and_support_unchanged(self):
        q=np.zeros(29);q[10]=-.86;effort=np.arange(29,dtype=float)
        obj=AnkleBraking(enabled=True);out=self.call(obj,q=q,effort=effort)
        mask=np.ones(29,bool);mask[obj.indices]=False;np.testing.assert_array_equal(out[mask],effort[mask])
        modes=list(self.command().mode);modes[10]=0
        out=self.call(AnkleBraking(enabled=True),q=q,effort=effort,snapshot=replace(self.command(),mode=modes))
        self.assertEqual(out[10],effort[10])
        out=self.call(AnkleBraking(enabled=True),q=q,effort=effort,support=True,snapshot=None,status=dict(mode='waiting'))
        np.testing.assert_array_equal(out,effort)
    def test_existing_final50nm_clip_remains_needed(self):
        q=np.zeros(29);q[10]=-.86;dq=np.zeros(29);dq[10]=-5;effort=np.zeros(29);effort[10]=45
        obj=AnkleBraking(enabled=True);out=self.call(obj,q=q,dq=dq,effort=effort)
        self.assertEqual(out[10],65.)
        self.assertEqual(np.clip(out,-obj.limits.effort,obj.limits.effort)[10],50.)
        row=obj.rows[0];self.assertEqual(row[obj.columns.index('right_ankle_pitch_joint_expected_effort_after_existing_clip')],50.)
    def test_stale_missing_rejected_nonfinite_and_clock_fail(self):
        for changes in (dict(wall=10.26),dict(snapshot=None),dict(status=dict(mode='rejected')),dict(wall=9.99)):
            with self.assertRaisesRegex(ValueError,'fresh accepted'):self.call(AnkleBraking(enabled=True),**changes)
        q=np.zeros(29);q[0]=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid braking'):self.call(AnkleBraking(enabled=True),q=q)
        obj=AnkleBraking(enabled=True);self.call(obj)
        with self.assertRaisesRegex(ValueError,'clock'):self.call(obj)
    def test_default_off_and_parent_integration_evidence(self):
        sentinel=object();self.assertIs(AnkleBraking().apply(sentinel,None,None,wall=None,sim=None,support=None,status=None,snapshot=None),sentinel)
        q=np.zeros(29);q[10]=-.86;dq=np.zeros(29);dq[10]=-3;c=self.command();raw=np.asarray(bridge.command_efforts(c,q,dq))
        obj=AnkleTargetLimiter(enabled=True,include_pitch=True,experimental_braking=True)
        out=obj.apply(c,raw,q,dq,wall=10.01,sim=1.,support=False,status=dict(mode='active'))
        self.assertGreater(out[10],raw[10]);self.assertEqual(len(obj.braking.rows),1)
        with tempfile.TemporaryDirectory() as d:
            report=obj.save(d);self.assertTrue(report['experimental_braking'])
            self.assertTrue((Path(d)/'ankle_braking.npz').exists());self.assertTrue((Path(d)/'ankle_braking.json').exists())
            self.assertEqual(report['braking_evidence']['snapshot_index_source']['file'],'ankle_target_limiter.npz')
            with self.assertRaises(FileExistsError):obj.save(d)

if __name__=='__main__':unittest.main()
