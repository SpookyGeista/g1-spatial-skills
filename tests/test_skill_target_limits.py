"""Synthetic native-PD unit tests; no simulation/reliability acceptance."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from punktnull.skills.legacy import LEGACY,load
from punktnull.skills.target_limits import AnkleTargetLimiter
bridge=load('target_limiter_test_bridge',LEGACY/'bridge.py')

class TargetLimiterTests(unittest.TestCase):
    def command(self):
        q=np.zeros(29);q[5]=.4;q[11]=-.4
        return bridge.CommandSnapshot(q=q,dq=np.ones(29)*.1,kp=np.ones(29)*10,kd=np.ones(29)*2,tau=np.ones(29)*.3,received_at=10.)
    def invoke(self,limiter,c=None,**changes):
        c=self.command() if c is None else c
        q=np.zeros(29);dq=np.ones(29)*.2;torque=np.asarray(bridge.command_efforts(c,q,dq))
        values=dict(wall=10.01,sim=1.,support=False,status=dict(mode='active',last_rejection=None))
        values.update(changes)
        return limiter.apply(c,torque,q,dq,**values),torque

    def test_default_off_is_exact_passthrough(self):
        obj=AnkleTargetLimiter();raw=object()
        self.assertIs(obj.apply(None,raw,None,None,wall=None,sim=None,support=None,status=None),raw)
    def test_only_two_enabled_target_terms_change(self):
        obj=AnkleTargetLimiter(enabled=True);c=self.command();actual,raw=self.invoke(obj,c)
        desired=np.asarray(c.q).copy();desired[obj.indices]=np.clip(desired[obj.indices],obj.lower,obj.upper)
        expected=np.asarray(bridge.command_efforts(replace(c,q=desired),np.zeros(29),np.ones(29)*.2))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
        untouched=np.ones(29,bool);untouched[obj.indices]=False
        np.testing.assert_array_equal(actual[untouched],raw[untouched]);self.assertEqual(c.q[11],-.4)
    def test_pitch_is_separate_opt_in_and_logs_all_four_joints(self):
        c=self.command();q=list(c.q);q[4]=-.95;q[10]=-1.09;c=replace(c,q=q)
        default=AnkleTargetLimiter(enabled=True);old,raw=self.invoke(default,c)
        self.assertEqual(old[4],raw[4]);self.assertEqual(old[10],raw[10])
        new=AnkleTargetLimiter(enabled=True,include_pitch=True);actual,raw=self.invoke(new,c)
        desired=np.asarray(c.q).copy();desired[new.indices]=np.clip(desired[new.indices],new.lower,new.upper)
        expected=np.asarray(bridge.command_efforts(replace(c,q=desired),np.zeros(29),np.ones(29)*.2))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
        self.assertEqual(set(new.indices),{4,5,10,11})
        with tempfile.TemporaryDirectory() as d:
            report=new.save(d);self.assertTrue(report['include_pitch']);self.assertEqual(len(report['joints']),4)
            with np.load(Path(d)/'ankle_target_limiter.npz',allow_pickle=False) as z:self.assertEqual(z['data'].shape,(1,49))

    def test_disabled_motor_and_supported_startup_are_untouched(self):
        c=self.command();mode=list(c.mode);mode[11]=0;c=replace(c,mode=mode)
        obj=AnkleTargetLimiter(enabled=True);actual,raw=self.invoke(obj,c)
        self.assertEqual(actual[11],0.)
        obj=AnkleTargetLimiter(enabled=True);actual,raw=self.invoke(obj,c,support=True)
        np.testing.assert_array_equal(actual,raw)
    def test_absent_supported_command_is_disclosed_without_modification(self):
        obj=AnkleTargetLimiter(enabled=True);raw=np.zeros(29)
        out=obj.apply(None,raw,raw,raw,wall=10.,sim=0.,support=True,status=dict(mode='waiting'))
        np.testing.assert_array_equal(out,raw)
        self.assertEqual(obj.rows[0][4],-1)
    def test_active_missing_stale_future_and_mismatch_fail_closed(self):
        for age in (.26,-.001):
            obj=AnkleTargetLimiter(enabled=True)
            with self.assertRaisesRegex(ValueError,'stale/future'):self.invoke(obj,wall=10+age)
        obj=AnkleTargetLimiter(enabled=True);q=np.zeros(29)
        with self.assertRaisesRegex(ValueError,'missing'):obj.apply(None,q,q,q,wall=10.01,sim=1.,support=False,status=dict(mode='active'))
        obj=AnkleTargetLimiter(enabled=True)
        with self.assertRaisesRegex(ValueError,'exact native'):obj.apply(self.command(),q,q,q,wall=10.01,sim=1.,support=False,status=dict(mode='active'))
    def test_snapshot_and_clock_cannot_change_silently(self):
        obj=AnkleTargetLimiter(enabled=True);self.invoke(obj)
        c=replace(self.command(),q=np.zeros(29))
        with self.assertRaisesRegex(ValueError,'without new receipt'):self.invoke(obj,c,wall=10.02,sim=1.005)
        obj=AnkleTargetLimiter(enabled=True);self.invoke(obj)
        with self.assertRaisesRegex(ValueError,'clock'):self.invoke(obj)
    def test_margin_and_nonfinite_rejected(self):
        for margin in (-.01,.081,float('nan')):
            with self.assertRaises(ValueError):AnkleTargetLimiter(enabled=True,margin_rad=margin)
        obj=AnkleTargetLimiter(enabled=True);q=np.zeros(29);bad=q.copy();bad[0]=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid limiter'):obj.apply(self.command(),bad,q,q,wall=10.01,sim=1.,support=False,status=dict(mode='active'))
    def test_exact_snapshot_evidence_saved_once(self):
        obj=AnkleTargetLimiter(enabled=True);c=self.command();self.invoke(obj,c)
        self.invoke(obj,c,wall=10.02,sim=1.005)
        with tempfile.TemporaryDirectory() as d:
            result=obj.save(d);self.assertEqual(result['native_snapshots'],1);self.assertEqual(result['physics_samples'],2)
            with np.load(Path(d)/'ankle_target_limiter.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['native_command_snapshots'][0,0],c.q)
                self.assertEqual(z['data'].shape,(2,27))
            with self.assertRaises(FileExistsError):obj.save(d)

if __name__=='__main__':unittest.main()
