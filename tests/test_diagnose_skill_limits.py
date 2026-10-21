"""Synthetic failed-trace tests; not robot validation."""
import copy,hashlib,json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from punktnull.skills.limits import load_joint_limits
from tools.diagnose_skill_limits import verify_contract,diagnose,audit_braking

class FailedTraceDiagnosticTests(unittest.TestCase):
    def fixture(self):
        limits=load_joint_limits()
        m=dict(outcome='FAILED',error='synthetic aborted trial',acceptance=dict(limits=limits.provenance()))
        meta=dict(joint_names=list(limits.names),effort_limits_nm=limits.effort.tolist(),events=[],end_reason='fall',error=None)
        return limits,m,meta
    def test_mismatched_recorded_contract_refuses_comparison(self):
        limits,m,meta=self.fixture();self.assertTrue(verify_contract(m,meta,limits)['matched'])
        altered=copy.deepcopy(m);altered['acceptance']['limits']['position_lower_rad'][0]-=.01
        with self.assertRaisesRegex(ValueError,'contract mismatch'):verify_contract(altered,meta,limits)
        altered=copy.deepcopy(meta);altered['effort_limits_nm'][20]=6.
        with self.assertRaisesRegex(ValueError,'metadata'):verify_contract(m,altered,limits)
        with self.assertRaisesRegex(ValueError,'provenance'):verify_contract({},meta,limits)
    def test_missing_completion_events_does_not_hide_measured_crossing(self):
        limits,m,meta=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'sim').mkdir()
            (run/'manifest.json').write_text(json.dumps(m));(run/'sim/metadata.json').write_text(json.dumps(meta))
            cols=['wall_monotonic','sim_time','support','physics_step_index','tilt_deg','guard_reference_frame']+[f'root_{i}' for i in range(13)]+[f'{key}_{i}' for key in ('q','dq','tau') for i in range(29)]
            a=np.zeros((3,len(cols)));a[:,0]=[10,10.005,10.01];a[:,1]=[0,.005,.01];a[:,2]=[1,1,0];a[:,3]=[0,1,2]
            a[:,5]=-1;a[:,cols.index('root_2')]=.8;a[:,cols.index('root_3')]=1.
            a[2,cols.index('q_10')]=limits.lower[10]-.003
            np.savez_compressed(run/'sim/trace.npz',data=a,columns=cols)
            before=hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest();r=diagnose(run)
            self.assertTrue(r['diagnostic_complete']);self.assertEqual(r['recorded_outcome_unchanged'],'FAILED')
            self.assertEqual(r['violation_counts']['position']['unsupported_sample_joint_crossings'],1)
            self.assertEqual(r['violations'][0]['joint'],'right_ankle_pitch_joint')
            self.assertFalse(r['limiter_telemetry']['available']);self.assertEqual(r['acceptance_result'],'NOT_REEVALUATED_BY_THIS_DIAGNOSTIC')
            self.assertEqual(before,hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest())

class BrakingEvidenceTests(unittest.TestCase):
    def make_evidence(self,run):
        from punktnull.skills.target_limits import AnkleTargetLimiter
        from punktnull.skills.legacy import LEGACY,load
        bridge=load('braking_diagnostic_test_bridge',LEGACY/'bridge.py')
        command=bridge.CommandSnapshot(q=[0.]*29,dq=[0.]*29,kp=[10.]*29,kd=[2.]*29,tau=[0.]*29,received_at=10.)
        limiter=AnkleTargetLimiter(enabled=True,include_pitch=True,experimental_braking=True)
        columns=['sim_time']+[f'{k}_{i}' for k in ('q','dq','tau') for i in range(29)];trace=[]
        for i in range(2):
            q=np.zeros(29);q[10]=-.86;dq=np.zeros(29);dq[10]=-3
            raw=bridge.command_efforts(command,q,dq)
            output=limiter.apply(command,raw,q,dq,wall=10.01+i*.005,sim=1+i*.005,support=False,status=dict(mode='active'))
            trace.append(np.r_[1+i*.005,q,dq,np.clip(output,-limiter.limits.effort,limiter.limits.effort)])
        limiter.save(run/'sim')
        from tools.diagnose_skill_limits import ROOT
        for relative in ('punktnull/skills/ankle_braking.py','punktnull/skills/limits.py'):
            target=run/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
        return np.asarray(trace),columns

    def test_missing_requested_braking_is_not_zero_correction(self):
        with tempfile.TemporaryDirectory() as d:
            result,cache=audit_braking(Path(d),dict(experimental_ankle_braking=True),np.empty((0,0)),[])
            self.assertFalse(result['verified']);self.assertFalse(result['available']);self.assertIsNone(cache)
            self.assertIn('missing',result['errors'][0])

    def test_exact_native_target_braking_and_final_clip_verified(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);trace,columns=self.make_evidence(run)
            result,cache=audit_braking(run,dict(experimental_ankle_braking=True),trace,columns)
            self.assertTrue(result['verified'],result);self.assertIsNotNone(cache)
            altered=trace.copy();altered[1,columns.index('tau_10')]+=.1
            result,cache=audit_braking(run,dict(experimental_ankle_braking=True),altered,columns)
            self.assertFalse(result['verified']);self.assertIsNone(cache)
            self.assertIn('effort mismatch',result['errors'][0])

    def test_hash_valid_but_wrong_same_tick_state_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);trace,columns=self.make_evidence(run);path=run/'sim/ankle_braking.npz'
            with np.load(path,allow_pickle=False) as z:items={key:z[key].copy() for key in z.files}
            cols=list(items['columns']);items['data'][0,cols.index('right_ankle_pitch_joint_q')]+=.001
            np.savez_compressed(path,**items)
            metadata=run/'sim/ankle_braking.json';bm=json.loads(metadata.read_text());bm['npz_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();metadata.write_text(json.dumps(bm))
            metadata=run/'sim/ankle_target_limiter.json';lm=json.loads(metadata.read_text());lm['braking_evidence']=bm;metadata.write_text(json.dumps(lm))
            result,cache=audit_braking(run,dict(experimental_ankle_braking=True),trace,columns)
            self.assertFalse(result['verified']);self.assertIsNone(cache)
            self.assertIn('same physics tick',result['errors'][0])

if __name__=='__main__':unittest.main()
