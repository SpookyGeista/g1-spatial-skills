"""Synthetic negative-test audit fixtures, not native/Isaac execution evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.test_mission_faults import audit,listener_rows,validate_manifest


def fixture(folder,kind='unavailable_pose'):
    run=Path(folder);out=run/('fault_test_'+kind)
    out.mkdir();(run/'sim').mkdir();(run/'controller').mkdir()
    failed='Takeover requested but live whole-body POSE unavailable' if kind=='unavailable_pose' else 'Operator abort'
    request=2.;end=3. if kind=='unavailable_pose' else 2.02
    events=[dict(event='transition',state='SCANNING',monotonic=1,sim_time=1)]
    if kind=='unavailable_pose':events += [dict(event='operator_command',command='takeover',state='SCANNING',monotonic=request,sim_time=request),
        dict(event='transition',state='ASSISTANCE_REQUESTED',monotonic=2.01,sim_time=2.01)]
    events.append(dict(event='transition',state='ABORTED',monotonic=end,sim_time=end,detail=failed))
    manifest=dict(test='explore',sim_only=True,real_actuators_enabled=False,live_human_tested=False,injection_kind=None,
        controller_command=['/existing/controller','lo'],events=events,outcome='FAILED',failure=failed,cleanup_errors=[],mission_started_monotonic=1.)
    (run/'manifest.json').write_text(json.dumps(manifest))
    meta=dict(physics_dt_s=.005,end_reason='user_stop',error=None,events=[])
    (run/'sim/metadata.json').write_text(json.dumps(meta))
    plan=dict(fault=kind,declared_monotonic=.8)
    (out/'plan.json').write_text(json.dumps(plan))
    injection=dict(pose_listener_before_stale=dict(listener_absent=True),pose_listener_before_fresh=dict(listener_absent=True),
        stale_command_window=dict(sent_monotonic=1.4,confirmed_consumed_monotonic=1.8,all_files_consumed=True,
            commands=[dict(command=c,monotonic=-9) for c in ['takeover','release']]),
        fresh_command=dict(command='takeover' if kind=='unavailable_pose' else 'abort',monotonic=request,sim_time=request,state='SCANNING'))
    (out/'injection.json').write_text(json.dumps(injection));(out/'stop_observation.json').write_text(json.dumps(dict(simulator_pid_absent=True)))
    names=['wall_monotonic','sim_time','physics_step_index','support','tilt_deg']+[f'root_{i}' for i in range(13)]+[f'{k}_{i}' for k in ('q','dq','tau') for i in range(29)]+['inactive_guard_nan']
    t=np.arange(0,end+.015,.005);c={n:i for i,n in enumerate(names)};data=np.zeros((len(t),len(names)))
    data[:,c['wall_monotonic']]=t;data[:,c['sim_time']]=t;data[:,c['physics_step_index']]=np.arange(len(t))
    data[t<.5,c['support']]=1;data[:,c['root_2']]=.8;data[:,c['inactive_guard_nan']]=np.nan
    yaw=np.clip(t-1,0,1)*.5
    data[:,c['root_3']]=np.cos(yaw/2);data[:,c['root_6']]=np.sin(yaw/2)
    np.savez(run/'sim/trace.npz',data=data,columns=names)
    for name,key,value in [('motion_name','motion_name','planner_motion'),('encoder_mode','encoder_mode_0','0'),('motion_playing','playing_0','1')]:
        rows=['index,time_monotonic_ms,'+key]+[f'{i},{stamp*1000},{value}' for i,stamp in enumerate(np.arange(0,end+.03,.02))]
        (run/'controller'/(name+'.csv')).write_text('\n'.join(rows)+'\n')
    return out,manifest,data,names


class FaultAuditTest(unittest.TestCase):
    def test_tcp_ipv4_ipv6_and_non_listener_are_distinguished(self):
        header='sl local remote st tx rx tr tm retr uid inode\n'
        rows=header+' 0: 0100007F:15B4 00000000:0000 0A 0 0 0 0 0 123\n'
        self.assertEqual(len(listener_rows(rows)),1)
        self.assertFalse(listener_rows(rows.replace('0A','01')))
        self.assertEqual(len(listener_rows(rows.replace('0100007F:','00000000000000000000000000000000:'))),1)
        with self.assertRaises(ValueError):listener_rows(header+'bad line')

    def test_non_simulation_or_other_injection_rejected(self):
        good=dict(test='explore',sim_only=True,real_actuators_enabled=False,live_human_tested=False,injection_kind=None,controller_command=['bin','lo'])
        for key,value in [('sim_only',False),('controller_command',['bin','eth0']),('injection_kind','explicit_test_operator_review'),('live_human_tested',True)]:
            with self.assertRaises(ValueError):validate_manifest(dict(good,**{key:value}))

    def test_unavailable_source_negative_pass_keeps_mission_failed(self):
        with tempfile.TemporaryDirectory() as d:
            out,_,_,_=fixture(d);r=audit(Path(d),out)
            self.assertTrue(r['negative_test_pass'],r['checks']);self.assertEqual(r['mission_outcome'],'FAILED');self.assertFalse(r['mission_completed'])

    def test_explicit_abort_is_a_separate_case(self):
        with tempfile.TemporaryDirectory() as d:
            out,_,_,_=fixture(d,'explicit_abort');r=audit(Path(d),out)
            self.assertTrue(r['negative_test_pass'],r['checks'])
            self.assertNotIn('settled_before_unavailable_source_abort',r['checks'])

    def test_accepted_stale_command_is_not_hidden_by_expected_final_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out,m,_,_=fixture(d);m['events'].append(dict(event='operator_command',command='release',monotonic=1.5,sim_time=1.5))
            Path(d,'manifest.json').write_text(json.dumps(m));r=audit(Path(d),out)
            self.assertFalse(r['checks']['stale_takeover_release_consumed_without_acceptance']);self.assertFalse(r['negative_test_pass'])

    def test_missing_pose_must_not_ever_enter_operator_control(self):
        with tempfile.TemporaryDirectory() as d:
            out,m,_,_=fixture(d);m['events'].append(dict(event='transition',state='OPERATOR_CONTROL',monotonic=2.1,sim_time=2.1))
            Path(d,'manifest.json').write_text(json.dumps(m));self.assertFalse(audit(Path(d),out)['negative_test_pass'])

    def test_late_physics_stop_and_support_reattach_fail(self):
        with tempfile.TemporaryDirectory() as d:
            out,_,data,names=fixture(d)
            data[:,names.index('wall_monotonic')]+=.3;data[-1,names.index('support')]=1
            np.savez(Path(d,'sim/trace.npz'),data=data,columns=names)
            r=audit(Path(d),out);self.assertFalse(r['checks']['physics_stopped_within_025s_of_abort'])
            self.assertFalse(r['checks']['support_removed_never_reattached'])

    def test_runtime_success_never_counts_as_expected_negative_pass(self):
        with tempfile.TemporaryDirectory() as d:
            out,m,_,_=fixture(d);m['outcome']='FOUND_AND_RETURNED';Path(d,'manifest.json').write_text(json.dumps(m))
            self.assertFalse(audit(Path(d),out)['negative_test_pass'])


if __name__=='__main__':unittest.main()
