"""Synthetic audit regressions; none are native/Isaac integration evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.replay_mission_review import audit, native_audit, validate_manifest, validate_run_dir


def manifest_contract():
    return dict(test='explore', sim_only=True, real_actuators_enabled=False,
        injection_kind='explicit_test_operator_review', injected_review_after_room='west', live_human_tested=False)


def synthetic_run(folder):
    """Build labelled synthetic CSV/trace records solely to exercise rejection."""
    run = Path(folder); (run/'sim').mkdir(); (run/'controller').mkdir(); (run/'mission_review_replay').mkdir()
    events = []
    for start, finish in [(1, 3), (4, 7), (15, 18), (22, 25)]:
        events.extend([dict(event='scan_started', monotonic=start),
                       dict(event='scan_completed', monotonic=finish, measured_yaw_radians=2*np.pi, sensor_frames=30)])
    events += [dict(event='navigation_limit', status='OPERATOR_REVIEW_REQUIRED', monotonic=8),
        dict(event='transition', state='TAKEOVER_PENDING', monotonic=8.5),
        dict(event='transition', state='OPERATOR_CONTROL', monotonic=8.8),
        dict(event='transition', state='RETURN_CHECK', monotonic=12),
        dict(event='autonomy_resumed', monotonic=13, scene_reset=False),
        dict(event='mission_result', monotonic=30, outcome='FOUND_AND_RETURNED')]
    events.sort(key=lambda e:e['monotonic'])
    rooms = [('west', [-2.1,-.1], 2), ('east', [2.,-.1], 3), ('north', [0.,3.25], 4)]
    manifest = {**manifest_contract(), 'events':events, 'outcome':'FOUND_AND_RETURNED',
        'mission_started_monotonic':.8, 'mission_seconds':29.2,
        'completed_room_scans':[dict(id=r, observation_xy_m=xy, observed_scan_origin_m=xy, scan_number=n) for r,xy,n in rooms],
        'target_observation':dict(target_id=7, confirming_frames=[10,11,12]), 'cleanup_errors':[], 'synthetic_test_fixture':True}
    (run/'manifest.json').write_text(json.dumps(manifest))
    metadata = dict(physics_dt_s=.005, error=None, scene=dict(
        rooms=[dict(id='west', bounds_xy=[[-4,-2],[-1,2]]), dict(id='east', bounds_xy=[[1,-2],[4,2]]),
               dict(id='north', bounds_xy=[[-2,2],[2,5]])], exit_zone=dict(center_xy=[0,0], radius_m=.55)))
    (run/'sim/metadata.json').write_text(json.dumps(metadata))
    (run/'mission_review_replay/source.json').write_text(json.dumps(dict(source='recorded_pose_replay', live_human_acceptance=False)))
    times=np.arange(0,31.005,.005)
    names=['wall_monotonic','sim_time','support','physics_step_index','tilt_deg']+[f'root_{i}' for i in range(13)]+[f'{k}_{i}' for k in ('q','dq','tau') for i in range(29)]+['guard_optional']
    col={n:i for i,n in enumerate(names)};data=np.zeros((len(times),len(names)))
    for key in ['wall_monotonic','sim_time']: data[:,col[key]]=times
    data[:,col['physics_step_index']]=np.arange(len(times)); data[times<.5,col['support']]=1
    data[:,col['root_2']]=.8;data[:,col['guard_optional']]=np.nan
    points=np.array([[0,0,0],[3,0,0],[4,-2.1,-.1],[13,-2.1,-.1],[15,2,-.1],[18,2,-.1],
                     [22,0,3.25],[25,0,3.25],[29,0,0],[31,0,0]],float)
    for i in [0,1]:data[:,col['root_'+str(i)]]=np.interp(times,points[:,0],points[:,i+1])
    for i in [0,1]:data[:,col['root_'+str(7+i)]]=np.gradient(data[:,col['root_'+str(i)]],.005)
    yaw=np.zeros(len(times))
    for start,finish in [(1,3),(4,7),(15,18),(22,25)]: yaw+=np.clip((times-start)/(finish-start),0,1)*2*np.pi
    data[:,col['root_3']]=np.cos(yaw/2);data[:,col['root_6']]=np.sin(yaw/2)
    np.savez(run/'sim/trace.npz',data=data,columns=names)
    for field,key in [('motion_name','motion_name'),('encoder_mode','encoder_mode_0'),('motion_playing','playing_0')]:
        lines=['index,time_monotonic_ms,'+key]
        for index,t in enumerate(np.arange(0,31,.02)):
            operator=8.56<=t<12.20
            value=('streamed' if operator else 'planner_motion') if field=='motion_name' else (2 if operator else 0) if field=='encoder_mode' else 1
            lines.append(f'{index},{t*1000:.4f},{value}')
        (run/'controller'/(field+'.csv')).write_text('\n'.join(lines)+'\n')
    return manifest, data, names


class ReviewAuditTest(unittest.TestCase):
    def native_fixture(self):
        rows=[]
        for index,t in enumerate(np.arange(0,31,.02)):
            value=('streamed',2,1) if 8.60<=t<12.20 else ('planner_motion',0,1)
            if 8.56<=t<8.60:value=('temporary_motion',0,0)
            rows.append((index,float(t),*value))
        return rows

    def native_result(self, rows):
        return native_audit(rows,8,8.5,8.8,12,13,30)

    def test_source_verified_paused_transition_is_explicitly_bounded(self):
        result=self.native_result(self.native_fixture())
        self.assertTrue(result['pass']);self.assertEqual(result['nonsteady_source_mode_samples'],2)
        self.assertEqual(result['invalid_source_mode_samples'],0)
        self.assertTrue(result['bounded_native_transition_hold']['accepted'])
        self.assertAlmostEqual(result['bounded_native_transition_hold']['duration_until_stream_s'],.04)

    def test_temporary_motion_in_steady_phase_or_return_is_never_exempted(self):
        for t in [8.84,10.,12.10,13.2,20.]:
            rows=self.native_fixture();i=min(range(len(rows)),key=lambda i:abs(rows[i][1]-t))
            rows[i]=(*rows[i][:2],'temporary_motion',0,0)
            self.assertFalse(self.native_result(rows)['pass'],t)

    def test_playing_or_wrong_encoder_temporary_motion_is_rejected(self):
        for wrong in [('temporary_motion',0,1),('temporary_motion',2,0),('unknown',0,0)]:
            rows=self.native_fixture();i=next(i for i,r in enumerate(rows) if r[2]=='temporary_motion')
            rows[i]=(*rows[i][:2],*wrong)
            self.assertFalse(self.native_result(rows)['pass'],wrong)

    def test_long_transition_and_six_samples_are_rejected(self):
        rows=self.native_fixture()
        for i,r in enumerate(rows):
            if 8.52<=r[1]<8.64:rows[i]=(*r[:2],'temporary_motion',0,0)
        self.assertFalse(self.native_result(rows)['pass'])
        # Two samples alone do not justify a pause extending beyond100 ms.
        times=np.r_[np.arange(0,8.52,.02),8.52,8.54,np.arange(8.64,31,.02)]
        rows=[]
        for i,t in enumerate(times):
            value=('temporary_motion',0,0) if 8.51<t<8.55 else ('streamed',2,1) if 8.63<t<12.2 else ('planner_motion',0,1)
            rows.append((i,float(t),*value))
        result=self.native_result(rows)
        self.assertFalse(result['pass']);self.assertEqual(result['bounded_native_transition_hold']['samples'],2)
        self.assertGreater(result['bounded_native_transition_hold']['duration_until_stream_s'],.10)

    def test_paused_copy_requires_direct_prior_planner_and_following_stream(self):
        for side in ['before','after']:
            rows=self.native_fixture();indices=[i for i,r in enumerate(rows) if r[2]=='temporary_motion']
            i=indices[0]-1 if side=='before' else indices[-1]+1
            rows[i]=(*rows[i][:2], 'streamed' if side=='before' else 'planner_motion', 2 if side=='before' else 0,1)
            self.assertFalse(self.native_result(rows)['pass'],side)

    def test_ordinary_or_hardware_run_rejected(self):
        for key,value in [('sim_only',False),('real_actuators_enabled',True),('test','handover'),
                          ('injection_kind',None),('injected_review_after_room','invalid'),('live_human_tested',True)]:
            m=manifest_contract();m[key]=value
            with self.assertRaises(ValueError): validate_manifest(m)
        with self.assertRaises(ValueError):validate_run_dir('/tmp/unrelated_run')

    def test_good_synthetic_evidence_remains_explicitly_nonhuman(self):
        with tempfile.TemporaryDirectory() as d:
            synthetic_run(d);result=audit(Path(d))
            self.assertTrue(result['native_replay_review_pass'],result['checks'])
            self.assertFalse(result['live_human_acceptance']);self.assertFalse(result['useful_operator_recovery_demonstrated'])
            self.assertGreater(result['post_resume_max_displacement_from_release_m'],.2)
            self.assertEqual(len(result['room_scans']['rooms']),3)

    def test_failed_mission_cannot_be_reported_as_review_pass(self):
        with tempfile.TemporaryDirectory() as d:
            m,_,_=synthetic_run(d);m['outcome']='FAILED';Path(d,'manifest.json').write_text(json.dumps(m))
            result=audit(Path(d));self.assertFalse(result['native_replay_review_pass'])
            self.assertFalse(result['checks']['final_outcome_is_found_and_returned'])

    def test_native_pose_outside_owner_window_is_rejected(self):
        rows=[]
        for index,t in enumerate(np.arange(0,31,.02)):
            pose=8.56<=t<12.20 or 20<=t<20.1
            rows.append((index,t,'streamed' if pose else 'planner_motion',2 if pose else 0,1))
        r=native_audit(rows,8,8.5,8.8,12,13,30)
        self.assertFalse(r['pass']);self.assertGreater(r['unexpected_streamed_samples'],0)

    def test_support_reattach_fails_without_hiding_recorded_mission_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            _,data,names=synthetic_run(d);data[4000:,names.index('support')]=1
            np.savez(Path(d,'sim/trace.npz'),data=data,columns=names)
            result=audit(Path(d));self.assertFalse(result['checks']['trace_clock_and_support_preserved'])
            self.assertFalse(result['native_replay_review_pass'])

    def test_clock_reset_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _,data,names=synthetic_run(d);data[4000:,names.index('sim_time')]-=1
            np.savez(Path(d,'sim/trace.npz'),data=data,columns=names)
            with self.assertRaises(ValueError):audit(Path(d))

    def test_claimed_scans_without_physical_turn_fail(self):
        with tempfile.TemporaryDirectory() as d:
            _,data,names=synthetic_run(d);data[:,names.index('root_3')]=1;data[:,names.index('root_6')]=0
            np.savez(Path(d,'sim/trace.npz'),data=data,columns=names)
            result=audit(Path(d));self.assertFalse(result['checks']['three_actual_completed_room_scans'])

    def test_missing_room_cannot_pass_from_waypoint_or_two_scans(self):
        with tempfile.TemporaryDirectory() as d:
            m,_,_=synthetic_run(d);m['completed_room_scans'].pop();Path(d,'manifest.json').write_text(json.dumps(m))
            self.assertFalse(audit(Path(d))['checks']['three_actual_completed_room_scans'])

    def test_duplicate_takeover_is_not_silently_chosen(self):
        with tempfile.TemporaryDirectory() as d:
            m,_,_=synthetic_run(d);m['events'].append(dict(event='autonomy_resumed',monotonic=20,scene_reset=False))
            Path(d,'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):audit(Path(d))


if __name__=='__main__':unittest.main()
