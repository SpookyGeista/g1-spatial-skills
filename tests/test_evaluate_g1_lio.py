"""Evaluator regressions: synthetic fixtures test auditing, never integration."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.evaluate_g1_lio import (rigid, trajectory_errors, room_visits,
    timing_audit, trace_audit, pcd_audit, evaluate, source_audit, room_scan_audit, odometry_health)


def truth(stamp, position=(0,0,0)):
    t=np.eye(4);t[:3,3]=position
    return dict(stamp_ns=stamp,T_world_imu=t.tolist())


def odom(stamp, position=(0,0,0), q=(1,0,0,0)):
    return dict(stamp_ns=stamp,position_m=list(position),quaternion_wxyz=list(q))


class LioEvaluatorTest(unittest.TestCase):
    def test_fixed_registration_is_used_without_first_pose_fit(self):
        origin=np.eye(4);origin[:3,3]=[4,3,2]
        gt=[truth(0,[4,3,2]),truth(5_000_000,[4,3,2])]
        good=trajectory_errors([odom(0),odom(5_000_000)],gt,origin)
        self.assertEqual(good['translation_max_m'],0.)
        biased=trajectory_errors([odom(0,[.6,0,0]),odom(5_000_000,[.6,0,0])],gt,origin)
        self.assertAlmostEqual(biased['translation_rmse_m'],.6)
        self.assertGreater(biased['translation_max_m'],.4)

    def test_fixed_registration_rotation_applies_to_position(self):
        origin=np.array([[0,-1,0,2],[1,0,0,3],[0,0,1,0],[0,0,0,1]],float)
        gt=[truth(0,[2,4,0])];gt[0]['T_world_imu']=origin.copy().tolist()
        gt[0]['T_world_imu'][1][3]=4
        m=trajectory_errors([odom(0,[1,0,0])],gt,origin)
        self.assertEqual(m['translation_max_m'],0.)
        self.assertEqual(m['rotation_max_deg'],0.)

    def test_nonrigid_origin_rejected(self):
        for bad in [np.zeros((4,4)),np.eye(3),np.diag([-1,1,1,1])]:
            with self.assertRaises(ValueError):rigid(bad)

    def test_nan_position_and_invalid_quaternion_rejected(self):
        for bad in [odom(0,[float('nan'),0,0]),odom(0,q=[0,0,0,0])]:
            with self.assertRaises(ValueError):trajectory_errors([bad],[truth(0)],np.eye(4))

    def test_duplicate_or_noninteger_output_stamp_rejected(self):
        for rows in [[odom(0),odom(0)],[odom(.5)]]:
            with self.assertRaises(ValueError):trajectory_errors(rows,[truth(0)],np.eye(4))

    def test_no_truth_extrapolation_or_large_gap(self):
        for gt,stamp in [([truth(0),truth(5_000_000)],6_000_000),([truth(0),truth(50_000_000)],25_000_000)]:
            with self.assertRaises(ValueError):trajectory_errors([odom(stamp)],gt,np.eye(4))

    def test_distinct_physical_room_visits_not_map_coverage(self):
        rooms=[dict(id='west',bounds_xy=[[-4,-3],[-1,2]]),
               dict(id='east',bounds_xy=[[1,-3],[4,2]]),
               dict(id='north',bounds_xy=[[-4,2.3],[4,5]])]
        r=room_visits([[0,0],[0,3],[0,3.1]], [0,1,2], rooms)
        self.assertEqual([v['pelvis_center_entered'] for v in r],[False,False,True])
        self.assertEqual(r[2]['dwell_sim_s'],1.)

    def test_point94_loop_does_not_pass_retained_timing_band(self):
        wall=np.arange(0,12,.005)
        result=timing_audit(wall,.94*wall)
        self.assertFalse(result['pass_band'])
        self.assertTrue(all(not v['pass_band'] for v in result['windows']))
        self.assertTrue(timing_audit(wall,wall)['pass_band'])

    def test_timing_reset_rejected(self):
        with self.assertRaises(ValueError):timing_audit([0,1,2],[0,1,0])

    def fixture_trace(self, folder, reattach=False, nan_core=False):
        names=['wall_monotonic','sim_time','support','physics_step_index','tilt_deg']+[f'root_{i}' for i in range(13)]+[f'{p}_{i}' for p in ('q','dq','tau') for i in range(29)]+['guard_height_error_m']
        n=1601;data=np.zeros((n,len(names)));c={k:i for i,k in enumerate(names)}
        t=np.arange(n)*.005
        data[:,c['wall_monotonic']]=t;data[:,c['sim_time']]=t
        data[:,c['physics_step_index']]=np.arange(n);data[:,c['root_2']]=.8;data[:,c['root_3']]=1
        data[:,c['guard_height_error_m']]=np.nan
        if reattach:data[800:,c['support']]=1
        if nan_core:data[3,c['root_0']]=np.nan
        (folder/'sim').mkdir();np.savez(folder/'sim/trace.npz',data=data,columns=names)
        manifest=dict(mission_started_monotonic=.5,mission_seconds=7)
        events=[dict(event='mission_result',monotonic=7.5)]
        meta=dict(physics_dt_s=.005,scene=dict(rooms=[dict(id=r,bounds_xy=[[-4,-3],[4,5]]) for r in ['west','east','north']],exit_zone=dict(center_xy=[0,0],radius_m=.55)))
        return manifest,events,meta

    def test_optional_inactive_guard_nan_does_not_invalidate_physics(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args=self.fixture_trace(p)
            _,checks=trace_audit(p,*args)
            self.assertTrue(checks['full_rate_monotonic_physics'])
            self.assertTrue(checks['terminal_stable_full_window'])

    def test_core_nonfinite_rejected_and_support_reattach_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args=self.fixture_trace(p,nan_core=True)
            with self.assertRaises(ValueError):trace_audit(p,*args)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args=self.fixture_trace(p,reattach=True)
            _,checks=trace_audit(p,*args)
            self.assertFalse(checks['support_removed_no_reattach'])

    def test_missing_source_snapshot_fails(self):
        with tempfile.TemporaryDirectory() as d:
            m={'source_sha256':{'run.py':'a'*64},'extension_source_sha256':{'tool.py':'b'*64}}
            self.assertEqual(len(source_audit(Path(d),m)['changed']),2)

    def test_missing_evidence_fails_closed_and_report_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);result=evaluate(p,p/'report.json')
            self.assertFalse(result['audit_pass']);self.assertFalse(result['sensor_lio_integration_pass'])
            self.assertFalse(result['checks']['recording_evidence_valid'])
            with self.assertRaises(FileExistsError):evaluate(p,p/'report.json')

    def test_terminal_odometry_stall_and_count_loss_are_not_hidden_by_good_prefix(self):
        rows=[dict(stamp_ns=t,latest_sensor_clock_ns=t+10_000_000,simulation_lag_ns=10_000_000) for t in [0,100_000_000]]
        good=odometry_health(rows,[dict(acquisition_end_ns=200_000_000)],dict(counts=dict(odometry=2)))
        self.assertTrue(good['freshness_pass']);self.assertTrue(good['output_count_matches_bridge'])
        stalled=odometry_health(rows,[dict(acquisition_end_ns=900_000_000)],dict(counts=dict(odometry=3)))
        self.assertFalse(stalled['freshness_pass']);self.assertFalse(stalled['output_count_matches_bridge'])
        rows[0]['simulation_lag_ns']=0
        with self.assertRaises(ValueError):odometry_health(rows,[dict(acquisition_end_ns=200_000_000)],dict(counts=dict(odometry=2)))

    def scan_fixture(self, folder, oscillatory=False):
        (folder/'sim/sensors').mkdir(parents=True)
        ts=np.arange(0,12.005,.005);names=['sim_time','root_0','root_1','root_3','root_4','root_5','root_6']
        data=np.zeros((len(ts),len(names)));data[:,0]=ts
        yaw=np.zeros(len(ts));events=[];done=[];plan=[];rooms=[];used=[];frame_id=1
        for i,name in enumerate(['west','east','north']):
            start,end=i*4.,i*4.+3.9;mask=(ts>=start)&(ts<=end);x=i*3.
            u=(ts[mask]-start)/3.9
            yaw[mask]=.5*np.sin(u*16*np.pi) if oscillatory else u*6.2
            data[mask,1]=x
            events += [dict(event='scan_started',sim_time=start,base=[x,0,0]),dict(event='scan_completed',sim_time=end,measured_yaw_radians=6.2)]
            done.append(dict(id=name,scan_number=i+1,observed_scan_origin_m=[x,0]))
            plan.append(dict(id=name,observation_xy_m=[x,0]));rooms.append(dict(id=name,bounds_xy=[[x-1,-1],[x+1,1]]))
            for t in np.arange(start,end,.2):
                meta=dict(frame_id=frame_id,sim_time=float(t),height=2,width=2)
                np.savez(folder/'sim/sensors'/f'frame_{frame_id:06d}.npz',metadata=json.dumps(meta),rgb=np.zeros((2,2,3),np.uint8),depth=np.ones((2,2),np.float32))
                used.append(meta);frame_id+=1
        data[:,3]=np.cos(yaw/2);data[:,6]=np.sin(yaw/2)
        np.savez(folder/'sim/trace.npz',data=data,columns=names)
        (folder/'sensor_used.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in used))
        return dict(inspection_plan=dict(rooms=plan,arrival_radius_m=.18,maximum_settled_scan_displacement_m=.45),completed_room_scans=done),events,dict(scene=dict(rooms=rooms))

    def test_declared_room_scans_require_actual_net_yaw_and_retained_frames(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args=self.scan_fixture(p)
            self.assertTrue(room_scan_audit(p,*args)['pass_all'])
            (p/'sim/sensors/frame_000001.npz').unlink()
            with self.assertRaises(FileNotFoundError):room_scan_audit(p,*args)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args=self.scan_fixture(p,oscillatory=True)
            self.assertFalse(room_scan_audit(p,*args)['pass_all'])

    def test_truncated_map_rejected(self):
        header=b'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 100\nDATA binary\n'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'map.pcd';p.write_bytes(header+bytes(12*99))
            with self.assertRaises(ValueError):pcd_audit(p)
            p.write_bytes(header+bytes(12*100));self.assertEqual(pcd_audit(p)['points'],100)

if __name__=='__main__':unittest.main()
