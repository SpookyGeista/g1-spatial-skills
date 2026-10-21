#!/usr/bin/env python3
"""Independent post-run sensor fixture check; evaluation truth never feeds adapter."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def box_surface_distance(points,center,size):
    d=np.abs(points-np.asarray(center))-np.asarray(size)/2
    return np.linalg.norm(np.maximum(d,0),axis=1)+np.abs(np.minimum(np.max(d,axis=1),0))

def evaluate(run):
    run=Path(run);summary=json.loads((run/'smoke_summary.json').read_text())
    truth={r['stamp_ns']:r for r in map(json.loads,(run/'evaluation_truth.jsonl').read_text().splitlines())}
    events=list(map(json.loads,(run/'events.jsonl').read_text().splitlines()))
    imu=[r for r in events if r['kind']=='imu'];clouds=[r for r in events if r['kind']=='cloud']
    T_LI=np.asarray(summary['sensor']['T_lidar_imu'])
    residuals=[];counts=[];offsets=[];hashes={};all_complete=True;self_hits=0
    for event in clouds:
        path=run/event['file'];hashes[event['file']]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as scan:
            xyz=scan['xyz_sensor_m'];dt=scan['pointdt_ns'];m=json.loads(str(scan['metadata'].item()))
            assert xyz.dtype==np.float32 and dt.dtype==np.uint32
            assert np.isfinite(xyz).all() and xyz.shape==(len(dt),3)
            assert m['scanstart_ns']==event['scanstart_ns'] and np.all(dt<100000000)
            assert np.all(np.diff(dt.astype(np.int64))>=0) and np.all(dt%5000000==0)
            assert not any(k in scan.files for k in ('position','pose','orientation','groundtruth'))
            all_complete &= m['complete'];counts.append(len(xyz));offsets.extend(dt.tolist())
            for offset in np.unique(dt):
                stamp=int(event['scanstart_ns'])+int(offset);pose=truth[stamp]
                WI=np.asarray(pose['imu_to_world']);WL=WI@np.linalg.inv(T_LI)
                local=xyz[dt==offset];world=local@WL[:3,:3].T+WL[:3,3]
                d=np.min(np.stack([box_surface_distance(world,p,size) for _,p,size in summary['fixture']['walls']]),axis=0)
                # The rig collider rotates/translates under PhysX; use its measured transform.
                from punktnull.sensors.mid360 import rotation_wxyz
                torso=pose['torso_pose_xyz_wxyz'];R=rotation_wxyz(torso[3:])
                rig_local=(world-np.asarray(torso[:3]))@R
                dr=box_surface_distance(rig_local,[0,0,0],[.1,.1,.1]);self_hits+=int(np.sum(dr<d))
                residuals.extend(np.minimum(d,dr).tolist())
    r=np.asarray(residuals);stamps=np.array([e['stamp_ns'] for e in imu],dtype=np.int64)
    result={'scope':'Posthoc exact procedural-fixture surface check using withheld PhysX truth; not LIO accuracy',
            'clouds':len(clouds),'imu_records':len(imu),'total_returns':len(r),'self_rig_returns':self_hits,
            'returns_per_scan_min_max':[min(counts),max(counts)],'surface_residual_m_p50_p99_max':np.percentile(r,[50,99,100]).tolist(),
            'checks':{'all_scans_complete':bool(all_complete),'all_imu_intervals5ms':bool(np.all(np.diff(stamps)==5000000)),
                      'point_clock5ms_quantized':True,'scan_end_bracketed_by_imu':int(stamps[-1])>=clouds[-1]['scanstart_ns']+100000000,
                      'all_point_surface_residual_below1mm':bool(np.max(r)<.001)},
            'input_sha256':{'events.jsonl':hashlib.sha256((run/'events.jsonl').read_bytes()).hexdigest(),
                             'evaluation_truth.jsonl':hashlib.sha256((run/'evaluation_truth.jsonl').read_bytes()).hexdigest(),**hashes}}
    dest=run/'sensor_geometry_check.json'
    with dest.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}))
    return all(result['checks'].values())
if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    raise SystemExit(0 if evaluate(a.run) else 1)
