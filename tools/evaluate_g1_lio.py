#!/usr/bin/env python3
"""Read-only CPU audit of retained G1 FAST-LIO runs; never starts ROS/Isaac.

Uses the exact recorded runtime registration, without fitting any estimate to
truth. Simulator truth is evaluation input only. This audit does not replace
independent RGB target decoding, skill admission, live-human or hardware tests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = dict(translation_rmse_m=.20, translation_max_m=.40,
                  rotation_max_deg=10., post_initialization_scan_coverage=.90,
                  odometry_max_age_sim_s=.5, loop_ratio_min=.95,
                  loop_ratio_max=1.05, stable_seconds=.60,
                  stable_height_m=[.70, .90], stable_tilt_deg=12.,
                  stable_speed_m_s=.13, stable_joint_speed_rad_s=2.,
                  upright_min_height_m=.38, upright_max_tilt_deg=50.,
                  room_scan_minimum_net_yaw_rad=2*math.pi-.10)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text())


def json_lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def rigid(value):
    t = np.asarray(value, float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(t[:3, :3]) - 1) > 1e-6):
        raise ValueError('Invalid proper rigid transform')
    return t


def quat_matrix(value):
    q = np.asarray(value, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > .001:
        raise ValueError('Invalid unit quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def stamps(rows):
    if not rows or any(type(r['stamp_ns']) is not int for r in rows):
        raise ValueError('Missing records or noninteger timestamps')
    ts = np.asarray([r['stamp_ns'] for r in rows], dtype=np.int64)
    if ts[0] < 0 or np.any(np.diff(ts) <= 0):
        raise ValueError('Nonmonotonic or negative timestamps')
    return ts


def truth_at(ts, poses, stamp):
    i = int(np.searchsorted(ts, stamp))
    if i < len(ts) and ts[i] == stamp: return poses[i]
    if i == 0 or i == len(ts): raise ValueError('Odometry outside retained truth range')
    if ts[i] - ts[i-1] > 10_000_000: raise ValueError('Truth gap exceeds 10 ms')
    a, b = poses[i-1], poses[i]
    u = float(stamp - ts[i-1]) / float(ts[i] - ts[i-1])
    rel = a[:3, :3].T @ b[:3, :3]
    angle = math.acos(float(np.clip((np.trace(rel)-1)/2, -1, 1)))
    if angle > .2: raise ValueError('Unphysical rotation jump in truth interpolation')
    out = np.eye(4)
    if angle < 1e-10: out[:3, :3] = a[:3, :3]
    else:
        v = np.array([rel[2,1]-rel[1,2], rel[0,2]-rel[2,0], rel[1,0]-rel[0,1]]) / (2*math.sin(angle))
        x, y, z = v
        k = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        out[:3, :3] = a[:3, :3] @ (np.eye(3)+math.sin(u*angle)*k+(1-math.cos(u*angle))*(k@k))
    out[:3, 3] = (1-u)*a[:3, 3]+u*b[:3, 3]
    return out


def trajectory_errors(odometry, truth, registration):
    """No alignment parameter is estimated here, including from the first sample."""
    origin = rigid(registration)
    ot, tt = stamps(odometry), stamps(truth)
    gt = [rigid(r['T_world_imu']) for r in truth]
    translations, rotations = [], []
    for row, stamp in zip(odometry, ot):
        p = np.asarray(row['position_m'], float)
        if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError('Invalid odometry translation')
        t = np.eye(4); t[:3,:3] = quat_matrix(row['quaternion_wxyz']); t[:3,3] = p
        actual, measured = origin @ t, truth_at(tt, gt, int(stamp))
        translations.append(float(np.linalg.norm(actual[:3,3]-measured[:3,3])))
        rotations.append(math.degrees(math.acos(float(np.clip((np.trace(actual[:3,:3].T@measured[:3,:3])-1)/2,-1,1)))))
    return dict(samples=len(ot), first_stamp_ns=int(ot[0]), last_stamp_ns=int(ot[-1]),
                translation_rmse_m=float(np.sqrt(np.mean(np.square(translations)))),
                translation_max_m=max(translations), rotation_max_deg=max(rotations),
                odometry_gap_sim_s_max=float(np.max(np.diff(ot))*1e-9) if len(ot)>1 else None)


def odometry_health(odometry, clouds, bridge):
    ts=stamps(odometry)
    if not clouds: raise ValueError('No acquired cloud coverage')
    for row in odometry:
        age=row['latest_sensor_clock_ns']-row['stamp_ns']
        if age<0 or age!=row['simulation_lag_ns']: raise ValueError('Inconsistent odometry acquisition-age record')
    final_age=(clouds[-1]['acquisition_end_ns']-int(ts[-1]))*1e-9
    max_gap=float(np.max(np.diff(ts))*1e-9) if len(ts)>1 else None
    max_age=max(float(r['simulation_lag_ns'])*1e-9 for r in odometry)
    return dict(output_count_matches_bridge=len(odometry)==bridge['counts']['odometry'],
                final_complete_scan_to_last_estimate_sim_s=final_age,
                largest_output_gap_sim_s=max_gap,
                freshness_pass=bool(max_gap is not None and 0<=final_age<=.5 and max_gap<=.5 and max_age<=.5))


def room_visits(xy, sim_times, rooms):
    xy, times = np.asarray(xy, float), np.asarray(sim_times, float)
    if (xy.shape != (len(times), 2) or not len(times) or not np.isfinite(xy).all()
            or not np.isfinite(times).all() or np.any(np.diff(times)<=0)):
        raise ValueError('Invalid physical room-visit samples')
    if not rooms or len({r['id'] for r in rooms}) != len(rooms): raise ValueError('Missing/duplicate room definitions')
    dt = np.r_[np.diff(times), 0.]
    result = []
    for room in rooms:
        bounds = np.asarray(room['bounds_xy'], float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]):
            raise ValueError('Invalid room bounds')
        inside = np.all((xy>=bounds[0]) & (xy<=bounds[1]), axis=1)
        result.append(dict(id=room['id'], pelvis_center_entered=bool(inside.any()),
                           samples_inside=int(inside.sum()), dwell_sim_s=float(dt[inside].sum()),
                           first_entry_sim_s=float(times[inside][0]) if inside.any() else None,
                           bounds_xy=bounds.tolist()))
    return result


def timing_audit(wall, sim, window_s=5.):
    wall, sim = np.asarray(wall, float), np.asarray(sim, float)
    if (wall.shape!=sim.shape or len(wall)<2 or not np.isfinite(wall).all()
            or not np.isfinite(sim).all() or np.any(np.diff(wall)<=0) or np.any(np.diff(sim)<=0)):
        raise ValueError('Invalid/nonmonotonic timing samples')
    windows, i = [], 0
    while i < len(wall)-1:
        j = int(np.searchsorted(wall, wall[i]+window_s))
        if j >= len(wall): break  # The short final tail is included in overall ratio only.
        ratio = float((sim[j]-sim[i])/(wall[j]-wall[i]))
        windows.append(dict(start_sim_s=float(sim[i]), end_sim_s=float(sim[j]), ratio=ratio,
                            pass_band=bool(.95<=ratio<=1.05)))
        i = j
    overall = float((sim[-1]-sim[0])/(wall[-1]-wall[0]))
    return dict(overall_ratio=overall, windows=windows,
                complete_five_second_windows=len(windows),
                pass_band=bool(windows and .95<=overall<=1.05 and all(r['pass_band'] for r in windows)),
                interpretation='Simulation seconds / wall seconds. Fixed-fps video is not a wall-time performance measurement.')


def pcd_audit(path):
    raw = path.read_bytes(); pos = 0; fields = {}
    while True:
        end = raw.find(b'\n', pos)
        if end < 0: raise ValueError('Incomplete PCD header')
        row = raw[pos:end].decode('ascii').strip(); pos = end+1
        if row and not row.startswith('#'):
            key, *value = row.split(); fields[key] = value
            if key == 'DATA': break
    n = int(fields['POINTS'][0]); names = fields['FIELDS']
    sizes = list(map(int, fields['SIZE'])); counts = list(map(int, fields.get('COUNT', ['1']*len(names))))
    types = fields['TYPE']
    if not len(names)==len(sizes)==len(counts)==len(types): raise ValueError('Bad PCD schema')
    if fields['DATA'] != ['binary'] or n < 100: raise ValueError('Expected nonempty native binary PCD')
    offsets = np.cumsum([0]+[s*c for s,c in zip(sizes, counts)])
    for key in ('x','y','z'):
        i = names.index(key)
        if (types[i],sizes[i],counts[i]) != ('F',4,1): raise ValueError('Unexpected PCD XYZ type')
    if len(raw)-pos != n*int(offsets[-1]): raise ValueError('Truncated/extra native PCD bytes')
    dt = np.dtype(dict(names=['x','y','z'],formats=['<f4']*3,
                       offsets=[int(offsets[names.index(k)]) for k in ('x','y','z')],itemsize=int(offsets[-1])))
    data = np.frombuffer(raw,dtype=dt,count=n,offset=pos)
    if not all(np.isfinite(data[k]).all() for k in ('x','y','z')): raise ValueError('Nonfinite PCD XYZ')
    return dict(points=n, bytes=len(raw), sha256=digest(path), finite_xyz=True,
                meaning='Native registered map retained; nonempty/finite does not by itself prove geometric map accuracy.')


def source_audit(run, manifest):
    checked, changed = {}, []
    for key, prefix in [('source_sha256','source_snapshot'), ('extension_source_sha256','source_snapshot/extension')]:
        if not manifest.get(key): raise ValueError('Missing declared source inventory: '+key)
        for name, expected in manifest[key].items():
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe source inventory path')
            p = run / prefix / rel
            actual = digest(p) if p.is_file() else None
            checked[str(Path(prefix)/rel)] = actual
            if actual != expected: changed.append(str(Path(prefix)/rel))
    return dict(files=len(checked), changed=changed, actual_sha256=checked,
                scope='Frozen baseline and extension source snapshots only; does not retroactively certify a binary not hashed at launch.')


def trace_audit(run, manifest, events, metadata):
    with np.load(run/'sim/trace.npz', allow_pickle=False) as z:
        data, names = z['data'], list(z['columns'])
    required = ['wall_monotonic','sim_time','support','physics_step_index','tilt_deg'] + [f'root_{i}' for i in range(13)] + [f'{k}_{i}' for k in ('q','dq','tau') for i in range(29)]
    if data.ndim!=2 or data.shape[1]!=len(names) or len(set(names))!=len(names): raise ValueError('Invalid physical trace shape')
    columns = {k:data[:,names.index(k)] for k in required}
    if len(data)<2 or not np.isfinite(np.column_stack(list(columns.values()))).all(): raise ValueError('Nonfinite physical state')
    # Optional guard diagnostics legitimately contain NaN while no guard is active.
    wall, sim, step = columns['wall_monotonic'], columns['sim_time'], columns['physics_step_index']
    if np.any(np.diff(wall)<=0) or np.any(np.diff(sim)<=0) or np.any(np.diff(step)!=1): raise ValueError('Reset, gap or nonmonotonic full-rate trace')
    dt = float(metadata['physics_dt_s'])
    clock_error = float(np.max(np.abs(np.diff(sim)-dt)))
    terminal = [e for e in events if e.get('event')=='mission_result']
    if len(terminal)!=1: raise ValueError('Need exactly one final mission_result event')
    end = terminal[0]['monotonic']; begin = manifest['mission_started_monotonic']
    active = (wall>=begin)&(wall<=end)
    if active.sum()<2 or end>wall[-1] or begin<wall[0]: raise ValueError('Trace does not cover entire mission interval')
    root = np.column_stack([columns[f'root_{i}'] for i in range(13)])
    norm = np.linalg.norm(root[:,3:7],axis=1)
    if np.max(np.abs(norm-1))>.001: raise ValueError('Invalid trace orientation')
    support = columns['support']
    release = np.flatnonzero(support==0)
    unsupported = bool(len(release) and np.all(support[release[0]:]==0) and np.all(support[active]==0))
    xyz = root[active,:3]; at = sim[active]
    rooms = room_visits(xyz[:,:2], at, metadata['scene']['rooms'])
    exit_zone = metadata['scene']['exit_zone']
    terminal_i = np.flatnonzero(active)[-1]
    stable_window = (sim>=sim[terminal_i]-.6)&(sim<=sim[terminal_i])
    speed = np.linalg.norm(root[:,7:10],axis=1)
    dq = np.column_stack([columns[f'dq_{i}'] for i in range(29)])
    stable = ((support==0)&(root[:,2]>=.70)&(root[:,2]<=.90)&(columns['tilt_deg']<=12)&(speed<=.13)&(np.max(np.abs(dq),axis=1)<=2.))
    return_distance = float(np.linalg.norm(xyz[-1,:2]-xyz[0,:2]))
    exit_distance = float(np.linalg.norm(xyz[-1,:2]-np.asarray(exit_zone['center_xy'],float)))
    timing = timing_audit(wall[active], at)
    segments = []
    transitions = [e for e in events if e.get('event')=='transition' and begin<=e['monotonic']<=end]
    for i,e in enumerate(transitions):
        finish = transitions[i+1]['monotonic'] if i+1<len(transitions) else end
        mask = (wall>=e['monotonic'])&(wall<=finish)
        if mask.sum()>1:
            a,b = np.flatnonzero(mask)[[0,-1]]
            segments.append(dict(state=e['state'],wall_s=float(wall[b]-wall[a]),sim_s=float(sim[b]-sim[a]),ratio=float((sim[b]-sim[a])/(wall[b]-wall[a]))))
    timing['phase_segments'] = segments
    result = dict(samples=len(data), mission_samples=int(active.sum()), mission_wall_s=float(end-begin),
                  manifest_mission_wall_s=manifest.get('mission_seconds'), mission_sim_s=float(at[-1]-at[0]),
                  clock_step_error_s_max=clock_error, support_never_reattached=unsupported,
                  min_root_height_m=float(xyz[:,2].min()), max_tilt_deg=float(columns['tilt_deg'][active].max()),
                  physical_start_xy_m=xyz[0,:2].tolist(), physical_terminal_xy_m=xyz[-1,:2].tolist(),
                  physical_return_distance_m=return_distance, exit_zone_distance_m=exit_distance,
                  exit_zone_radius_m=float(exit_zone['radius_m']), terminal_stability_samples=int(stable_window.sum()),
                  terminal_stability_fraction=float(stable[stable_window].mean()),
                  rooms=rooms, room_visit_meaning='Pelvis center in saved room rectangle; no claim of complete room scan, camera coverage or whole-body clearance.',timing=timing)
    checks = dict(full_rate_monotonic_physics=clock_error<1e-6, support_removed_no_reattach=unsupported,
                  bounded_upright_physics=result['min_root_height_m']>=.38 and result['max_tilt_deg']<=50,
                  physical_return_in_exit_zone=exit_distance<=float(exit_zone['radius_m']),
                  terminal_stable_full_window=bool(stable_window.sum()>=120 and np.all(stable[stable_window])),
                  all_three_rooms_physically_entered={r['id'] for r in rooms}=={'west','east','north'} and all(r['pelvis_center_entered'] for r in rooms),
                  mission_loop_ratio_in_retained_band=timing['pass_band'])
    return result, checks


def capture_audit(run):
    capture = run/'sim/mid360'; rows = json_lines(capture/'events.jsonl')
    clouds, imus, previous, points, hashes = [], [], -1, 0, {}
    for e in rows:
        if e['kind']=='imu':
            if set(e) != {'kind','stamp_ns','angular_velocity_rad_s','specific_force_m_s2'}: raise ValueError('Unexpected IMU payload fields')
            for key in ('angular_velocity_rad_s','specific_force_m_s2'):
                v=np.asarray(e[key],float)
                if v.shape!=(3,) or not np.isfinite(v).all(): raise ValueError('Invalid IMU vectors')
            event_time=e['stamp_ns']; imus.append(e)
        elif e['kind']=='cloud':
            rel=Path(e['file'])
            if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe cloud path')
            p=capture/rel
            with np.load(p,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata'].item())); xyz=z['xyz_sensor_m']; offsets=z['pointdt_ns']
                n=len(offsets); start=int(meta['scanstart_ns']); end=int(meta['scan_end_ns'])
                if xyz.shape!=(n,3) or n==0 or not np.isfinite(xyz).all(): raise ValueError('Invalid LiDAR points')
                if offsets.dtype.kind not in 'ui' or np.any(np.diff(offsets.astype(np.int64))<0) or int(offsets.max())>=end-start: raise ValueError('Invalid per-point timestamps')
                if end-start!=100_000_000 or start!=e['stamp_ns'] or not meta['complete']: raise ValueError('Incomplete/inconsistent scan')
                for k in ('line','intensity'):
                    if z[k].shape!=(n,) or not np.isfinite(z[k]).all(): raise ValueError('Invalid point attributes')
            clouds.append(dict(stamp_ns=start,acquisition_end_ns=end,points=n,file=e['file']))
            event_time=end; points+=n; hashes[e['file']]=digest(p)
        else: raise ValueError('Unknown sensor event')
        if type(event_time) is not int or event_time<previous: raise ValueError('Nonmonotonic sensor acquisition clock')
        previous=event_time
    stamps(imus); stamps(clouds)
    publication=json_lines(run/'lio/publication_index.jsonl')
    desired=[(e['kind'],e['stamp_ns']) for e in rows]
    actual=[(e['kind'],e['stamp_ns']) for e in publication]
    if desired!=actual: raise ValueError('Retained input stream differs from published stream')
    published_clouds=[e for e in publication if e['kind']=='cloud']
    if [(e['stamp_ns'],e['acquisition_end_ns'],e['points']) for e in clouds]!=[(e['stamp_ns'],e['acquisition_end_ns'],e['points']) for e in published_clouds]: raise ValueError('Published cloud shape/timing differs')
    meta=json_file(capture/'metadata.json'); bridge=json_file(run/'lio/bridge_result.json')
    counts=bridge['counts']
    if (counts['imu'],counts['cloud'],counts['points'])!=(len(imus),len(clouds),points): raise ValueError('Bridge counts differ from inputs')
    result=dict(imu=len(imus),clouds=len(clouds),points=points,scan_file_sha256=hashes,
                bridge=bridge,sensor_model=meta,source_contract='Acquired point XYZ/per-point time + angular velocity and specific force; no truth pose/orientation fields in IMU events.')
    valid=(not bridge.get('error') and not meta.get('error') and counts['clock_rejections']==0
           and counts['rejected_odometry']==0 and bridge.get('groundtruth_corrections') is False
           and bridge.get('orientation_measurements') is False and bridge.get('physical_device') is False)
    return result, clouds, published_clouds, bool(valid)


def room_scan_audit(run, manifest, events, metadata):
    """Cross-check declared scans against physical pelvis yaw and retained camera frames."""
    plan = manifest['inspection_plan']
    planned = {r['id']:r for r in plan['rooms']}
    completed = manifest.get('completed_room_scans', [])
    if set(planned) != {'west','east','north'} or len(completed)!=3 or {r['id'] for r in completed}!=set(planned):
        raise ValueError('Three distinct planned/completed room scans are required')
    scan_pairs=[];pending=None
    for e in events:
        if e.get('event')=='scan_started':
            if pending is not None: raise ValueError('Overlapping/uncompleted scan')
            pending=e
        elif e.get('event')=='scan_completed':
            if pending is None: raise ValueError('Scan completion without start')
            scan_pairs.append((pending,e));pending=None
    if pending is not None: raise ValueError('Incomplete final scan')
    used=json_lines(run/'sensor_used.jsonl');ids=[r['frame_id'] for r in used]
    if len(ids)!=len(set(ids)) or np.any(np.diff(ids)<=0): raise ValueError('Duplicate/nonmonotonic consumed camera frames')
    with np.load(run/'sim/trace.npz',allow_pickle=False) as z:
        data=z['data'];names=list(z['columns'])
    col={n:data[:,i] for i,n in enumerate(names)}
    q=np.column_stack([col[f'root_{i}'] for i in range(3,7)])
    yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    xy=np.column_stack([col['root_0'],col['root_1']]);times=col['sim_time']
    rooms={r['id']:r for r in metadata['scene']['rooms']};results=[];seen=set()
    for r in completed:
        number=r['scan_number']
        if type(number) is not int or number<1 or number>len(scan_pairs) or number in seen: raise ValueError('Invalid/reused scan number')
        seen.add(number);start,end=scan_pairs[number-1]
        mask=(times>=start['sim_time'])&(times<=end['sim_time'])
        if mask.sum()<2: raise ValueError('Missing physical scan samples')
        unwrapped=np.unwrap(yaw[mask]);rotation=abs(float(unwrapped[-1]-unwrapped[0]))
        drift=float(np.max(np.linalg.norm(xy[mask]-xy[mask][0],axis=1)))
        bounds=np.asarray(rooms[r['id']]['bounds_xy'],float)
        inside=bool(np.all((xy[mask]>=bounds[0])&(xy[mask]<=bounds[1])))
        observed=np.asarray(r['observed_scan_origin_m'],float)
        desired=np.asarray(planned[r['id']]['observation_xy_m'],float)
        if observed.shape!=(2,) or desired.shape!=(2,) or not np.isfinite(observed).all() or not np.isfinite(desired).all(): raise ValueError('Invalid observation coordinate')
        if not np.allclose(observed,start['base'][:2],atol=1e-9,rtol=0): raise ValueError('Declared scan origin does not match scan-start event')
        frames=[f for f in used if start['sim_time']-.01<=f['sim_time']<=end['sim_time']+.01]
        retained=[]
        for f in frames:
            path=run/'sim/sensors'/f"frame_{f['frame_id']:06d}.npz"
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata'].item()));rgb=z['rgb'];depth=z['depth']
                if meta['frame_id']!=f['frame_id'] or abs(meta['sim_time']-f['sim_time'])>1e-8: raise ValueError('Consumed camera frame does not match archive')
                if rgb.dtype!=np.uint8 or rgb.shape!=(meta['height'],meta['width'],3) or depth.shape!=rgb.shape[:2] or not np.any(np.isfinite(depth)&(depth>0)): raise ValueError('Invalid rendered camera archive')
            retained.append(dict(frame_id=f['frame_id'],sim_time=f['sim_time'],sha256=digest(path)))
        gaps=np.diff([f['sim_time'] for f in frames])
        camera_ok=len(frames)>=20 and len(gaps)>0 and float(gaps.max())<=.45
        result=dict(id=r['id'],scan_number=number,physical_net_yaw_rad=rotation,
            physical_scan_drift_m=drift,physical_center_inside_room_entire_scan=inside,
            physical_start_xy_m=xy[mask][0].tolist(),estimated_arrival_distance_m=float(np.linalg.norm(observed-desired)),
            physical_arrival_distance_m=float(np.linalg.norm(xy[mask][0]-desired)),
            camera_frames=len(frames),camera_max_gap_sim_s=float(gaps.max()) if len(gaps) else None,
            retained_camera_frames=retained,reported_yaw_rad=end['measured_yaw_radians'])
        # Retained supervisor's 2*pi-.10 completion threshold, checked against
        # actual net physical rotation instead of integrated estimated motion.
        result['pass']=bool(rotation>=THRESHOLDS['room_scan_minimum_net_yaw_rad'] and drift<=plan['maximum_settled_scan_displacement_m']
            and inside and result['estimated_arrival_distance_m']<=plan['arrival_radius_m'] and camera_ok)
        results.append(result)
    return dict(rooms=results,pass_all=all(r['pass'] for r in results),
        scope='Actual net pelvis yaw, observed position and captured RGB-D continuity. Does not prove every room surface was visible or target identity was detected.')


def decode_video(path, ffmpeg=None):
    binary=ffmpeg or shutil.which('ffmpeg')
    if not binary:
        conda=Path('/home/loya2/miniconda3/envs/unitree_sim_env/bin/python')
        if conda.is_file():
            found=subprocess.run([str(conda),'-c','import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())'],capture_output=True,text=True,timeout=15,check=True)
            binary=found.stdout.strip().splitlines()[-1]
    if not binary: raise RuntimeError('No installed CPU ffmpeg decoder; full video integrity remains unverified')
    command=[str(binary),'-v','error','-threads','1','-i',str(path),'-map','0:v:0','-f','null','-','-progress','pipe:1','-nostats']
    p=subprocess.run(command,capture_output=True,text=True,timeout=180)
    frames=[int(x.split('=',1)[1]) for x in p.stdout.splitlines() if x.startswith('frame=')]
    if p.returncode or p.stderr.strip() or 'progress=end' not in p.stdout or not frames: raise ValueError('Full video decode failed: '+p.stderr[-1000:])
    return dict(decoded_frames=frames[-1],decoder=str(binary),return_code=p.returncode,sha256=digest(path))


def evaluate(run, output, ffmpeg=None):
    run,output=Path(run).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('Refusing to replace evidence/report: '+str(output))
    if not output.parent.is_dir(): raise ValueError('Output parent must exist')
    report=dict(schema_version=1,run=str(run),scope='Independent retained actual-G1 LIO integration audit; no simulation/ROS execution',
                alignment='Exact saved runtime T_map_initial_imu multiplied by every native estimate. Zero posthoc alignment, fitting, resetting or truth correction.',
                thresholds=THRESHOLDS,threshold_provenance='LIO bounds from ACCEPTANCE.md; 0.95–1.05 timing and physical stable/upright thresholds retained from baseline evaluator. Three-room visits are evaluated physically, not inferred from mapped cells.',
                checks={},errors=[],warnings=[],input_sha256={},evaluator_sha256=digest(__file__),
                original_full_demo_acceptance='NOT_EVALUATED: live human intervention, terrain skills, raw RGB-D target re-detection and external/hardware validation are separate gates.')
    checks=report['checks']
    # Each independent evidence group continues after another fails.
    def section(name, call):
        try: call()
        except Exception as exc:
            checks[name+'_evidence_valid']=False;report['errors'].append(dict(section=name,error=repr(exc)))
    manifest={}; events=[]; metadata={}; clouds=[]; pubs=[]
    def basics():
        nonlocal manifest,events,metadata
        manifest=json_file(run/'manifest.json');events=json_lines(run/'events.jsonl');metadata=json_file(run/'sim/metadata.json')
        report['declared_outcome']=manifest.get('outcome')
        results=[e for e in events if e.get('event')=='mission_result']
        checks['declared_completed_result']=bool(manifest.get('outcome') in ('FOUND_AND_RETURNED','SEARCH_COMPLETE_NOT_FOUND') and len(results)==1 and results[0].get('outcome')==manifest['outcome'] and not manifest.get('error') and not manifest.get('cleanup_errors') and metadata.get('error') is None)
        checks['simulation_only']=manifest.get('sim_only') is True and manifest.get('real_actuators_enabled') is False
        report['source_integrity']=source_audit(run,manifest);checks['recorded_source_snapshots_match']=not report['source_integrity']['changed']
        old_label=manifest.get('map_summary',{}).get('localization')
        if old_label and 'simulator pose' in old_label: report['warnings'].append('Inherited map_summary localization label is inconsistent with LIO localization declaration: '+old_label)
        report['interventions']=dict(assistance_request_transitions=sum(e.get('event')=='transition' and e.get('state')=='ASSISTANCE_REQUESTED' for e in events),
            operator_control_transitions=sum(e.get('event')=='transition' and e.get('state')=='OPERATOR_CONTROL' for e in events),
            explicit_release_commands=sum(e.get('event')=='operator_command' and e.get('command')=='release' for e in events),
            live_human_tested=bool(manifest.get('live_human_tested')))
        report['target_observation']=dict(retained=(run/'target_observation.json').is_file(),image_retained=(run/'target_observation.png').is_file(),independently_redecoded_by_this_audit=False)
    section('manifest_sources',basics)
    def capture():
        nonlocal clouds,pubs
        report['sensors'],clouds,pubs,ok=capture_audit(run);checks['sensor_transport_integrity']=ok
    section('sensor_capture',capture)
    def estimator():
        origin=json_file(run/'localization_origin.json');truth=json_lines(run/'sim/mid360/evaluation_truth.jsonl');odom=json_lines(run/'lio/odometry.jsonl')
        if any('rejected' in r for r in odom): raise ValueError('Rejected native odometry retained')
        if not clouds: raise ValueError('Valid scan index unavailable')
        if any(r.get('frame_id')!='camera_init' or r.get('child_frame_id')!='body' for r in odom): raise ValueError('Unexpected native odometry frames')
        cov=np.asarray([r['covariance'] for r in odom],float)
        if cov.shape!=(len(odom),36) or not np.isfinite(cov).all(): raise ValueError('Invalid native covariance')
        checks['runtime_origin_manifest_matches']=origin==manifest.get('initial_localization_registration')
        checks['runtime_origin_matches_first_truth']=bool(np.allclose(rigid(origin['T_map_initial_imu']),rigid(truth[0]['T_world_imu']),atol=1e-8,rtol=0))
        report['runtime_registration']=origin
        health=odometry_health(odom,clouds,json_file(run/'lio/bridge_result.json'));report['odometry_health']=health
        checks['native_odometry_count_matches_bridge']=health['output_count_matches_bridge']
        m=trajectory_errors(odom,truth,origin['T_map_initial_imu']);report['trajectory']=m
        ts=stamps(odom);first=next((i for i,s in enumerate(clouds) if s['stamp_ns']<=ts[0]<s['acquisition_end_ns']),None)
        if first is None: raise ValueError('First native estimate is not inside an acquired scan')
        after=clouds[first:];matched=[bool(np.any((ts>=s['stamp_ns'])&(ts<s['acquisition_end_ns']))) for s in after]
        m.update(post_initialization_scans=len(after),matched_post_initialization_scans=sum(matched),post_initialization_scan_coverage=sum(matched)/len(after),missing_scan_files=[s['file'] for s,ok in zip(after,matched) if not ok],initialization_sim_s=(int(ts[0])-clouds[0]['stamp_ns'])*1e-9)
        lag=np.asarray([r['simulation_lag_ns'] for r in odom],float)*1e-9
        if not np.isfinite(lag).all() or np.any(lag<0): raise ValueError('Invalid odometry lag')
        latency=[]
        for r in odom:
            c=next((s for s in pubs if s['stamp_ns']<=r['stamp_ns']<s['acquisition_end_ns']),None)
            if c is None: raise ValueError('Odometry without corresponding published cloud')
            latency.append(r['received_monotonic']-c['published_monotonic'])
        if not np.isfinite(latency).all() or min(latency)<0: raise ValueError('Invalid wall latency')
        report['latency']=dict(sim_lag_max_s=float(lag.max()),sim_lag_p95_s=float(np.percentile(lag,95)),publication_to_receipt_wall_max_s=max(latency),publication_to_receipt_wall_p95_s=float(np.percentile(latency,95)),samples=len(latency))
        checks.update(lio_translation_rmse=m['translation_rmse_m']<=.20,lio_translation_max=m['translation_max_m']<=.40,lio_orientation_max=m['rotation_max_deg']<=10,lio_post_initialization_coverage=m['post_initialization_scan_coverage']>=.90,lio_freshness=health['freshness_pass'])
    section('estimator',estimator)
    def map_check():
        report['native_map']=pcd_audit(run/'fastlio_map.pcd');checks['native_map_saved_finite_nonempty']=manifest.get('native_map_saved') is True
        rows=json_lines(run/'lio/registered_index.jsonl');count=0
        for r in rows:
            p=run/'lio'/r['file']
            if not p.resolve().is_relative_to((run/'lio').resolve()): raise ValueError('Unsafe registered cloud path')
            with np.load(p,allow_pickle=False) as z:
                xyz=z['xyz']
                if xyz.shape!=(r['points'],3) or not np.isfinite(xyz).all(): raise ValueError('Invalid registered cloud')
                count+=len(xyz)
        stamps(rows);report['registered_clouds']=dict(files=len(rows),points=count)
        b=json_file(run/'lio/bridge_result.json')['counts']
        checks['registered_clouds_retained']=len(rows)==b['registered_clouds'] and count==b['registered_points'] and count>0
    section('native_map',map_check)
    def physical():
        report['physics'],c=trace_audit(run,manifest,events,metadata)
        if manifest.get('skill_fixture_contract'):
            # A ground skill cannot satisfy the historical upright-only rule.
            # Replace that one rule only with independently checked ownership,
            # full-rate limits, actual traversal/recovery and upright state
            # outside both declared skill intervals. Every LIO/map/scan/return,
            # timing and recording criterion above remains mandatory.
            if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
            from punktnull.mission_skill_evidence import evaluate as evaluate_skills
            report['integrated_crawl']=evaluate_skills(run)
            report['physics']['historical_upright_only_diagnostic']=c.pop('bounded_upright_physics')
            c['authorized_crawl_and_upright_physics']=report['integrated_crawl']['accepted']
        checks.update(c)
    section('physics',physical)
    def scans():
        if 'inspection_plan' in manifest:
            report['room_scans']=room_scan_audit(run,manifest,events,metadata)
            checks['declared_room_scans_physically_verified']=report['room_scans']['pass_all']
        else:
            report['room_scans']={'status':'NO_DECLARED_INSPECTION_PLAN_OR_THREE_ROOM_SCAN_RECORD; physical room entry checked separately'}
    section('room_scans',scans)
    def video():
        recording=json_file(run/'sim/recording.json');ts=json_lines(run/'sim/video_timestamps.jsonl')
        decoded=decode_video(run/'sim/uncut_runtime.mp4',ffmpeg)
        times=np.asarray([r['sim_time'] for r in ts],float);walls=np.asarray([r['wall_monotonic'] for r in ts],float)
        if len(times)<2 or not np.isfinite(times).all() or not np.isfinite(walls).all() or np.any(np.diff(times)<=0) or np.any(np.diff(walls)<=0): raise ValueError('Invalid video timestamps')
        end=next(e['monotonic'] for e in events if e.get('event')=='mission_result')
        report['recording']=dict(metadata=recording,decode=decoded,timestamps=len(ts),sim_start_s=float(times[0]),sim_end_s=float(times[-1]),disclosure='Uncut captured frames at declared fixed fps keyed to simulation cadence, not guaranteed 1:1 wall time; original wall/sim timestamps retained.')
        checks['recording_complete_decodable']=bool(not recording.get('error') and recording.get('uncut') is True and recording.get('rendered_from_running_physx') is True and recording['frames']==decoded['decoded_frames']==len(ts) and walls[0]<=manifest['mission_started_monotonic'] and walls[-1]>=end and np.max(np.abs(np.diff(times)-1/float(recording['fps'])))<1e-5)
    section('recording',video)
    for name in ['manifest.json','events.jsonl','localization_origin.json','sim/trace.npz','sim/metadata.json','sim/mid360/evaluation_truth.jsonl','sim/mid360/events.jsonl','sim/mid360/metadata.json','lio/odometry.jsonl','lio/bridge_result.json','lio/publication_index.jsonl','lio/registered_index.jsonl','fastlio_map.pcd','sim/recording.json','sim/video_timestamps.jsonl']:
        if (run/name).is_file():report['input_sha256'][name]=digest(run/name)
    lio_names=['sensor_transport_integrity','runtime_origin_manifest_matches','runtime_origin_matches_first_truth','lio_translation_rmse','lio_translation_max','lio_orientation_max','lio_post_initialization_coverage','lio_freshness','native_odometry_count_matches_bridge','native_map_saved_finite_nonempty','registered_clouds_retained','recorded_source_snapshots_match']
    report['sensor_lio_integration_pass']=all(checks.get(k,False) for k in lio_names)
    report['failed_checks']=[k for k,v in checks.items() if not v]
    report['audit_pass']=bool(checks) and all(checks.values()) and not report['errors']
    report['verdict']='PASS_BOUNDED_G1_LIO_AUDIT' if report['audit_pass'] else 'FAILED_OR_INCOMPLETE_BOUNDED_G1_LIO_AUDIT'
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--ffmpeg',type=Path)
    a=p.parse_args();r=evaluate(a.run,a.output,a.ffmpeg)
    print(json.dumps({k:r[k] for k in ('verdict','sensor_lio_integration_pass','failed_checks','errors')},indent=2))
    return 0 if r['audit_pass'] else 1

if __name__=='__main__': raise SystemExit(main())
