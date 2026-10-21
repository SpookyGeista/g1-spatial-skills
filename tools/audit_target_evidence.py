#!/usr/bin/env python3
"""Re-decode the retained marker images and recompute their FAST-LIO locations.

This reuses the existing tested marker detector, but reads the archived RGB-D
and native odometry again; it does not trust a mission's success flag. It is not
an independent vision algorithm, general object recognition or scene truth.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from punktnull.retained import prepare,URDF
from punktnull.localization import PoseHistory,RobotFrames
prepare()
from perception import CaseDetector
from cameras import HEAD_EYE_IN_TORSO


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(run):
    observation=json.loads((run/'target_observation.json').read_text())
    frame_ids=observation['confirming_frames'];times=observation['confirming_sim_times']
    if (len(frame_ids)<3 or len(frame_ids)!=len(times) or len(set(frame_ids))!=len(frame_ids)
            or np.any(np.diff(frame_ids)<=0) or np.any(np.diff(times)<=0)):
        raise ValueError('Distinct chronological confirming frames required')
    registration=json.loads((run/'localization_origin.json').read_text())
    history=PoseHistory(registration['T_map_initial_imu'])
    for line in (run/'lio/odometry.jsonl').read_text().splitlines():
        record=json.loads(line)
        if times[0]-.3<=record['stamp_ns']/1e9<=times[-1]+.3:history.append(record)
    T_TI=RobotFrames(URDF).T_TI
    pitch=math.radians(10);forward=np.array([math.cos(pitch),0.,-math.sin(pitch)])
    right=np.array([0.,-1.,0.]);T_TC=np.eye(4)
    T_TC[:3,:3]=np.column_stack((right,np.cross(forward,right),forward))
    T_TC[:3,3]=HEAD_EYE_IN_TORSO
    detector=CaseDetector(7);frames=[];last_rgb=None
    inputs=[run/'target_observation.json',run/'target_observation.png',run/'lio/odometry.jsonl',
            run/'localization_origin.json']
    for frame_id,stamp in zip(frame_ids,times):
        archive=run/'sim/sensors'/f'frame_{frame_id:06d}.npz'
        with np.load(archive,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata'].item()))
            rgb=data['rgb'];depth=data['depth'];last_rgb=rgb.copy()
            if metadata['frame_id']!=frame_id or abs(metadata['sim_time']-stamp)>1e-8:
                raise ValueError('Image identity/time mismatch')
            T_MI,age=history.at(round(stamp*1e9),max_age_ns=200_000_000,require_bracket=True)
            found=detector.observe(rgb,depth,metadata['intrinsics'],T_MI@np.linalg.inv(T_TI)@T_TC,frame_id,stamp)
            candidates=[f for f in found if f['target_id']==7]
            frames.append(dict(frame_id=frame_id,stamp=stamp,marker_7_decoded=bool(candidates),
                               lio_bracketed=True,candidates=candidates,archive_sha256=digest(archive)))
            inputs.append(archive)
    decoded=detector.confirmed
    report_image=cv2.imread(str(run/'target_observation.png'))
    image_matches=bool(report_image is not None and np.array_equal(cv2.cvtColor(report_image,cv2.COLOR_BGR2RGB),last_rgb))
    error=float(np.linalg.norm(np.asarray(decoded['position_m'])-observation['position_m'])) if decoded else None
    corners_error=float(np.max(np.abs(np.asarray(decoded['corners_px'])-observation['corners_px']))) if decoded else None
    checks=dict(three_archived_images_decode_marker_7=all(f['marker_7_decoded'] for f in frames),
                actual_multi_frame_confirmation=decoded is not None,
                saved_inspection_image_matches_last_raw_rgb=image_matches,
                declared_identity_matches=observation.get('target_id')==7,
                lio_reprojection_matches_recorded_position=error is not None and error<=1e-5,
                decoded_corners_match_record= corners_error is not None and corners_error<=1e-4)
    return dict(pass_all=all(checks.values()),checks=checks,frames=frames,
                recomputed_position_m=decoded['position_m'] if decoded else None,
                reported_position_m=observation['position_m'],reprojection_difference_m=error,
                source_sha256={str(p.relative_to(run)):digest(p) for p in inputs},
                evaluator_sha256=digest(Path(__file__)),
                scope='Archived RGB-D redetection with the retained detector and native FAST-LIO pose reprojection. Identifies marker7 surface, not arbitrary bags, package contents or metric map accuracy.',
                simulator_camera_pose_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=args.output or args.run/'target_evidence_audit.json'
    if output.exists():parser.error('Refusing to overwrite retained report')
    cv2.setNumThreads(1)
    try:report=evaluate(args.run)
    except Exception as exc:report=dict(pass_all=False,error=repr(exc))
    with output.open('x') as file:json.dump(report,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps(dict(output=str(output),pass_all=report['pass_all'],error=report.get('error'))))
    return 0 if report['pass_all'] else 2

if __name__=='__main__':raise SystemExit(main())
