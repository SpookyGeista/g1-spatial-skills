#!/usr/bin/env python3
"""Produce a NEW time-warped crawl candidate; no physical execution/control edit.

At source frames800..839, source-time speed smoothly falls from1 to.5.
The later get-up follows the same piecewise-linear root/joint path at half
source-time speed, with quaternion SLERP. All50Hz FK/velocities are rebuilt.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation,Slerp
from punktnull.skills.reference import FPS,ReferenceClip,write_skill_reference
from punktnull.skills.legacy import ground_module
from punktnull.skills.limits import load_joint_limits
from punktnull.skills.session import SkillSession,REGISTRY


def source_mapping(last_frame,start=800,end=839,final_rate=.5):
    if not (0<start<end<last_frame and .5<=final_rate<1):raise ValueError('Invalid bounded get-up timewarp')
    # The rate is a function of source-frame coordinate. Integrate dt/ds=1/r,
    # then invert. A20k grid makes the deterministic inversion error negligible
    # relative to one50Hz source frame; compare a doubled grid below.
    def build(n):
        source=np.linspace(start,end,n+1);u=(source-start)/(end-start)
        smooth=u*u*u*(10+u*(-15+6*u));rate=1-(1-final_rate)*smooth
        output=start+np.r_[0,cumulative_trapezoid(1/rate,source)]
        duration=float(output[-1]+(last_frame-end)/final_rate)
        t=np.arange(math.ceil(duration)+1,dtype=float)
        s=np.where(t<=start,t,np.where(t>=output[-1],end+(t-output[-1])*final_rate,np.interp(t,output,source)))
        return t,np.minimum(s,last_frame),duration
    t,s,duration=build(20000);_,fine,fine_duration=build(40000)
    if np.max(abs(s-fine))>1e-7 or abs(duration-fine_duration)>1e-7:raise ValueError('Timewarp integration convergence failed')
    if not (np.all(np.diff(s)>0) and np.all(np.diff(s)<=1+1e-10) and s[0]==0 and s[-1]==last_frame):
        raise ValueError('Timewarp must advance continuously through exact endpoints')
    return t/FPS,s,dict(source_transition_frames=[start,end],source_rate_start=1.,source_rate_final=final_rate,
        interpolation='Invert integral(dt/dsource_frame)=1/rate(source_frame); quintic smoothstep rate ramp',
        integrated_duration_s=duration/FPS,exported_duration_s=float(t[-1]/FPS),
        endpoint_rounding_s=float((t[-1]-duration)/FPS),quadrature_refinement_max_source_frame_error=float(np.max(abs(s-fine))))


def generate(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    clip=ReferenceClip.load(source);SkillSession(clip,experimental=True,allow_generated=True)
    parent_manifest=json.loads((source/'generated_reference.json').read_text())
    parent_hashes={str(source/name):sha for name,sha in clip.hashes.items()}
    baseline=ReferenceClip.load(REGISTRY['crawl'].source)
    sources={**parent_manifest['sources_sha256'],**parent_hashes}
    for name,digest in baseline.hashes.items():
        path=str(baseline.source/name)
        if sources.get(path)!=digest:raise ValueError('Missing unchanged original crawl source '+name)
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out_seconds,frame,warp=source_mapping(clip.frames-1)
    old_frames=np.arange(clip.frames,dtype=float);perm=ground_module().MJ_TO_IL
    q0=clip.data['joint_pos'][:,perm]
    q=np.column_stack([np.interp(frame,old_frames,q0[:,i]) for i in range(29)])
    xyz=np.column_stack([np.interp(frame,old_frames,clip.root[:,i]) for i in range(3)])
    quat=Slerp(old_frames,Rotation.from_quat(clip.quaternion[:,[1,2,3,0]]))(frame).as_quat()[:,[3,0,1,2]]
    mapping=np.c_[np.arange(len(frame)),out_seconds,frame,frame/FPS]
    stream=io.StringIO();np.savetxt(stream,mapping,delimiter=',',fmt='%.12f',
        header='output_frame,output_time_s,source_frame,source_time_s',comments='')
    mapping_bytes=stream.getvalue().encode();mapping_hash=hashlib.sha256(mapping_bytes).hexdigest()
    change=dict(operation='Smooth source-time slowdown of get-up; resampling only',**warp,
        parent_source=str(source),parent_ordered_channel_digest=clip.digest,
        mapping_file='source_time_mapping.csv',mapping_sha256=mapping_hash,
        root_geometry='Same source piecewise-linear XYZ path and exact start/end coordinates; no translation, rotation, scaling or reset',
        root_orientation='SLERP along unchanged source quaternion path',joint_interpolation='Linear per SDK joint, inherited pitch margin unchanged',
        channels='Rebuild all body FK, joint velocities, body linear/angular velocities at50Hz with retained exporter',
        controller_changed=False,physical_limits_changed=False,physical_execution=False,
        reason='Reduce get-up impact/momentum after active pitch/roll stop crossings; actual improvement unproven')
    new=write_skill_reference(output,xyz,quat,q,sources,[*parent_manifest['modifications'],change],skill='crawl')
    (output/'source_time_mapping.csv').write_bytes(mapping_bytes)
    # Check the serialized candidate, not only floating-point inputs.
    checked=ReferenceClip.load(output);SkillSession(checked,experimental=True,allow_generated=True)
    limits=load_joint_limits();actual_q=checked.data['joint_pos'][:,perm];actual_dq=checked.data['joint_vel'][:,perm]
    position_excess=np.maximum(np.maximum(limits.lower-actual_q,actual_q-limits.upper),0)
    speed_excess=np.maximum(abs(actual_dq)-limits.velocity,0)
    errors=[limits.check(qq,vv,require_effort=False) for qq,vv in zip(actual_q,actual_dq)]
    unchanged=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in parent_hashes.items())
    early=np.arange(801)
    checks=dict(parent_asset_bytes_unchanged=unchanged,full_original_source_provenance=True,
        all_serialized_position_speed_limits=not any(errors),strict_source_time_progress=bool(np.all(np.diff(frame)>0)),
        no_source_frame_skip=bool(np.max(np.diff(frame))<=1+1e-10),
        initial_801_positions_unchanged=bool(np.max(abs(actual_q[early]-q0[early]))<=1e-9 and np.max(abs(checked.root[early]-clip.root[early]))<=1e-8),
        root_start_end_unchanged=bool(np.max(abs(checked.root[[0,-1]]-clip.root[[0,-1]]))<=1e-8),
        joint_start_end_unchanged=bool(np.max(abs(actual_q[[0,-1]]-q0[[0,-1]]))<=1e-9),
        quaternion_start_end_same_rotation=bool(np.max(1-abs(np.sum(checked.quaternion[[0,-1]]*clip.quaternion[[0,-1]],axis=1)))<=1e-8),
        generated_session_admission_checked=True,mapping_hash_verified=hashlib.sha256((output/'source_time_mapping.csv').read_bytes()).hexdigest()==mapping_hash)
    report=dict(schema=1,kind='CPU source-time-warped crawl candidate generation',accepted=all(checks.values()),checks=checks,
        physical_execution=False,actual_limits_containment_unproven=True,reference=checked.provenance(),parent_reference=clip.provenance(),
        timewarp=warp,original_frames=clip.frames,generated_frames=checked.frames,original_seconds=clip.seconds,generated_seconds=checked.seconds,
        maximum_serialized_position_excess_rad=float(position_excess.max()),maximum_serialized_speed_excess_rad_s=float(speed_excess.max()),
        maximum_per_joint_speed_rad_s=np.max(abs(actual_dq),axis=0).tolist(),
        parent_asset_hashes_verified_unchanged=parent_hashes,
        limitation='Reference geometry/limits/provenance checks only; real native tracking, contacts, effort, recovery and passage clearance still require simulation.')
    (output/'generation_check.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['accepted']:raise ValueError('Serialized candidate checks failed; retained evidence in '+str(output))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'assets/crawl_ankle_margin_003')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=generate(args.source,args.output)
    print(json.dumps({k:result[k] for k in ('accepted','checks','original_frames','generated_frames','original_seconds','generated_seconds','timewarp')},indent=2))
    print('New candidate:',result['reference']['source'],'digest:',result['reference']['ordered_channel_digest'])

if __name__=='__main__':main()
