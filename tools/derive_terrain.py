#!/usr/bin/env python3
"""Derive proposed low-passage bounds from retained actual G1 reference geometry."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from punktnull.retained import prepare,URDF,REPO
prepare()
from bridge import BODY_JOINT_NAMES
from punktnull.terrain.kinematics import CollisionFK
from punktnull.skills.reference import ReferenceClip
from punktnull.skills.legacy import ground_module

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--trace',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('New output required')
    a.output.mkdir(parents=True)
    source=REPO/'data/bones_seed_crawl_extension_2026-09-16/02_crawl_only'
    clip=ReferenceClip.load(source);ref=ground_module().Reference(source)
    fk=CollisionFK(URDF,BODY_JOINT_NAMES)
    poses=fk.links(ref.xyz,ref.quat,ref.joints);bounds=fk.bounds(poses)
    def candidates(b):
        result=[]
        for x in np.arange(.7,2.,.1):
            cross=(b[:,:,0,0]<=x+.8)&(b[:,:,1,0]>=x)
            if not cross.any():continue
            top=float(b[:,:,1,2][cross].max());width=float(np.max(np.abs(b[:,:,:,1][np.repeat(cross[:,:,None],2,axis=2)]))*2)
            result.append(dict(x0=float(x),x1=float(x+.8),maximum_intersecting_geometry_z_m=top,
                               occupied_lateral_width_m=width,required_ceiling_with_10cm_margin_m=top+.10))
        return result
    report=dict(scope='Geometric proposal only; no physical passage validation',urdf=str(URDF),
        urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),reference=clip.provenance(),
        collision_model='All URDF primitive collisions, cylinders replaced by capsules as actual Isaac import; conservative AABBs',
        collision_count=len(fk.collisions),standing_frame0_height_m=float(bounds[0,:,1,2].max()-bounds[0,:,0,2].min()),
        reference_candidates=candidates(bounds))
    if a.trace:
        trace=np.load(a.trace,allow_pickle=False)
        names=[str(x) for x in trace['columns']] if 'columns' in trace else []
        data=trace['data'];columns=lambda keys:data[:,[names.index(k) for k in keys]]
        measured=fk.bounds(fk.links(columns(['root_'+str(i) for i in range(3)]),
            columns(['root_'+str(i) for i in range(3,7)]),columns(['q_'+str(i) for i in range(29)])))
        report['measured_trace']=str(a.trace);report['measured_trace_sha256']=hashlib.sha256(a.trace.read_bytes()).hexdigest()
        report['measured_candidates']=candidates(measured)
        report['measured_all_physics_samples']=len(data)
        report['limitation']='Swept AABBs on prior flat-floor measured motion bound a proposed fixture; only a new collision-active trial validates passage.'
        np.savez_compressed(a.output/'measured_bounds.npz',bounds=measured,root=columns(['root_'+str(i) for i in range(3)]))
    np.savez_compressed(a.output/'reference_bounds.npz',bounds=bounds,root=ref.xyz)
    (a.output/'geometry.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
