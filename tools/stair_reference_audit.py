#!/usr/bin/env python3
"""CPU geometric proposal audit; no simulator or execution claim."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from punktnull.skills.reference import ReferenceClip
from punktnull.skills.stairs import URDF
from punktnull.skills.limits import load_joint_limits
from punktnull.skills.legacy import OLD,load
from punktnull.terrain.kinematics import CollisionFK
from punktnull.terrain.stairs import collision_boxes

def audit(path):
 c=ReferenceClip.load(path);L=load_joint_limits();cv=load('stair_audit_converter',OLD/'tools/convert_bones_to_sonic_reference.py');fk=CollisionFK(URDF,L.names)
 poses=fk.links(c.root,c.quaternion,c.data['joint_pos'][:,cv.MJ_TO_IL]);rows=[]
 for box in collision_boxes():
  low=np.array(box['aabb_min']);high=np.array(box['aabb_max'])
  for g in fk.collisions:
   if g['kind'] not in ('cylinder','sphere'):continue
   transform=poses[g['link']]@g['origin'];radius=g['size'][0];half=g['size'][1]/2 if g['kind']=='cylinder' else 0
   minimum=float('inf');index=None
   for i,t in enumerate(transform):
    start=t[:3,3]-t[:3,2]*half;end=t[:3,3]+t[:3,2]*half
    def d2(u):
     p=start+u*(end-start);return np.sum((p-np.clip(p,low,high))**2)
    result=minimize_scalar(d2,bounds=(0,1),method='bounded')
    gap=min(result.fun,d2(0),d2(1))**.5-radius
    if gap<minimum:minimum=float(gap);index=i
   if minimum<.04:rows.append(dict(prim=box['prim'],link=g['link'],geometry_index=g['index'],minimum_capsule_box_gap_m=minimum,frame=index,time_s=index/50))
 return dict(schema=1,physical_execution=False,reference=c.provenance(),
  exact_shape_scope='Capsule/sphere vs axis-aligned static box; actual URDF cylinder-to-capsule importer. Meshes retain earlier conservative AABB.',
  close_geometry=rows,reference_target_only=True,
  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),URDF,ROOT/'punktnull/terrain/stairs.py')})
if __name__=='__main__':
 if len(sys.argv)!=3:raise SystemExit('stair_reference_audit.py REFERENCE NEW_OUTPUT.json')
 out=Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(audit(sys.argv[1]),indent=2)+'\n');print(out)
