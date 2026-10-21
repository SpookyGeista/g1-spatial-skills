#!/usr/bin/env python3
"""Historical actual FK replay for rendering-only portal camera design."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from punktnull.terrain.mission_camera import PROFILES,assess_points
from punktnull.terrain.inspection_passage import MISSION_CONTRACT,collision_boxes
from punktnull.terrain.kinematics import CollisionFK
from punktnull.retained import URDF,LEGACY,load

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('New evidence directory required')
 a.output.mkdir(parents=True)
 fk=CollisionFK(URDF,load('_camera_fk_bridge',LEGACY/'bridge.py').BODY_JOINT_NAMES)
 boxes=collision_boxes();rows=[];hashes={}
 for run in ('006_passage_diagnostic','007_passage_10fps','008_passage_side'):
  path=ROOT/'runs/skills'/run/'sim/trace.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  z=np.load(path);c=list(z['columns']);d=z['data'][::20];get=lambda ns:d[:,[c.index(n) for n in ns]]
  root=get(['root_'+str(i) for i in range(7)]);bounds=fk.bounds(fk.links(root[:,:3],root[:,3:],get(['q_'+str(i) for i in range(29)])))
  centers=bounds.mean(axis=2)
  corners=np.stack([np.stack([bounds[:,:,i,0],bounds[:,:,j,1],bounds[:,:,k,2]],axis=-1)
      for i in (0,1) for j in (0,1) for k in (0,1)],axis=2).reshape(len(bounds),-1,3)
  for direction in ('ingress','egress'):
   gate=MISSION_CONTRACT[direction];ang=np.deg2rad(gate['heading_deg']);rot=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
   points=centers.copy();points[:,:,:2]=points[:,:,:2]@rot.T+gate['approach_xy_m']
   edges=corners.copy();edges[:,:,:2]=edges[:,:,:2]@rot.T+gate['approach_xy_m']
   # Exact measured root low; at least one body AABB crosses the roof longitudinal range.
   selected=(root[:,2]<.50)&(bounds[:,:,1,0].max(1)>=1.45)&(bounds[:,:,0,0].min(1)<=2.10)
   checks=[assess_points(**PROFILES[direction],points=p,boxes=boxes) for p in points[selected]]
   corner_checks=[assess_points(**PROFILES[direction],points=p,boxes=boxes) for p in edges[selected]]
   rows.append(dict(run=run,direction=direction,samples=int(selected.sum()),
     visible_geometry_centers=sum(q['visible_point_count'] for q in checks),total_geometry_centers=sum(q['point_count'] for q in checks),
     fully_framed_samples=sum(q['full_proxy_in_frame'] for q in checks),
     fully_framed_body_aabb_corner_samples=sum(q['full_proxy_in_frame'] for q in corner_checks),
     minimum_visible_body_aabb_corner_fraction=min(q['visible_point_count']/q['point_count'] for q in corner_checks),
     minimum_visible_center_fraction=min(q['visible_point_count']/q['point_count'] for q in checks),
     blocked_by=sorted(set(x for q in checks for x in q['blocked_by'])),
     limits='10Hz selected historical actual body collider centers, scene-AABB rays; no body self-occlusion or mesh visibility guarantee'))
 for p in (ROOT/'punktnull/terrain/mission_camera.py',Path(__file__)):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 result=dict(scope='CPU camera feasibility, not rendered validation',profiles=PROFILES,checks=rows,source_sha256=hashes)
 (a.output/'camera_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
