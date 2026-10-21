#!/usr/bin/env python3
"""CPU USD construction check; no Kit, renderer, PhysX or robot is launched.

Only omni.usd.get_context().get_stage is replaced with an in-memory USD stage.
Real installed USD APIs execute both retained and new procedural builders.
"""
import argparse,hashlib,json,os,sys,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pxr import Usd,UsdGeom,UsdPhysics
from punktnull.terrain import inspection_passage as scene

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Use new evidence directory')
 a.output.mkdir(parents=True)
 omni=types.ModuleType('omni');omni.usd=types.ModuleType('omni.usd')
 sys.modules['omni']=omni;sys.modules['omni.usd']=omni.usd
 rows=[]
 for variant in (0,1,2):
  for room in ('west','east','north'):
   for present in (False,True):
    os.environ.update(NULL_SCENE_VARIANT=str(variant),NULL_TARGET_ROOM=room,NULL_TARGET_PRESENT=str(int(present)))
    stage=Usd.Stage.CreateInMemory();omni.usd.get_context=lambda:types.SimpleNamespace(get_stage=lambda:stage)
    meta=scene.build_scene();cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    errors=[]
    for box in meta['collision_boxes']:
     prim=stage.GetPrimAtPath(box['prim'])
     if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):errors.append({'prim':box['prim'],'error':'missing actual USD collider'});continue
     b=cache.ComputeWorldBound(prim).ComputeAlignedRange()
     delta=max(abs(np.asarray(b.GetMin())-box['aabb_min']).max(),abs(np.asarray(b.GetMax())-box['aabb_max']).max())
     if delta>1e-7:errors.append({'prim':box['prim'],'max_bound_error_m':float(delta)})
    actual={str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    expected={x['prim'] for x in meta['collision_boxes']}|{'/World/Course/Floor'}
    if actual!=expected:errors.append({'unexpected':sorted(actual-expected),'missing':sorted(expected-actual)})
    rows.append(dict(variant=variant,target_room=room,target_present=present,expected_colliders=len(expected),
      actual_colliders=len(actual),errors=errors,passed=not errors))
 result=dict(scope='Actual CPU USD stage construction, not PhysX or rendered validation',
  stage_context='Only omni.usd stage lookup is stubbed; real installed USD cube/xform/material/collision APIs',
  usd_version=list(Usd.GetVersion()),scene_source_sha256=hashlib.sha256(Path(scene.__file__).read_bytes()).hexdigest(),
  checks=rows,passed=all(r['passed'] for r in rows))
 (a.output/'usd_construction.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
