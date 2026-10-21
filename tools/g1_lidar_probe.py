#!/usr/bin/env python3
"""Actual loaded G1 sensor/collider diagnostic, no DDS or native controller."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.device='cpu'
if a.output.exists():p.error('New output required')
a.output.mkdir(parents=True)
app=AppLauncher(a).app
try:
    import numpy as np
    import torch
    import omni.usd,omni.physx
    from pxr import Usd,UsdGeom,UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from punktnull.retained import REPO,EXPLORATION,load,prepare
    from punktnull.sensors.mid360 import SensorConfig,SimMid360,PhysxRaycaster,rotation_wxyz
    prepare();torch.set_num_threads(1)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device='cpu',enable_scene_query_support=True))
    scene=load('_g1_probe_scene',EXPLORATION/'scene.py');scene.build_scene(hall=False,course=True)
    cfgmod=load('_g1_probe_cfg',REPO/'gear_sonic/envs/manager_env/robots/g1.py')
    cfg=cfgmod.G1_CYLINDER_MODEL_12_DEX_CFG.copy();cfg.prim_path='/World/G1'
    cfg.spawn.asset_path=str(REPO/cfg.spawn.asset_path)
    cfg.spawn.usd_dir=str(REPO/'data/isaac_sonic_multicam_2026-09-16/robot_cache');cfg.spawn.usd_file_name='sonic_training_g1.usd'
    cfg.init_state.pos=(0,0,1.)
    robot=Articulation(cfg);sim.reset();robot.update(.005)
    torso=robot.data.body_link_state_w[0,list(robot.body_names).index('torso_link')].cpu().numpy()
    c=SensorConfig();origin=torso[:3]+rotation_wxyz(torso[3:7])@np.array(c.torso_to_lidar_xyz_m)
    query=omni.physx.get_physx_scene_query_interface();hits=[]
    for d in [(1,0,0),(0,1,0),(0,0,1),(0,0,-1),(.8,0,-.6)]:
        hit=query.raycast_closest(tuple(map(float,origin)),d,40.);hits.append({k:str(v) if not isinstance(v,(str,int,float,bool,tuple,list)) else v for k,v in hit.items()})
    stage=omni.usd.get_context().get_stage();cache=UsdGeom.XformCache();colliders=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/G1'),Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and 'torso' in str(prim.GetPath()):
            mat=np.asarray(cache.GetLocalToWorldTransform(prim)).T
            colliders.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),T_world=mat.tolist(),
                attributes={key:str(prim.GetAttribute(key).Get()) for key in ('radius','height','axis','extent')}))
    result=dict(origin=origin.tolist(),hits=hits,torso_colliders=colliders)
    (a.output/'probe.json').write_text(json.dumps(result,indent=2));print('G1_PROBE',json.dumps(result),flush=True)
finally:
    app.close(wait_for_replicator=False)
