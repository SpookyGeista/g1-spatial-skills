#!/usr/bin/env python3
"""Isolated actual-PhysX sensor rig diagnostic, no robot controller or DDS.

2s stationary, continuous physics translation/turn segments, final2s stationary.
Velocity commands drive a synthetic gravity-disabled rigid sensor rig; measured
pose/velocity feed sensors. Actual G1 URDF mount is used, but this is NOT a G1 run.
No navigation, scene geometry or evaluation truth enters FAST-LIO sensor records.
"""
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--nominal-rate',action='store_true',help='200k rays/s; default explicit20k developmentmode')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
args.device='cpu'
output=args.output.resolve()
if output.exists() or ROOT not in output.parents:parser.error('Choose a NEW directory under punktNULL')
output.mkdir(parents=True)
app=AppLauncher(args).app
try:
    import numpy as np
    import torch
    from pxr import UsdGeom
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject,RigidObjectCfg
    from punktnull.sensors.mid360 import SensorConfig,SimMid360,PhysxRaycaster,read_mid360_mount,save_scan,rotation_wxyz
    torch.set_num_threads(1)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,render_interval=20,device='cpu',enable_scene_query_support=True))
    stage=omni.usd.get_context().get_stage()
    if abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)>1e-8:raise RuntimeError('PhysX sensor requires metre stage units')
    walls=[('Floor',(0,0,-.1),(8,6,.2)),('West',(-4,0,1.5),(.15,6,3)),('East',(4,0,1.5),(.15,6,3)),
           ('South',(0,-3,1.5),(8,.15,3)),('North',(0,3,1.5),(8,.15,3)),
           ('Cabinet',(2,1.5,.85),(.65,.6,1.7)),('Column',(-2,-1,1.4),(.45,.65,2.8)),
           ('Lowbench',(-1,2,.45),(1.5,.5,.9))]
    for name,position,size in walls:
        cfg=sim_utils.CuboidCfg(size=size,collision_props=sim_utils.CollisionPropertiesCfg(),
                               visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.3,.4,.5)))
        cfg.func('/World/'+name,cfg,translation=position)
    rig=RigidObject(RigidObjectCfg(prim_path='/World/SensorRig',
        spawn=sim_utils.CuboidCfg(size=(.1,.1,.1),mass_props=sim_utils.MassPropertiesCfg(mass=1.),
          rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,linear_damping=0.,angular_damping=0.),
          collision_props=sim_utils.CollisionPropertiesCfg()),init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,1.))))
    sim.reset();rig.update(.005)
    urdf=Path('/home/loya2/WholeBodyTele/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf')
    xyz,rpy=read_mid360_mount(urdf)
    cfg=SensorConfig(rays_per_second=200000 if args.nominal_rate else 20000,torso_to_lidar_xyz_m=xyz,torso_to_lidar_rpy_rad=rpy)
    sensor=SimMid360(PhysxRaycaster(),cfg)
    start_wall=time.monotonic();imu_count=0;cloud_count=0;all_imu=[];trajectory=[];counts=[]
    events=(output/'events.jsonl').open('x');truth=(output/'evaluation_truth.jsonl').open('x')
    def capture(t):
        global imu_count,cloud_count
        state=rig.data.root_state_w[0].cpu().numpy().astype(float)
        packet=sensor.update_physics(t,state[:7],state[7:10],state[10:13])
        # Evaluation stream is physically separate from the estimator input file.
        T=np.eye(4);T[:3,:3]=rotation_wxyz(state[3:7]);T[:3,3]=state[:3]
        wi=T@cfg.T_torso_lidar@cfg.T_lidar_imu
        truth.write(json.dumps({'stamp_ns':round(t*1e9),'torso_pose_xyz_wxyz':state[:7].tolist(),
                               'imu_to_world':wi.tolist(),'source':'actual PhysX rigid-body state; EVALUATION ONLY'})+'\n')
        trajectory.append(state[:7].tolist())
        if packet['imu'] is not None:
            im=packet['imu'];im['kind']='imu';im['stamp_ns']=im.pop('timestamp_ns')
            events.write(json.dumps(im)+'\n');all_imu.append(im);imu_count+=1
        for cloud in packet['scans']:
            filename=f'scan_{cloud_count:06d}.npz';save_scan(output/filename,cloud)
            events.write(json.dumps({'kind':'cloud','stamp_ns':cloud['scanstart_ns'],
                                    'scanstart_ns':cloud['scanstart_ns'],'file':filename,'frame_id':cloud['frame_id'],
                                    'complete':cloud['complete']})+'\n')
            counts.append(len(cloud['xyz_sensor_m']));cloud_count+=1
    capture(0.)
    for i in range(1,2001):
        t=(i-1)*.005
        if t<2 or t>=8:v=(0,0,0,0,0,0)
        elif t<4:v=(.25,0,0,0,0,0)
        elif t<6:v=(.15,.12,0,0,0,.35)
        else:v=(-.20,.10,0,0,0,-.35)
        rig.write_root_velocity_to_sim(torch.tensor([v],dtype=torch.float32,device='cpu'))
        rig.write_data_to_sim();sim.step(render=False);rig.update(.005);capture(i*.005)
    events.close();truth.close()
    stage.Export(str(output/'sensor_rig.usda'))
    stationary=np.array([r['specific_force_m_s2'] for r in all_imu if r['stamp_ns']<=1950000000])
    mean_force=stationary.mean(0)
    expected=cfg.T_torso_lidar[:3,:3].T@np.array([0,0,9.81])
    displacement=float(np.linalg.norm(np.asarray(trajectory[-1])[:3]-np.asarray(trajectory[0])[:3]))
    summary={'scope':'Actual PhysX scene-query and synthetic moving rigid sensor rig; not fullG1 or realLivox validation',
      'simulation_seconds':10.,'elapsed_wall_seconds':time.monotonic()-start_wall,'imu_records':imu_count,'scan_records':cloud_count,
      'returns_per_scan_min_max':[min(counts),max(counts)],'stationary_specific_force_mean':mean_force.tolist(),
      'stationary_expected_specific_force':expected.tolist(),'stationary_error_m_s2':float(np.linalg.norm(mean_force-expected)),
      'measured_final_displacement_m':displacement,'sensor':sensor.metadata(),
      'actual_imported_urdf':str(urdf),'urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest(),
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'punktnull/sensors/mid360.py']},
      'fixture':{'model':'gravity-disabled1kg rigidcube; velocitycommands integratedbyPhysX; no pose teleports',
                 'sim_gravity_world':[0,0,-9.81],'walls':walls,'stated_motion':'hold0-2s;translate2-4s;translate+turn4-6s;translate+reverse turn6-8s;hold8-10s'},
      'checks':{'imu200Hz':imu_count==2000,'scans10Hz':cloud_count==100,'returns_present':min(counts)>500,
                'inverted_stationary_gravity_correct':float(np.linalg.norm(mean_force-expected))<1e-4,'measured_motion':displacement>.5},
      'not_proven':['FAST-LIO estimator integration','G1 self-occlusion/contact dynamics','Livox calibrated optics/noise','wall-time200k throughput','physical hardware']}
    with (output/'smoke_summary.json').open('x') as f:json.dump(summary,f,indent=2)
    print('SENSOR_SMOKE_RESULT',json.dumps({k:summary[k] for k in ('imu_records','scan_records','checks','elapsed_wall_seconds')}),flush=True)
    if not all(summary['checks'].values()):raise RuntimeError('Sensor smoke check failed; retain output')
finally:
    app.close(wait_for_replicator=False)
