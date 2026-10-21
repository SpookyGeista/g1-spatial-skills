#!/usr/bin/env python3
"""Add FAST-LIO localization to the retained observed-space inspection mission."""
from __future__ import annotations
import json
import argparse
import hashlib
import inspect
import textwrap
import math
import os
from pathlib import Path
import sys
import subprocess
import shutil
import time
import numpy as np
from scipy.spatial.transform import Rotation
from punktnull.retained import ROOT,EXPLORATION,URDF,load,prepare
from punktnull.localization import RobotFrames,PoseHistory
from punktnull.inspection import InspectionExplorer,LioOccupancyGrid
from punktnull.mission_skills import CrawlMissionMixin,CrawlInspectionExplorer
prepare()
old=load('_punktnull_retained_mission',EXPLORATION/'run_mission.py')
old.HERE=ROOT
old.Explorer=InspectionExplorer
old.OccupancyGrid=LioOccupancyGrid
# Keep the existing scan/controller implementation, but add measured rotation
# overlap before stopping. The native physical audit still requires 2*pi-.10.
# FAST-LIO can differ at the two endpoints; the old stop threshold provided no
# error margin and run012 missed that physical threshold by 0.073 degrees.
_scan_original=textwrap.dedent(inspect.getsource(old.Mission.turn_scan))
_scan_before='self.scan_total>=2*math.pi-.10'
_scan_after='self.scan_total>=2*math.pi+.25'
if _scan_original.count(_scan_before)!=1:raise RuntimeError('Retained scan implementation changed; review completion margin')
_scan_source=_scan_original.replace(_scan_before,_scan_after)
_scan_namespace=dict(old.__dict__)
exec(compile(_scan_source,str(EXPLORATION/'run_mission.py')+':disclosed_scan_overlap','exec'),_scan_namespace)
_retained_scan_with_overlap=_scan_namespace['turn_scan']

# Extend two bounded seams in the retained launcher/mission loop. Native
# command ownership and the observed-space walk implementation stay reused.
_launch_original=textwrap.dedent(inspect.getsource(old.Mission.launch))
_launch_source=_launch_original
for before,after in [
    ('    self.sim=ProcessLog(sim_command,', '    self.prepare_skill_simulator(sim_command,sim_env)\n    self.sim=ProcessLog(sim_command,'),
    ('self.explorer=Explorer(', 'self.explorer=self.make_explorer('),
    ("    self.manifest['controller_command']=command;", "    self.prepare_skill_controller(command)\n    self.manifest['controller_command']=command;")]:
    if _launch_source.count(before)!=1:raise RuntimeError('Retained launcher seam changed')
    _launch_source=_launch_source.replace(before,after)
_launch_namespace=dict(old.__dict__)
exec(compile(_launch_source,str(EXPLORATION/'run_mission.py')+':skill_launch','exec'),_launch_namespace)
old.Mission.launch=_launch_namespace['launch']
_explore_original=textwrap.dedent(inspect.getsource(old.Mission.explore))
_explore_needle="        if command.status=='SCAN_REQUIRED':self.turn_scan();continue"
if _explore_original.count(_explore_needle)!=1:raise RuntimeError('Retained mission dispatch changed')
_explore_source=_explore_original.replace(_explore_needle,
    "        if command.status=='CRAWL_REQUIRED':self.run_crawl_gate();last_failure=None;continue\n"+_explore_needle)
_explore_namespace=dict(old.__dict__)
exec(compile(_explore_source,str(EXPLORATION/'run_mission.py')+':skill_dispatch','exec'),_explore_namespace)
old.Mission.explore=_explore_namespace['explore']
from cameras import HEAD_EYE_IN_TORSO
from bridge import BODY_JOINT_NAMES
WAIST_INDICES=[BODY_JOINT_NAMES.index(n) for n in ('waist_yaw_joint','waist_roll_joint','waist_pitch_joint')]
OPTIONS=argparse.Namespace(start_yaw_offset=0.,inject_review_after_room=None,lidar_points_per_second=10000,rgb_lidar=False,immersive=False,
                           terrain_skills='none',crawl_reference=None,skill_preflight=None,ankle_roll_target_margin_rad=0.,limit_ankle_pitch=False,experimental_ankle_braking=False)

class ScanReplanRequested(Exception):
    """A measured stop outside the scan zone requires another observed approach."""

class LioMission(CrawlMissionMixin,old.Mission):
    def make_explorer(self,*args,**kwargs):
        # The source-checked retained launcher has a copied globals dictionary.
        # Resolve the selected backend from this mission, never a stale global.
        factory=CrawlInspectionExplorer if self.crawl_enabled else InspectionExplorer
        return factory(*args,**kwargs)

    def heartbeat(self):
        # Publish base and extension fields together. Two successive atomic
        # writes exposed the old labels to readers between those writes.
        status={'monotonic':time.monotonic(),'state':self.phase,
            'owner':self.owner,'detail':self.detail,'elapsed':time.monotonic()-self.started,
            'detected':bool(self.detector.confirmed),'sensor_frames':self.sensor_count,
            'target_position_m':self.detector.confirmed['position_m'] if self.detector.confirmed else None,
            'map_observed_cells':int(self.grid.seen.sum()),'test':self.args.test}
        status['injected_review_after_room']=OPTIONS.inject_review_after_room
        status['pose_input_kind']=('recorded_pose_replay' if (self.output/'mission_review_replay/source.json').is_file()
                                   else 'external_source_not_verified_as_live_human')
        status['lidar_rays_per_second']=OPTIONS.lidar_points_per_second
        status['autonomous_skill']=getattr(self,'skill_gate',None)
        old.atomic_json(self.output/'status.json',status)

    def transition(self,phase,detail,owner='autonomy / native SONIC planner'):
        if phase=='READY':detail='Auftrag: drei vorgegebene Räume scannen, markierten Koffer7 prüfen, zurückkehren'
        return super().transition(phase,detail,owner)

    def __init__(self,args):
        self.crawl_enabled=OPTIONS.terrain_skills=='crawl'
        if self.crawl_enabled:
            self.validate_skill_setup()
            from punktnull.mission_skills import configured_exit_colliders
            self.skill_collision_boxes=configured_exit_colliders(args.variant,args.target_room,args.target_absent)
        self.skill_active=False;self.skill_mapping_enabled=False;self.skill_session=None;self.skill_gate=None
        self.lio=self.lio_bridge=None;self.pose_history=None;self.odom_reader=None
        self.frames=RobotFrames(URDF);self.truth_state=None;self.last_lio_ns=-1
        self.initial_balance_only=True;self.pending_rgbd=[]
        super().__init__(args)
        self.manifest.update(localization='FAST-LIO estimated IMU pose + measured waist encoders; one-time initial simulator frame registration',
            map_source='FAST-LIO 3D LiDAR map; navigation occupancy from rendered depth transformed by FAST-LIO',
            sensor_model=f'{OPTIONS.lidar_points_per_second} rays/s MID360-like collision-ray proxy;200Hz synthetic internal IMU;5ms timestamp batches',
            terrain_skills='Known north-passage autonomous crawl gates' if self.crawl_enabled else 'Not enabled by this launch',
            original_baseline=str(EXPLORATION))
        self.manifest['initialization_balance_uses_simulator_state']=True
        self.manifest['start_yaw_offset_deg']=OPTIONS.start_yaw_offset
        self.manifest['rgb_lidar_display']=dict(enabled=OPTIONS.rgb_lidar,
            source='MID360-like PhysX rays + existing simulated torso RGB-D camera',
            pose_usage='per-ray-batch simulator transforms for display only; static scene',
            destination='shared Wholebody CloudXR session; stereo spatial window',
            real_d435i_calibration=False)
        self.manifest['immersive_display']=dict(enabled=OPTIONS.immersive,
            source='simulated RGB-D observed surface', geometry='metric depth mesh; no unseen surfaces',
            destination='OpenXR per-eye projection with shared DeviceIO tracking',
            real_hardware_calibrated=False)
        self.manifest['injected_review_after_room']=OPTIONS.inject_review_after_room
        self.manifest['injection_kind']='explicit_test_operator_review' if OPTIONS.inject_review_after_room else None
        self.manifest['inspection_plan']=json.loads((ROOT/'inspection_plan.json').read_text())
        if self.crawl_enabled:
            from punktnull.terrain.inspection_passage import MISSION_CONTRACT
            self.manifest['skill_fixture_contract']=MISSION_CONTRACT
            self.manifest['skill_online_collision_boxes']=self.skill_collision_boxes
            for room in self.manifest['inspection_plan']['rooms']:
                if room['id']=='north':room['observation_xy_m']=MISSION_CONTRACT['north_scan_xy_m']
            self.manifest['skill_selection_source']='Known fixed fixture gates; not general terrain perception'
            self.manifest['ground_rgbd_navigation_policy']='Retain raw pixels; suspend occupancy/target integration while ground skill owns robot; resume on observed standing exit'
            self.manifest['crawl_reference']=self.crawl_reference.provenance()
            self.manifest['experimental_ankle_roll_target_margin_rad']=OPTIONS.ankle_roll_target_margin_rad
            self.manifest['experimental_limit_ankle_pitch']=OPTIONS.limit_ankle_pitch
            self.manifest['experimental_ankle_braking']=OPTIONS.experimental_ankle_braking
            self.manifest['skill_geometry_preflight']=self.skill_preflight
            self.manifest['skill_geometry_preflight_path']=str(OPTIONS.skill_preflight.resolve())
            self.manifest['skill_geometry_preflight_sha256']=hashlib.sha256(OPTIONS.skill_preflight.read_bytes()).hexdigest()
            shutil.copytree(self.crawl_reference.source,self.output/'source_snapshot/crawl_reference')
            shutil.copy2(OPTIONS.skill_preflight,self.output/'skill_geometry_preflight.json')
        self.manifest['skill_integration_source_hooks']=dict(
            launch_original_sha256=hashlib.sha256(_launch_original.encode()).hexdigest(),
            launch_modified_sha256=hashlib.sha256(_launch_source.encode()).hexdigest(),
            explore_original_sha256=hashlib.sha256(_explore_original.encode()).hexdigest(),
            explore_modified_sha256=hashlib.sha256(_explore_source.encode()).hexdigest())
        self.manifest['scan_completion_overlap']=dict(estimated_stop_radians=2*math.pi+.25,
            unchanged_physical_audit_minimum_radians=2*math.pi-.10,
            source_function_sha256=hashlib.sha256(_scan_original.encode()).hexdigest(),
            exact_replacement=[_scan_before,_scan_after],
            reason='Measured turn overlap for estimator endpoint error; acceptance threshold unchanged')
        self.manifest['task']=self.manifest['inspection_plan']['task']
        self.manifest['extension_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*ROOT.glob('*.py'),*sorted((ROOT/'punktnull').rglob('*.py')),*sorted((ROOT/'tools').glob('*.py')),
                      ROOT/'ros2_ws/config/mid360_simdev.yaml',ROOT/'third_party/SOURCES.json',ROOT/'inspection_plan.json']}
        for relative in self.manifest['extension_source_sha256']:
            destination=self.output/'source_snapshot/extension'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,destination)

    def update_lio(self):
        if self.pose_history is None:
            try:
                origin=json.loads((self.output/'localization_origin.json').read_text())
                self.pose_history=PoseHistory(origin['T_map_initial_imu'])
                self.manifest['initial_localization_registration']=origin
            except (OSError,ValueError):return
        if self.odom_reader is None:
            try:self.odom_reader=(self.output/'lio/odometry.jsonl').open()
            except OSError:return
        while True:
            pos=self.odom_reader.tell();line=self.odom_reader.readline()
            if not line:break
            if not line.endswith('\n'):self.odom_reader.seek(pos);break
            record=json.loads(line);self.pose_history.append(record)
        if self.pose_history.records:self.last_lio_ns=self.pose_history.records[-1][0]

    def estimated_state(self,raw):
        T_WI,age=self.pose_history.at(round(raw['sim_time']*1e9),max_age_ns=500_000_000)
        T_WB=self.frames.pelvis(T_WI,[raw['q'][i] for i in WAIST_INDICES])
        q=Rotation.from_matrix(T_WB[:3,:3]).as_quat()[[3,0,1,2]]
        base=list(raw['base']);base[:3]=T_WB[:3,3].tolist();base[3:7]=q.tolist()
        T_WT=T_WI@np.linalg.inv(self.frames.T_TI)
        torso=list(raw['torso']);torso[:3]=T_WT[:3,3].tolist();torso[3:7]=Rotation.from_matrix(T_WT[:3,:3]).as_quat()[[3,0,1,2]].tolist()
        return dict(raw,base=base,torso=torso,yaw_deg=math.degrees(math.atan2(T_WB[1,0],T_WB[0,0])),
                    localization='FAST-LIO + measured waist FK',lio_age_ns=age)

    def tick(self,process_sensors=True):
        # Retain operator/ownership rules; separate true safety telemetry from
        # estimator-derived navigation state before any grid or route update.
        self.heartbeat()
        for path in sorted((self.output/'commands').glob('*.json')):
            try:msg=json.loads(path.read_text());path.unlink()
            except (OSError,ValueError):continue
            key=msg.get('command')
            if key=='abort':raise RuntimeError('Operator abort')
            if key not in ('start','takeover','release') or not 0<=time.monotonic()-msg.get('monotonic',0)<=2:continue
            if key=='start' and self.phase!='READY':continue
            if key=='release' and self.phase!='OPERATOR_CONTROL':continue
            if key=='takeover' and self.phase not in ('EXPLORING','SCANNING','RETURNING','ASSISTANCE_REQUESTED') and not self.phase.startswith('SKILL_'):continue
            if key not in self.commands:self.commands.append(key);self.event('operator_command',command=key,state=self.phase)
        if self.skill_active and 'takeover' in self.commands:
            raise RuntimeError('Operator override stops the autonomous ground skill; live ground-pose handover is not validated')
        for name,child in [('Isaac',self.sim),('SONIC',self.controller),('FAST-LIO',self.lio),('LIO bridge',self.lio_bridge)]:
            if child and child.process.poll() is not None:raise RuntimeError(name+' exited; inspect log')
        if self.sim and any(x in self.sim.text for x in ('ISAAC_ERROR','ISAAC_FALL','ISAAC_NONFINITE','ISAAC_MOTOR_COMMAND_WATCHDOG','ISAAC_SUPERVISOR_WATCHDOG')):
            raise RuntimeError('Simulator state/command failure; inspect sim.log')
        try:self.truth_state=json.loads((self.output/'sim_state.json').read_text())
        except (OSError,ValueError):pass
        self.update_lio()
        if self.truth_state:
            if self.pose_history and self.pose_history.records:self.state=self.estimated_state(self.truth_state)
            elif self.initial_balance_only:self.state=self.truth_state # supervised startup balance only, no route progression
            else:raise RuntimeError('FAST-LIO localization unavailable; autonomy prohibited')
        if self.released:
            if self.skill_active:
                from punktnull.skills.session import measured_error
                error=measured_error(self.truth_state,time.monotonic())
            else:error=old.state_error(self.truth_state,time.monotonic())
            if error:raise RuntimeError(error)
            if not self.initial_balance_only:
                self.grid.record_stable_pose(self.state['base'][:2],self.state['sim_time'],stable=old.stable(self.truth_state),support_free=True)
        while self.pose.poll(0):self.latest_pose=self.pose.recv();self.latest_pose_time=time.monotonic()
        if process_sensors:self.receive_sensor()
        if self.client:
            transport=self.client.status()
            if transport.get('fault') or not transport.get('worker_alive') or (transport.get('tick_age_s') is not None and transport['tick_age_s']>.25):
                raise RuntimeError('Planner transport stale/unavailable: '+str(transport))
        if time.monotonic()-self.started>self.args.max_seconds:raise RuntimeError('Overall mission timeout')
        time.sleep(.005 if self.skill_active else .02)

    def receive_sensor(self):
        newest=None
        while self.sensors.poll(0):newest=self.sensors.recv_multipart()
        if newest is not None:
            if len(newest)!=3:raise RuntimeError('Malformed RGB-D packet')
            meta=json.loads(newest[0])
            self.pending_rgbd.append((meta,newest))
            if len(self.pending_rgbd)>5:raise RuntimeError('RGB-D localization queue overrun')
        if not self.pending_rgbd:return
        meta,newest=self.pending_rgbd[0]
        if time.monotonic()-meta['monotonic']>.8:
            self.pending_rgbd.pop(0)
            if not self.initial_balance_only:raise RuntimeError('FAST-LIO did not bracket the RGB-D observation before deadline')
            return
        # FAST-LIO publishes the end of the previous scan. Applying that old
        # orientation to a new RGB-D frame caused false elevated floor returns
        # while turning. Hold pixels until their timestamp is bracketed by
        # actual estimates; never silently project with the last known pose.
        stamp_ns=round(meta['sim_time']*1e9)
        if not self.initial_balance_only and (not self.pose_history or not self.pose_history.brackets(stamp_ns)):return
        self.pending_rgbd.pop(0)
        h,w=meta['height'],meta['width']
        if (h,w)!=(360,640):raise RuntimeError('Unexpected calibrated sensor shape')
        rgb=np.frombuffer(newest[1],np.uint8).reshape(h,w,3);depth=np.frombuffer(newest[2],np.float32).reshape(h,w)
        if self.last_sensor and meta['frame_id']<=self.last_sensor['frame_id']:return
        self.last_sensor=meta;self.sensor_count+=1;self.latest_rgb=rgb.copy()
        if not self.released or self.initial_balance_only:return
        if self.skill_active and not self.skill_mapping_enabled:
            with (self.output/'sensor_excluded_during_skill.jsonl').open('a') as out:
                out.write(json.dumps(dict(frame_id=meta['frame_id'],sim_time=meta['sim_time'],
                    reason='Ground skill camera may see own body/floor; raw RGB-D retained, navigation map unchanged'))+'\n')
            return
        # The old raw camera_to_world/base fields are intentionally unused.
        T_WI,age=self.pose_history.at(round(meta['sim_time']*1e9),max_age_ns=200_000_000,require_bracket=True)
        T_WT=T_WI@np.linalg.inv(self.frames.T_TI)
        pitch=math.radians(10);forward=np.array([math.cos(pitch),0,-math.sin(pitch)])
        right=np.array([0,-1,0]);down=np.cross(forward,right)
        T_TC=np.eye(4);T_TC[:3,:3]=np.column_stack((right,down,forward));T_TC[:3,3]=HEAD_EYE_IN_TORSO
        T_WC=T_WT@T_TC
        origin,points,hits=old.depth_rays(depth,meta['intrinsics'],T_WC)
        base=self.frames.pelvis(T_WI,[self.truth_state['q'][i] for i in WAIST_INDICES])[:3,3]
        keep=np.linalg.norm(points[:,:2]-base[:2],axis=1)>.45
        update=self.grid.update_rays(origin,points[keep],hits[keep],now=meta['sim_time'])
        previously=self.detector.confirmed
        self.detector.observe(rgb,depth,meta['intrinsics'],T_WC,meta['frame_id'],meta['sim_time'])
        if self.detector.confirmed:
            self.detector.confirmed['localization']='FAST-LIO IMU estimate + calibrated torso camera extrinsic; one-time initial frame registration'
        if self.detector.confirmed and previously is None:
            self.manifest['target_observation']=self.detector.confirmed
            self.event('visual_target_confirmed',observation=self.detector.confirmed,localization='FAST-LIO')
            old.cv2.imwrite(str(self.output/'target_observation.png'),old.cv2.cvtColor(rgb,old.cv2.COLOR_RGB2BGR))
            old.atomic_json(self.output/'target_observation.json',self.detector.confirmed)
        with (self.output/'sensor_used.jsonl').open('a') as out:
            out.write(json.dumps(dict(frame_id=meta['frame_id'],sim_time=meta['sim_time'],lio_nearest_age_ns=age,
                localization='FAST-LIO; raw simulator camera pose ignored',**update))+'\n')
        if self.explorer and time.monotonic()-self.last_map_write>.35:self.draw_map()

    def settle(self,timeout=12):
        super().settle(timeout)
        if getattr(self,'_admitting_room_scan',False):
            self._admitting_room_scan=False
            record=self.explorer.admit_settled_room_scan(self.state['base'][:2],self.state['sim_time'])
            self.event('room_scan_admission',**record)
            if not record['admitted']:
                self.client.hold(math.radians(self.state['yaw_deg']))
                self.scan_yaw=None;self.scan_uses_initial_fixture=False;self.command_stabilizer.reset()
                raise ScanReplanRequested()
        if self.initial_balance_only and self.released:
            until=time.monotonic()+20
            while not self.pose_history or len(self.pose_history.records)<5:
                self.tick()
                self.client.hold(math.radians(self.state['yaw_deg']))
                if time.monotonic()>until:raise RuntimeError('FAST-LIO failed to initialize in measured free standing state')
            self.initial_balance_only=False
            self.event('lio_initialized',estimated_samples=len(self.pose_history.records),source='actual ROS2 FAST-LIO output')
            # READY requires a genuinely registered sensor frame, not merely
            # an estimator output followed by an artificial stale-sensor hold.
            until=time.monotonic()+5
            while not math.isfinite(self.grid.last_update):
                self.tick();self.client.hold(math.radians(self.state['yaw_deg']))
                if time.monotonic()>until:raise RuntimeError('No time-aligned RGB-D frame after FAST-LIO initialization')

    def turn_scan(self):
        self._admitting_room_scan=self.scan_yaw is None and self.explorer.room_scan
        try:return _retained_scan_with_overlap(self)
        except ScanReplanRequested:return
        finally:self._admitting_room_scan=False

    def launch(self):
        env=os.environ.copy();env.update(ROS_DOMAIN_ID='98',ROS_LOCALHOST_ONLY='1',PYTHONUNBUFFERED='1',
            ROS_LOG_DIR=str(self.output/'ros_logs'))
        (self.output/'ros_logs').mkdir()
        prefix=['/bin/bash','-c','source "$1"; shift; exec "$@"','bash',str(ROOT/'ros2_ws/activate.sh')]
        command=prefix+[str(ROOT/'ros2_ws/install/lib/fast_lio/fastlio_mapping'),'--ros-args','--params-file',
            str(ROOT/'ros2_ws/config/mid360_simdev.yaml'),'-p','use_sim_time:=true','-p','pcd_save.pcd_save_en:=true',
            '-p','map_file_path:='+str(self.output/'fastlio_map.pcd')]
        self.lio=old.ProcessLog(command,self.output/'fastlio.log',env,ROOT,terminal=False)
        self.lio_bridge=old.ProcessLog(prefix+['/usr/bin/python3',str(ROOT/'tools/lio_bridge.py'),'--output',str(self.output/'lio'),
            '--seconds',str(self.args.max_seconds+50)],self.output/'lio_bridge.log',env,ROOT,terminal=False)
        self.manifest['fastlio_command']=command
        super().launch()
        self.explorer.inject_review_after_room=OPTIONS.inject_review_after_room

    def prepare_skill_simulator(self,command,env):
        if self.crawl_enabled:
            command.extend(['--ground-reference',str(self.crawl_reference.source),'--controller-logs',str(self.output/'controller')])
            env['PUNKTNULL_MISSION_SCENE']='punktnull.terrain.inspection_passage'
            env['PUNKTNULL_ANKLE_ROLL_MARGIN_RAD']=str(OPTIONS.ankle_roll_target_margin_rad)
            env['PUNKTNULL_LIMIT_ANKLE_PITCH']='1' if OPTIONS.limit_ankle_pitch else '0'
            env['PUNKTNULL_ANKLE_BRAKING']='1' if OPTIONS.experimental_ankle_braking else '0'

    def validate_skill_setup(self):
        """Fail before process creation if geometry belongs to another clip/scene."""
        from punktnull.skills.reference import ReferenceClip
        from punktnull.skills.session import SkillSession,REGISTRY
        from punktnull.terrain import inspection_passage as scene
        if OPTIONS.crawl_reference is None or OPTIONS.skill_preflight is None:
            raise ValueError('--terrain-skills crawl requires --crawl-reference and --skill-preflight')
        self.crawl_reference=ReferenceClip.load(OPTIONS.crawl_reference)
        self.crawl_generated=str(self.crawl_reference.source)!=REGISTRY['crawl'].source
        SkillSession(self.crawl_reference,experimental=True,allow_generated=self.crawl_generated)
        proof=json.loads(OPTIONS.skill_preflight.read_text())
        required={'reference_source':str(self.crawl_reference.source),
            'reference_source_digest':self.crawl_reference.digest,
            'scene_source_sha256':hashlib.sha256(Path(scene.__file__).read_bytes()).hexdigest(),
            'fixture_id':scene.FIXTURE_ID,'contract':scene.MISSION_CONTRACT,'accepted':True}
        if any(proof.get(k)!=v for k,v in required.items()):
            raise ValueError('Skill geometry preflight is failed, stale or for a different scene/reference')
        margin=proof.get('minimum_clearance_m')
        if (not isinstance(margin,(int,float)) or not math.isfinite(margin) or
                margin<.08 or proof.get('required_margin_m')!=.08 or
                not .08<=scene.MISSION_CONTRACT['minimum_margin_m']<=margin):
            raise ValueError('Skill geometry preflight does not establish required .08m clearance')
        self.skill_preflight=proof

    def prepare_skill_controller(self,command):
        if self.crawl_enabled:command.extend(['--target-motion-logfile',str(self.output/'native_target_motion.csv')])

    def close(self):
        if self.explorer is not None:
            self.manifest['completed_room_scans']=self.explorer.completed_rooms
        try:super().close()
        finally:
            if self.lio and self.lio.process.poll() is None:
                env=os.environ.copy();env.update(ROS_DOMAIN_ID='98',ROS_LOCALHOST_ONLY='1',ROS_LOG_DIR=str(self.output/'ros_logs'))
                command=['/bin/bash','-c','source "$1"; shift; exec "$@"','bash',str(ROOT/'ros2_ws/activate.sh'),
                    'ros2','service','call','/map_save','std_srvs/srv/Trigger','{}']
                try:
                    result=subprocess.run(command,env=env,cwd=ROOT,capture_output=True,text=True,timeout=10)
                    (self.output/'map_save.log').write_text(result.stdout+result.stderr)
                    self.manifest['native_map_saved']=(self.output/'fastlio_map.pcd').is_file() and result.returncode==0
                except Exception as exc:self.manifest['map_save_error']=repr(exc)
            if self.odom_reader:self.odom_reader.close()
            for child in (self.lio_bridge,self.lio):
                if child:child.close()
            old.atomic_json(self.output/'manifest.json',self.manifest)

old.Mission=LioMission
if __name__=='__main__':
    extension=argparse.ArgumentParser(add_help=False)
    extension.add_argument('--start-yaw-offset',type=float,default=0.)
    extension.add_argument('--inject-review-after-room',choices=('west','east','north'))
    extension.add_argument('--lidar-points-per-second',type=int,choices=(10000,20000,200000),default=10000)
    extension.add_argument('--rgb-lidar',action='store_true',help='Publish display-only colored LiDAR for the wholebody XR session')
    extension.add_argument('--immersive',action='store_true',help='Publish dense RGB-D for the immersive OpenXR surface renderer')
    extension.add_argument('--terrain-skills',choices=('none','crawl'),default='none')
    extension.add_argument('--crawl-reference',type=Path,help='Verified retained/generated crawl reference; required with --terrain-skills crawl')
    extension.add_argument('--skill-preflight',type=Path,help='Passing geometry preflight bound to this scene and crawl reference')
    extension.add_argument('--ankle-roll-target-margin-rad',type=float,default=0.,help='Experimental native ankle-roll PD target limiting;0=off, maximum.08; physical limits unchanged')
    extension.add_argument('--limit-ankle-pitch',action='store_true',help='Extend the explicit native desired-position margin to ankle pitch; requires positive margin')
    extension.add_argument('--experimental-ankle-braking',action='store_true',help='Experimental bounded ankle end-stop assistance; actual limits unchanged, simulation only')
    OPTIONS,remaining=extension.parse_known_args()
    if '-h' in remaining or '--help' in remaining:
        print('FAST-LIO and experimental terrain-skill options:')
        extension.print_help()
        print('\nRetained mission options:')
    if not math.isfinite(OPTIONS.start_yaw_offset) or abs(OPTIONS.start_yaw_offset)>5:
        extension.error('Bounded start variation must be within +/-5 degrees')
    if not 0<=OPTIONS.ankle_roll_target_margin_rad<=.08:
        extension.error('Ankle target margin must be0..08rad')
    if OPTIONS.ankle_roll_target_margin_rad and OPTIONS.terrain_skills!='crawl':
        extension.error('Ankle target filtering requires explicit --terrain-skills crawl')
    if OPTIONS.limit_ankle_pitch and OPTIONS.ankle_roll_target_margin_rad==0:
        extension.error('--limit-ankle-pitch requires positive --ankle-roll-target-margin-rad')
    if OPTIONS.experimental_ankle_braking and not(OPTIONS.limit_ankle_pitch and OPTIONS.ankle_roll_target_margin_rad>0):
        extension.error('Experimental ankle braking requires positive margin and --limit-ankle-pitch')
    os.environ['PUNKTNULL_START_YAW_OFFSET']=str(OPTIONS.start_yaw_offset)
    os.environ['PUNKTNULL_LIDAR_RATE']=str(OPTIONS.lidar_points_per_second)
    os.environ['PUNKTNULL_RGB_LIDAR']='1' if OPTIONS.rgb_lidar else '0'
    os.environ['PUNKTNULL_IMMERSIVE']='1' if OPTIONS.immersive else '0'
    sys.argv=[sys.argv[0],*remaining]
    raise SystemExit(old.main())
