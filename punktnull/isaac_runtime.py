"""Sensor attachment to the retained motor loop; no new locomotion controller."""
import json
import hashlib
import inspect
import os
import queue
import threading
import time
import textwrap
import numpy as np
from .retained import EXPLORATION,URDF,load,prepare
from .sensor_transport import encode,connect
from .sensors.mid360 import SensorConfig,SimMid360,PhysxRaycaster,read_mid360_mount,rotation_wxyz,save_scan
prepare()
baseline=load('_punktnull_baseline_runtime',EXPLORATION/'runtime_view.py')
atomic_json,post_command=baseline.atomic_json,baseline.post_command

# Extend the retained atomic telemetry publication, not its controller or
# stable-state calculation. A second write would expose incomplete snapshots.
_update_original=textwrap.dedent(inspect.getsource(baseline.RuntimeView.update))
_update_needle="    atomic_json(self.run/'sim_state.json',state);self.current=state"
_update_replacement="""    state['tau']=np.asarray(self.current_applied_efforts).tolist()
    if self.skill_guard_enabled:
        from punktnull.skills import guard
        g=guard.ACTIVE_GUARD
        state['ground_guard_last']=g.last if g else None
        state['native_feedback']=g.native if g else None
    atomic_json(self.run/'sim_state.json',state);self.current=state"""
if _update_original.count(_update_needle)!=1:raise RuntimeError('Retained telemetry publication changed')
_update_source=_update_original.replace(_update_needle,_update_replacement)
_update_namespace=dict(baseline.__dict__)
exec(compile(_update_source,str(EXPLORATION/'runtime_view.py')+':skill_telemetry','exec'),_update_namespace)
_extended_update=_update_namespace['update']

class RuntimeView(baseline.RuntimeView):
    def __init__(self,*args,**kwargs):
        import sys
        self.rgb_lidar_display=None
        if os.environ.get('PUNKTNULL_RGB_LIDAR')=='1':
            from .rgb_lidar import DisplayFusion
            output=args[0] if args else kwargs['output']
            self.rgb_lidar_display=DisplayFusion(output/'rgb_lidar')
        self.skill_guard_enabled='--ground-reference' in sys.argv
        self.current_applied_efforts=np.zeros(29)
        self.lio_queue=queue.Queue(maxsize=1200);self.lio_error=None;self.lio_counts={'imu':0,'cloud':0}
        self.lio_socket=None;self.lio_initial=False;self.lio_stable_samples=0
        positional=list(args);gui=bool(positional[4]) if len(positional)>4 else bool(kwargs.get('gui',False))
        if len(positional)>4:positional[4]=False
        else:kwargs['gui']=False
        super().__init__(*positional,**kwargs)
        # Audience-only camera profile. The calibrated 640x360 RGB-D product,
        # sensor cadence and all 200Hz physics remain unchanged.
        from isaacsim.core.utils.render_product import set_resolution
        for name,product,_ in self.products:
            if name=='chase':set_resolution(product.path,(800,450))
        self.presentation_profile=dict(spectator_resolution=[800,450],
            rgbd_resolution=[640,360],recording_fps=self.fps,physics_hz=200,
            purpose='Reduce audience-render cost; mission sensors and collision physics unchanged')
        if self.scene.get('fixture_id')=='inspection_north_passage_p02_v2':
            from punktnull.terrain.mission_camera import install_mission_skill_camera
            self.presentation_profile['skill_spectator']=install_mission_skill_camera(self.rig,self.scene,self.mission_status)
        if gui:
            # The interactive viewport is a third, audience-only render product.
            # Keep the separate RGB-D and recorded chase products untouched.
            if self.rig.viewport is None:raise RuntimeError('GUI viewport unavailable')
            self.presentation_profile['gui_viewport_resolution_before']=list(self.rig.viewport.get_texture_resolution())
            self.rig.viewport.set_texture_resolution((640,360))
            self.presentation_profile['gui_viewport_resolution_requested']=[640,360]
            import omni.ui as ui
            self.window=ui.Window('.NULL | FAST-LIO INSPECTION',width=570,height=255)
            with self.window.frame:
                with ui.VStack(spacing=6):
                    ui.Label('.NULL / LIDAR-INERTIAL INSPECTION',style={'font_size':20})
                    self.labels=[ui.Label('Initializing') for _ in range(4)]
                    with ui.HStack():
                        for label,key in [('START','start'),('TAKEOVER','takeover'),('RELEASE','release'),('ABORT','abort')]:
                            ui.Button(label,clicked_fn=lambda k=key:post_command(self.run,k))
                    ui.Label('FAST-LIO localization / approximate MID360 / simulation')
            from punktnull.gui import ObservedMapPanel
            self.map_panel=ObservedMapPanel(self.run,self.output)
            self.map_panel.configure_presentation_layout(self.window,self.rig.window)
        atomic_json(self.output/'presentation_profile.json',self.presentation_profile)
        atomic_json(self.output/'telemetry_extension.json',dict(
            retained_update_sha256=hashlib.sha256(_update_original.encode()).hexdigest(),
            extended_update_sha256=hashlib.sha256(_update_source.encode()).hexdigest(),
            added_channels=['tau','ground_guard_last','native_feedback'],
            torque_source='actual applied/clipped200Hz torque command in SDK joint order'))
        xyz,rpy=read_mid360_mount(URDF)
        config=SensorConfig(rays_per_second=int(os.environ.get('PUNKTNULL_LIDAR_RATE','20000')),
                            torso_to_lidar_xyz_m=xyz,torso_to_lidar_rpy_rad=rpy)
        self.lidar=SimMid360(PhysxRaycaster(g1_optical_window=True),config)
        self.T_TI=config.T_torso_lidar@config.T_lidar_imu
        self.lio_folder=self.output/'mid360';self.lio_folder.mkdir()
        self.lio_truth=(self.lio_folder/'evaluation_truth.jsonl').open('w',buffering=1)
        self.lio_thread=threading.Thread(target=self._lio_worker,daemon=True);self.lio_thread.start()

    update=_extended_update

    def _sensor_worker(self):
        """Reuse rendered pixels, strip the baseline's ideal world-pose fields."""
        try:
            import zmq
            context=zmq.Context();publisher=context.socket(zmq.PUB)
            publisher.setsockopt(zmq.SNDHWM,2);publisher.setsockopt(zmq.LINGER,0)
            publisher.bind('tcp://127.0.0.1:'+os.environ.get('NULL_SENSOR_PORT','16710'))
            folder=self.output/'sensors';folder.mkdir(exist_ok=True)
            with (folder/'index.jsonl').open('w') as index:
                while True:
                    item=self.sensor_queue.get()
                    if item is None:break
                    meta,rgb,depth=item
                    if os.environ.get('PUNKTNULL_IMMERSIVE')=='1':
                        from .rgbd_surface import write_rgbd
                        write_rgbd(self.output/'rgbd/latest.npz',rgb,depth,meta)
                    if self.rgb_lidar_display is not None:
                        self.rgb_lidar_display.offer_frame(meta,rgb,depth)
                    meta={k:v for k,v in meta.items() if k not in ('camera_to_world','base')}
                    meta['localization']='Requires external FAST-LIO estimate; no world pose in this packet'
                    publisher.send_multipart([json.dumps(meta).encode(),rgb.tobytes(),depth.tobytes()])
                    np.savez_compressed(folder/f"frame_{meta['frame_id']:06d}.npz",rgb=rgb,depth=depth,metadata=json.dumps(meta))
                    index.write(json.dumps(meta)+'\n');self.sensor_frames+=1
            publisher.close();context.term()
        except Exception as exc:self.sensor_error=repr(exc)

    def _lio_worker(self):
        try:
            first=self.lio_queue.get()
            if first is None:return
            deadline=time.monotonic()+15
            while self.lio_socket is None:
                try:self.lio_socket=connect(int(os.environ.get('NULL_LIO_PORT','16720')))
                except OSError:
                    if time.monotonic()>deadline:raise RuntimeError('FAST-LIO sensor bridge unavailable')
                    time.sleep(.1)
            with (self.lio_folder/'events.jsonl').open('w',buffering=1) as index:
                e=first
                while True:
                    if e is None:break
                    self.lio_socket.sendall(encode(e));kind=e['kind'];self.lio_counts[kind]+=1
                    if kind=='cloud':
                        name=f"scan_{self.lio_counts['cloud']:06d}.npz"
                        save_scan(self.lio_folder/name,e)
                        index.write(json.dumps(dict(kind=kind,stamp_ns=e['scanstart_ns'],scanstart_ns=e['scanstart_ns'],
                                                   scan_end_ns=e['scan_end_ns'],complete=e['complete'],file=name))+'\n')
                    else:index.write(json.dumps(e)+'\n')
                    e=self.lio_queue.get()
        except Exception as exc:self.lio_error=repr(exc)
        finally:
            if self.lio_socket:self.lio_socket.close()

    def before_physics(self,torso,sim_time,physics_step_index,status):
        if self.lio_error:raise RuntimeError('MID360 transport failed: '+self.lio_error)
        # Isaac accumulates a float32-valued dt (e.g.0.099999998), which would
        # mis-bin a 100ms boundary. The independently checked actual engine step
        # index gives the exact 5ms acquisition clock, not invented point times.
        sensor_time=int(physics_step_index)*.005
        if abs(sensor_time-float(sim_time))>1e-4:raise RuntimeError('Physics index and engine clock disagree')
        if not self.lio_initial:
            full_rate=getattr(self,'current',{}).get('stable_full_rate_sim_seconds',0.)
            stable=status['mode']=='active' and full_rate>=.70
            self.lio_stable_samples=self.lio_stable_samples+1 if stable else 0
            # Initialize on observed free standing, at a real scan boundary.
            # Startup transients must not contaminate FAST-LIO's gravity average.
            if not stable or physics_step_index%20:return
        T_WT=np.eye(4);T_WT[:3,:3]=rotation_wxyz(torso[3:7]);T_WT[:3,3]=torso[:3]
        T_WI=T_WT@self.T_TI
        if not self.lio_initial:
            # Declared one-time simulation start-frame registration, never a
            # correction to the estimator. Runtime truth remains separate.
            atomic_json(self.run/'localization_origin.json',dict(T_map_initial_imu=T_WI.tolist(),
                source='ONE-TIME simulator settled-start frame registration; no subsequent truth correction',sim_time=float(sim_time),
                observed_stable_seconds=full_rate,clock='actual physics step index x5ms'))
            self.lio_initial=True
        self.lio_truth.write(json.dumps(dict(stamp_ns=int(physics_step_index)*5_000_000,engine_time_s=float(sim_time),
            physics_step_index=int(physics_step_index),T_world_imu=T_WI.tolist()))+'\n')
        if self.rgb_lidar_display is not None:
            self.rgb_lidar_display.record_pose(int(physics_step_index)*5_000_000,torso,self.lidar.config.T_torso_lidar)
        data=self.lidar.update_physics(sensor_time,torso[:7],torso[7:10],torso[10:13])
        if self.rgb_lidar_display is not None:
            for scan in data['scans']:
                self.rgb_lidar_display.offer_scan(scan)
        if self.lidar.total_ray_queries>=20_000 and self.lidar.total_returns<100:
            raise RuntimeError('Actual G1 LiDAR has insufficient returns; inspect optical housing and query geometry')
        records=[]
        if data['imu'] is not None:
            records.append(dict(kind='imu',stamp_ns=data['imu']['timestamp_ns'],
                angular_velocity_rad_s=data['imu']['angular_velocity_rad_s'],
                specific_force_m_s2=data['imu']['specific_force_m_s2']))
        records.extend(dict(kind='cloud',**scan) for scan in data['scans'] if scan['complete'] and len(scan['xyz_sensor_m']))
        for record in records:
            try:self.lio_queue.put_nowait(record)
            except queue.Full:raise RuntimeError('MID360 measurement queue overrun; no silent dropping')

    @staticmethod
    def status_lines(status):
        task=('RECORDED POSE INPUT / automated test commands / no live human'
              if status.get('pose_input_kind')=='recorded_pose_replay'
              else 'TASK: inspect rooms / identify tagged case / return')
        budget=status.get('lidar_rays_per_second',20000)
        return [status.get('state','STARTUP')+' | '+status.get('owner','initializing'),
                status.get('detail','')[:104],
                task,
                f'FAST-LIO / MID360 proxy {budget}/s | G1 / NVIDIA SONIC | SIMULATION']

    def close(self):
        try:
            if self.rgb_lidar_display is not None:
                self.rgb_lidar_display.close()
                atomic_json(self.output/'rgb_lidar'/'status.json',dict(error=self.rgb_lidar_display.error,
                    scope='Simulation-only display; does not modify FAST-LIO or control'))
            if 'gui_viewport_resolution_requested' in self.presentation_profile:
                self.presentation_profile['gui_viewport_resolution_final']=list(self.rig.viewport.get_texture_resolution())
                atomic_json(self.output/'presentation_profile.json',self.presentation_profile)
            self.lio_queue.put(None,timeout=2);self.lio_thread.join(timeout=10)
            if self.lio_thread.is_alive():self.lio_error='Sensor transport did not drain'
            self.lio_truth.close()
            atomic_json(self.lio_folder/'metadata.json',dict(self.lidar.metadata(),transport_counts=self.lio_counts,error=self.lio_error))
        finally:super().close()
