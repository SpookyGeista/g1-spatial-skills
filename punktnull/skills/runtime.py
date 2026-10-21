"""Measured telemetry/recording adapter for isolated skill validation."""
import math
import os
import time
import numpy as np
from .legacy import LEGACY, load
from . import guard

_retained=load('skill_runtime_view',LEGACY/'runtime_view.py')
atomic_json=_retained.atomic_json
post_command=_retained.post_command


def skill_camera_profile(scene, environ=None):
    """Rendering-only per-run configuration; no scene geometry is changed."""
    env = os.environ if environ is None else environ
    passage = scene.get('fixture_contract', {}).get('id') == 'service_passage_p01_v1'
    mode = env.get('PUNKTNULL_SKILL_CAMERA', 'side' if passage else 'chase').strip()
    if mode not in ('side', 'chase'):
        raise ValueError('PUNKTNULL_SKILL_CAMERA must be side or chase')
    profile = {'mode':mode, 'camera_slot':'chase', 'rendering_only':True,
               'first_person_unchanged':True, 'physics_unchanged':True,
               'configuration_source':'explicit environment or fixture-specific default',
               'rendered_validation':'pending; CPU geometry checks do not prove full mesh visibility'}
    if mode == 'chase':
        profile['projection']='retained perspective'
        return profile
    def vector(key, default):
        try:value=np.array([float(x) for x in env.get(key,default).split(',')])
        except (TypeError,ValueError) as exc:raise ValueError(key+' requires comma-separated finite XYZ metres') from exc
        if value.shape!=(3,) or not np.isfinite(value).all():
            raise ValueError(key+' requires comma-separated finite XYZ metres')
        return value
    eye=vector('PUNKTNULL_SKILL_CAMERA_EYE','1.6,-5.8,1.1')
    target=vector('PUNKTNULL_SKILL_CAMERA_TARGET','1.6,0,0.30')
    direction=target-eye
    if np.linalg.norm(direction)<.1 or np.linalg.norm(np.cross(direction,[0.,0.,1.]))<.01:
        raise ValueError('Fixed side view needs separated eye/target and a nonvertical sightline')
    width=float(env.get('PUNKTNULL_SKILL_CAMERA_WIDTH_M','5.8'))
    if not np.isfinite(width) or width<=0:
        raise ValueError('PUNKTNULL_SKILL_CAMERA_WIDTH_M must be finite and positive')
    profile.update(projection='orthographic',eye=eye.tolist(),target=target.tolist(),
                   up=[0.,0.,1.],orthographic_width_m=width,
                   automatic_occlusion_avoidance=False,
                   limitation='Opaque roof and real props remain visible; partial prop occlusion is possible.')
    return profile


class FixedSkillSpectator:
    """Preserve retained first-person calculation, replace only spectator pose.

    CameraRig.update still owns USD transforms and its per-frame camera log.
    This adapter performs no app/physics/render calls and issues no robot command.
    """
    def __init__(self, retained_geometry, profile):
        self.retained_geometry=retained_geometry
        self.pose={k:np.asarray(profile[k],dtype=float) for k in ('eye','target','up')}
        self.last=None

    def update(self,root,torso,dt):
        views=dict(self.retained_geometry.update(root,torso,dt))
        original_clip=views['clip']
        views['chase']={k:v.copy() for k,v in self.pose.items()}
        views['clip']={'mode':'fixed_skill_side','automatic_occlusion_avoidance':False,
                       'unoccluded':None,'full_mesh_visibility_verified':False,
                       'retained_follow_diagnostic_not_applied':original_clip}
        self.last=views
        return views


def install_skill_camera(rig, profile):
    """Use actual retained CameraRig set_pose/paths/geometry APIs, no new product."""
    if rig is None or profile['mode']!='side':return
    from pxr import UsdGeom
    camera=UsdGeom.Camera.Get(rig.stage,rig.paths['chase'])
    if not camera:raise RuntimeError('Retained spectator camera is unavailable')
    width=profile['orthographic_width_m']
    camera.CreateProjectionAttr('orthographic')
    camera.CreateHorizontalApertureAttr(width*10.)
    camera.CreateVerticalApertureAttr(width*10.*450./800.)
    rig.geometry=FixedSkillSpectator(rig.geometry,profile)
    rig.set_pose('chase',{k:profile[k] for k in ('eye','target','up')})


class RuntimeView(_retained.RuntimeView):
    def __init__(self,*args,**kwargs):
        positional=list(args);gui=bool(positional[4]) if len(positional)>4 else bool(kwargs.get('gui',False))
        if len(positional)>4:positional[4]=False
        else:kwargs['gui']=False
        super().__init__(*positional,**kwargs)
        self.skill_camera=skill_camera_profile(self.scene)
        install_skill_camera(self.rig,self.skill_camera)
        if gui:
            import omni.ui as ui
            self.window=ui.Window('.NULL | EXPERIMENTAL SKILL TEST',width=570,height=210)
            with self.window.frame:
                with ui.VStack(spacing=6):
                    ui.Label('Isolated diagnostic | spectator slot 3: '+self.skill_camera['mode'])
                    self.labels=[ui.Label('Initializing') for _ in range(4)]
                    ui.Button('ABORT / STOP SIMULATION',clicked_fn=lambda:post_command(self.run,'abort'))
                    ui.Label('No live takeover inside this experimental ground skill.')
        from isaacsim.core.utils.render_product import set_resolution
        for name,product,_ in self.products:
            if name=='chase':set_resolution(product.path,(800,450))
        atomic_json(self.output/'presentation_profile.json',dict(spectator_resolution=[800,450],
            first_resolution=[640,360],recording_fps=self.fps,physics_hz=200,controller_hz=50,
            spectator_camera=self.skill_camera,camera_per_frame_metadata='cameras.json: chase + chase_occlusion.mode'))
        self.stable_since_sim=None;self.stable_samples=0

    def update(self,base,torso,q,dq,bodies,names,support,status,sim_time,step,tilt):
        stable=(not support and status['mode']=='active' and .70<=float(base[2])<=.90
                and tilt<=12 and np.linalg.norm(base[7:10])<=.10 and np.sqrt(np.mean(dq*dq))<=.20)
        if stable:
            if self.stable_since_sim is None:self.stable_since_sim=sim_time;self.stable_samples=0
            self.stable_samples+=1
        else:self.stable_since_sim=None;self.stable_samples=0
        now=time.monotonic()
        if now-self.last_write<.02:return
        self.last_write=now
        g=guard.ACTIVE_GUARD
        w,x,y,z=base[3:7]
        state={'monotonic':now,'sim_time':float(sim_time),'physics_step_index':int(step),
               'support':bool(support),'command_mode':status['mode'],'base':base.tolist(),
               'q':q.tolist(),'dq':dq.tolist(),
               'tau':np.asarray(self.current_applied_efforts).tolist() if hasattr(self,'current_applied_efforts') else None,
               'effort_source':'current explicit clipped joint command; no contact-force measurement',
               'tilt_deg':float(tilt),
               'yaw_deg':math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))),
               'stable_full_rate_sim_seconds':0. if self.stable_since_sim is None else float(sim_time-self.stable_since_sim),
               'stable_physics_samples':self.stable_samples,
               'ground_guard_last':g.last if g else None,'native_feedback':g.native if g else None,
               'phase_evidence_basis':'measured tracking near submitted cursor; no native absolute frame telemetry'}
        atomic_json(self.run/'sim_state.json',state)
        self.current=state
        if self.labels:
            for label,text in zip(self.labels,self.status_lines(self.mission_status())):label.text=text

    @staticmethod
    def status_lines(status):
        return [status.get('state','STARTUP')+' | '+status.get('owner','none'),
                status.get('detail','')[:110],
                'EXPERIMENTAL reference skill | native Unitree G1 + SONIC',
                'Actual PhysX state | prerecorded skill, no live operator']
