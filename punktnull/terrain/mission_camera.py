"""CPU-safe spectator-only portal view for the disclosed inspection passage.

No command, physics update, render product, sensor pose or scene edit occurs.
Visibility is a scene-box/body-proxy diagnostic, never a full-mesh guarantee.
"""
import numpy as np

PROFILES={
 'ingress':{'eye':[.65,-.80,.55],'target':[0.,2.225,.30],'up':[0.,0.,1.],'focal_length_mm':24.},
 'egress':{'eye':[-.20,4.82,.55],'target':[0.,2.225,.30],'up':[0.,0.,1.],'focal_length_mm':18.},
}


def assess_points(eye,target,up,points,boxes,focal_length_mm=24.):
    """Perspective36x20.25mm ray-to-point scene-box diagnostics."""
    eye=np.asarray(eye,float);points=np.asarray(points,float).reshape(-1,3)
    delta=points-eye;blocked=np.zeros(len(points),bool);blocking=[]
    for box in boxes:
        lo=np.asarray(box['aabb_min'],float);hi=np.asarray(box['aabb_max'],float)
        parallel=abs(delta)<1e-10;safe=np.where(parallel,1.,delta)
        a=(lo-eye)/safe;b=(hi-eye)/safe
        near=np.where(parallel,-np.inf,np.minimum(a,b)).max(axis=1)
        far=np.where(parallel,np.inf,np.maximum(a,b)).min(axis=1)
        outside=(parallel&((eye<lo)|(eye>hi))).any(axis=1)
        hit=(~outside)&(np.maximum(near,0.)<np.minimum(far,1.-1e-6))
        blocked|=hit
        if hit.any():blocking.append(box['prim'])
    forward=np.asarray(target,float)-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,up);right/=np.linalg.norm(right);vertical=np.cross(right,forward)
    depth=delta@forward
    ndc=np.column_stack(((delta@right)/np.maximum(depth,1e-9)/(18./focal_length_mm),
                         (delta@vertical)/np.maximum(depth,1e-9)/(10.125/focal_length_mm)))
    framed=(depth>.015)&(abs(ndc).max(axis=1)<=1.)
    return dict(point_count=len(points),visible_point_count=int((~blocked).sum()),
       framed_point_count=int(framed.sum()),occluded=bool(blocked.any()),blocked_by=blocking,
       full_proxy_in_frame=bool(framed.all()),maximum_proxy_ndc_extent=float(abs(ndc).max()))


def low_body_probes(root,heading_deg):
    """Deliberately labelled coarse prone-body proxy; measured root only."""
    r=np.asarray(root,float);a=np.deg2rad(heading_deg)
    forward=np.array([np.cos(a),np.sin(a)]);side=np.array([-np.sin(a),np.cos(a)])
    points=[]
    for x in (-.65,0.,.85):
        for y in (-.35,.35):
            for z in (.03,max(.15,float(r[2])+.20)):
                points.append([*(r[:2]+forward*x+side*y),z])
    return np.asarray(points)


class MissionSkillSpectator:
    def __init__(self,retained,boxes,status_callback,set_focal_length=None):
        self.retained=retained;self.boxes=boxes;self.status_callback=status_callback
        self.active_direction=None;self.set_focal_length=set_focal_length;self.focal_length=24.

    def lens(self,value):
        if value!=self.focal_length:
            if self.set_focal_length:self.set_focal_length(value)
            self.focal_length=value

    def update(self,root,dt):
        original_pose,original_metadata=self.retained.update(root,dt)
        status=self.status_callback() or {};gate=status.get('autonomous_skill') or {}
        direction=gate.get('direction');phase=str(status.get('state',''))
        applicable=direction in PROFILES and phase.startswith('SKILL_')
        if not applicable or float(root[2])>=.60:self.active_direction=None
        elif float(root[2])<.50:self.active_direction=direction
        if self.active_direction is None:
            self.lens(24.)
            return original_pose,{**original_metadata,'mission_skill_camera':'retained',
                'mission_phase':phase,'first_person_unchanged':True}
        profile=PROFILES[self.active_direction]
        pose={k:np.asarray(profile[k],float).copy() for k in ('eye','target','up')}
        self.lens(profile['focal_length_mm'])
        heading=90. if self.active_direction=='ingress' else 270.
        check=assess_points(**profile,points=low_body_probes(root,heading),boxes=self.boxes)
        return pose,{**check,'mode':'mission_low_portal','direction':self.active_direction,
            'mission_phase':phase,'actual_root_height_m':float(root[2]),
            'projection':'perspective','focal_length_mm':self.focal_length,'automatic_occlusion_avoidance':False,
            'rendering_only':True,'first_person_unchanged':True,'physics_unchanged':True,
            'full_mesh_visibility_verified':False,
            'visibility_basis':'Scene AABB rays to coarse prone-body proxy from actual root; self-occlusion is not modelled',
            'retained_view_diagnostic_not_applied':original_metadata}


def install_mission_skill_camera(rig,scene,status_callback):
    """Wrap only CameraRig.spectator; CameraRig owns poses and camera logs."""
    contract=scene.get('inspection_passage',{}).get('mission_contract',{})
    if contract.get('fixture_id')!='inspection_north_passage_p02_v2':
        return {'installed':False,'reason':'No matching disclosed passage fixture'}
    if isinstance(rig.spectator,MissionSkillSpectator):raise RuntimeError('Mission spectator already installed')
    # First camera, projection type, depth calibration and products unchanged.
    # Only the spectator lens widens for the short north-room egress view.
    from pxr import UsdGeom
    camera=UsdGeom.Camera.Get(rig.stage,rig.paths['chase'])
    if not camera or camera.GetProjectionAttr().Get()!='perspective':
        raise RuntimeError('Mission portal camera requires retained perspective chase')
    values=[camera.GetFocalLengthAttr().Get(),camera.GetHorizontalApertureAttr().Get(),camera.GetVerticalApertureAttr().Get()]
    if not np.allclose(values,[24.,36.,20.25],atol=1e-5):raise RuntimeError('Retained chase lens changed')
    rig.spectator=MissionSkillSpectator(rig.spectator,
        [*scene['collision_boxes'],*scene.get('spectator_visual_occluders',[])],status_callback,
        lambda value:camera.GetFocalLengthAttr().Set(float(value)))
    return dict(installed=True,profiles=PROFILES,projection='perspective; spectator lens24mm ingress /18mm egress; restores24mm',
        activates='Actual SKILL phase with matching gate and measured root height below0.50m',
        restores='Retained view above0.60m or outside actual skill phase',
        first_person_unchanged=True,physics_unchanged=True,extra_render_products=0,
        rendered_validation='pending actual integrated run; CPU rays do not prove mesh visibility')
