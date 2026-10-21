"""Additive, collision-active north-door crawl fixture for the known annex.

Scene boxes/targets are construction and evaluation truth. MISSION_CONTRACT is
an explicitly supplied fixed-fixture mission prior, not a perceived obstacle or
proof that the crawl skill is admitted. No robot command or root motion here.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from punktnull.retained import EXPLORATION, load

BASE='/World/InspectionPassage'
FIXTURE_ID='inspection_north_passage_p02_v2'
ROOF_Y_M=(1.90,2.55)
CLEAR_WIDTH_M=2.20
CLEAR_HEIGHT_M=.65
MISSION_CONTRACT={
    'fixture_id':FIXTURE_ID,
    'ingress':{'id':'north_ingress','approach_xy_m':[.08,.45],'heading_deg':90.,
               'exit_region':{'xmin':-.55,'xmax':.55,'ymin':3.20,'ymax':4.05},
               'entry_tolerance_m':.10,'heading_tolerance_deg':3.},
    'egress':{'id':'north_egress','approach_xy_m':[-.08,4.00],'heading_deg':270.,
              'exit_region':{'xmin':-.55,'xmax':.55,'ymin':.40,'ymax':1.20},
              'entry_tolerance_m':.10,'heading_tolerance_deg':3.},
    'minimum_margin_m':.08,
    'north_scan_xy_m':[0.,3.75],
    'geometry_source':'disclosed fixed simulation fixture',
    'skill_admitted':False,
    'two_passes_required':True,
    'coordinate_frame':'registered building map; one-time initial simulation registration disclosed',
    'admission_note':'Geometric pose bounds only. Actual state, reference posture, support-free stability, controller ownership and skill tracking must pass separately.',
}
DISCLOSURES=[
    'Known fixed north-door service passage; the mission checkpoint is an operator-supplied fixture prior, not detected terrain.',
    'Static shored duct blocks the existing north doorway at 0.65m height. Both entry and return require physical low passage traversal; no hidden bypass.',
    'All new roof, shoring and footings are visible collision boxes; no scripted movement, disappearing roof or hidden support.',
    'Existing north workbench and both legs move together to the east wall to provide a measured recovery area.',
    'North doorway and its adjoining approach throat are widened with actual visible wall colliders to 2.20m; original outer walls, floor, robot and target case placements are unchanged.',
    'Low-passage length 0.65m and 2.20m clear width are prototype fixtures, not general collapsed-building capability.',
    'Roof of the wider building remains omitted for presentation; the actual passage roof remains opaque and collidable.',
    'New integrated passage is not validated by earlier standalone crawl trials, which failed their complete admission criteria.',
]
_LIVE_LAYERS=[]


def _base():return load('_punktnull_inspection_passage_base_scene',EXPLORATION/'scene.py')


def _relocations():
    # Same three actual boxes, rigidly relocated/rotated as one workbench group.
    return {
      'ServiceBay/Workbench':{'center':[3.63,3.25,.75],'yaw':90.},
      'ServiceBay/BenchLeg0':{'center':[3.63,2.59,.34],'yaw':90.},
      'ServiceBay/BenchLeg1':{'center':[3.63,3.91,.34],'yaw':90.},
    }


def _structure_changes():
    return {
      'Structure/WestNorthPartition':{'center':[-1.19,1.455,1.30]},
      'Structure/EastNorthPartition':{'center':[1.19,1.455,1.30]},
      'Structure/NorthPartitionWest':{'center':[-2.50,2.20,1.30],'size':[2.80,.18,2.60]},
      'Structure/NorthPartitionEast':{'center':[2.50,2.20,1.30],'size':[2.80,.18,2.60]},
      'Structure/NorthDoorLintel':{'size':[2.20,.18,.40]},
    }


def passage_specs():
    boxes=[]
    def add(name,center,size,material):boxes.append(dict(name=name,center=list(center),size=list(size),material=material,yaw=0.))
    add('Duct/SolidHousing',[0.,sum(ROOF_Y_M)/2,.84],[2.60,ROOF_Y_M[1]-ROOF_Y_M[0],.38],'steel')
    # Inner faces coincide with the explicitly widened physical doorway.
    for side,sign in (('West',-1),('East',1)):
        for i,y in enumerate((ROOF_Y_M[0]+.07,ROOF_Y_M[1]-.07)):
            add(f'Shoring/{side}Post{i}',[sign*1.17,y,.325],[.14,.14,.65],'amber')
            # Footings extend away from the usable opening, never into it.
            add(f'Shoring/{side}Foot{i}',[sign*1.235,y,.035],[.27,.24,.07],'dark')
    return boxes


def obstacle_specs(layout=None):
    cfg=layout or load_layout();specs=deepcopy(_base().obstacle_specs(cfg))
    for item in specs:
        item['prim']='/World/Course/'+item['name']
        if item['name'] in _relocations():item.update(_relocations()[item['name']])
        if item['name'] in _structure_changes():item.update(_structure_changes()[item['name']])
    for item in passage_specs():specs.append({**item,'prim':BASE+'/'+item['name']})
    return specs


def collision_boxes(layout=None):
    boxes=[]
    for s in obstacle_specs(layout):
        low,high=_base().bounds(s['center'],s['size'],s['yaw'])
        boxes.append(dict(prim=s['prim'],center=s['center'],size=s['size'],yaw_degrees=s['yaw'],aabb_min=low,aabb_max=high))
    return boxes


def load_layout():
    cfg=deepcopy(_base().load_layout());cfg['title']='.NULL / Inspection annex with shored north access'
    cfg['prototype_disclosures']+=DISCLOSURES
    cfg['inspection_passage']={'fixture_id':FIXTURE_ID,'roof_y_interval_m':list(ROOF_Y_M),
       'clear_width_m':CLEAR_WIDTH_M,'clear_height_m':CLEAR_HEIGHT_M,'roof_top_z_m':1.03,
       'mission_contract':deepcopy(MISSION_CONTRACT),'workbench_relocations':deepcopy(_relocations()),
       'structural_modifications':deepcopy(_structure_changes()),
       'actual_integrated_scene_validated':False,
       'passage_side_camera':{'eye':[-2.80,3.90,1.10],'target':[0.,2.225,.32],'up':[0.,0.,1.],
          'orthographic_width':5.8,'scope':'Proposed spectator view; first-person sensor is unchanged; actual rendered inspection required.'}}
    for door in cfg['doorways']:
        if door['id']=='north':door.update(structural_clear_height_m=door['clear_height_m'],clear_height_m=CLEAR_HEIGHT_M,clear_width_m=CLEAR_WIDTH_M,
            passage_depth_m=ROOF_Y_M[1]-ROOF_Y_M[0],upright_blocked=True)
    return cfg


def scene_meta():
    """CPU-accessible evaluation construction contract, never a perception feed."""
    cfg=load_layout()
    return {'scene_kind':'three_room_annex_with_north_service_passage','evaluation_only':True,
        'fixture_id':FIXTURE_ID,'collision_boxes':collision_boxes(cfg),'floor':cfg['floor'],
        'rooms':cfg['rooms'],'doorways':cfg['doorways'],'bounds_xy':cfg['bounds_xy'],
        'robot_start_xy_m':cfg['robot_start_xy_m'],'robot_start_yaw_deg':cfg['robot_start_yaw_deg'],
        'exit_zone':cfg['exit_zone'],'inspection_passage':cfg['inspection_passage'],
        'prototype_disclosures':cfg['prototype_disclosures']}


def build_scene(hall=True,course=True):
    if not course:raise ValueError('Integrated inspection requires its real collision scene')
    import omni.usd
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
    metadata=_base().build_scene(hall=hall,course=True)
    stage=omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(BASE):raise RuntimeError('Fresh stage required: inspection passage exists')
    layer=Sdf.Layer.CreateAnonymous('punktnull_inspection_passage.usda')
    stage.GetSessionLayer().subLayerPaths.insert(0,layer.identifier);_LIVE_LAYERS.append(layer)
    def box(name,center,size,material,collision=False):
        cube=UsdGeom.Cube.Define(stage,BASE+'/'+name);cube.CreateSizeAttr(1.)
        cube.CreateExtentAttr([Gf.Vec3f(-.5),Gf.Vec3f(.5)])
        xf=UsdGeom.Xformable(cube.GetPrim());xf.AddTranslateOp().Set(Gf.Vec3d(*center));xf.AddScaleOp().Set(Gf.Vec3d(*size))
        mat=UsdShade.Material.Get(stage,'/World/Course/Materials/'+material)
        if not mat:raise RuntimeError('Missing retained material '+material)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        if collision:UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    with Usd.EditContext(stage,layer):
        UsdGeom.Xform.Define(stage,BASE)
        for name,change in {**_relocations(),**_structure_changes()}.items():
            prim=stage.GetPrimAtPath('/World/Course/'+name)
            if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):raise RuntimeError('Expected collidable fixture part missing: '+name)
            xf=UsdGeom.Xformable(prim);ops=xf.GetOrderedXformOps()
            translate=next(op for op in ops if op.GetOpType()==UsdGeom.XformOp.TypeTranslate)
            scale=next(op for op in ops if op.GetOpType()==UsdGeom.XformOp.TypeScale)
            if 'center' in change:translate.Set(Gf.Vec3d(*change['center']))
            if 'size' in change:scale.Set(Gf.Vec3d(*change['size']))
            if 'yaw' in change:
                rotate=xf.AddRotateZOp();rotate.Set(change['yaw']);xf.SetXformOpOrder([translate,rotate,scale])
        # Move existing surface paint with the widened throat, not floating in air.
        for side,sign in (('West',-1),('East',1)):
            for j in range(10):
                prim=stage.GetPrimAtPath(f'/World/Course/Detail/{side}DoorStripe1_{j}')
                op=next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate)
                op.Set(Gf.Vec3d(sign*1.089,.825,.10+j*.215))
        for s in passage_specs():box(s['name'],s['center'],s['size'],s['material'],True)
        # Visible shoring/duct details are backed by the solid collision housing.
        for i,x in enumerate((-.96,-.64,-.32,0.,.32,.64,.96)):
            box(f'Detail/TopRib{i}',[x,2.225,1.034],[.014,.62,.008],'dark')
        for mouth,y in (('South',ROOF_Y_M[0]-.004),('North',ROOF_Y_M[1]+.004)):
            for i,x in enumerate((-.94,-.72,-.50,-.28,-.06,.16,.38,.60,.82,1.04)):
                box(f'Detail/{mouth}Hazard{i}',[x,y,.84],[.105,.008,.34],'amber' if i%2==0 else 'dark')
        for name,entry in (('Ingress',MISSION_CONTRACT['ingress']),('Egress',MISSION_CONTRACT['egress'])):
            x,y=entry['approach_xy_m']
            for side,sign in (('West',-1),('East',1)):
                box(f'Detail/{name}{side}Mark',[x+sign*.64,y,.003],[.025,.50,.006],'cyan')
    cfg=load_layout();metadata.update(scene_meta())
    metadata['scene_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['base_scene_source_sha256']=hashlib.sha256((EXPLORATION/'scene.py').read_bytes()).hexdigest()
    metadata['passage_overlay_layer']=layer.identifier
    metadata['layout_validation'].update(obstacle_count=len(metadata['collision_boxes']),passage_roof_present=True,
        geometry_kind='Original boxes with three workbench relocations and five declared real structural changes, plus visible CollisionAPI duct/shoring boxes')
    metadata['title']=cfg['title']
    return metadata
