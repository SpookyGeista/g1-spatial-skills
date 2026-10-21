"""One real120mm service-platform riser and long landing; no dynamic supports."""
from copy import deepcopy
import hashlib
from pathlib import Path
from punktnull.skills.legacy import LEGACY,load
BASE='/World/ServicePlatform'
FIXTURE_CONTRACT=dict(id='service_platform_p12_v1',start_xy_m=[0.,0.],start_yaw_deg=0.,
    rise_m=.120,front_x_m=.270,landing_end_x_m=2.5,clear_width_m=1.55,
    upper_height_m=.120,physical_execution_validated=False,
    scope='One12cm prototype service-platform step-up; not a staircase or general climbing')
def _legacy():return load('platform_base_scene',LEGACY/'scene.py')
def load_layout():
    cfg=deepcopy(_legacy().load_layout());cfg.update(title='.NULL |120mm service platform / experimental',robot_start_xy_m=[0.,0.],robot_start_yaw_deg=0.)
    cfg['fixture_contract']=deepcopy(FIXTURE_CONTRACT)
    cfg['cameras']={'Isometric':dict(eye=[1.1,-4.5,1.8],target=[.65,0.,.65],up=[0.,0.,1.]),'TopDown':dict(eye=[1.1,-5.,1.3],target=[.65,0.,.65],up=[0.,0.,1.],orthographic_width=4.2)}
    return cfg
def collision_boxes():
    lo=[.270,-.775,0.];hi=[2.5,.775,.120]
    return [dict(prim=BASE+'/Landing',aabb_min=lo,aabb_max=hi,center=[(a+b)/2 for a,b in zip(lo,hi)],size=[b-a for a,b in zip(lo,hi)],yaw_degrees=0.)]
def build_scene(hall=False,course=True):
    if hall or not course:raise ValueError('Platform trial requires real collision-active --no-hall course')
    import omni.usd
    from pxr import Gf,UsdGeom,UsdPhysics,UsdShade
    meta=_legacy().build_scene(hall=False,course=False);stage=omni.usd.get_context().get_stage();cfg=load_layout()
    if stage.GetPrimAtPath(BASE):raise RuntimeError('Fresh scene required')
    UsdGeom.Xform.Define(stage,BASE)
    for item in collision_boxes():
        cube=UsdGeom.Cube.Define(stage,item['prim']);cube.CreateSizeAttr(1.);cube.CreateExtentAttr([Gf.Vec3f(-.5),Gf.Vec3f(.5)])
        xf=UsdGeom.Xformable(cube.GetPrim());xf.AddTranslateOp().Set(Gf.Vec3d(*item['center']));xf.AddScaleOp().Set(Gf.Vec3d(*item['size']))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(UsdShade.Material.Get(stage,'/World/Course/Materials/wall_dark'))
    meta.update(course_enabled=True,course_prim=BASE,layout_obstacles_match_scene=True,
        title=cfg['title'],layout_path=str(Path(__file__)),layout_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_contract=deepcopy(FIXTURE_CONTRACT),collision_boxes=collision_boxes(),robot_start_xy_m=[0.,0.],robot_start_yaw_deg=0.,
        camera_eye=cfg['cameras']['Isometric']['eye'],camera_target=cfg['cameras']['Isometric']['target'],target_present=False,
        target_prim=BASE+'/NoInspectionTarget',target_id='NONE',target_source='No inspection objective',
        prototype_disclosures=['One rigid120mm riser with2230mm long landing and1550mm width.',
          'Long platform exists before first physics step; no geometry removal or motion during execution.',
          'Configured geometry selects this component experiment; not general terrain perception.',
          'Not a full staircase, generic climbing, physical G1 test or human takeover.'])
    return meta
