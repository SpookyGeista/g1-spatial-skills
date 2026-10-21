"""Collision-active, measured-G1 service-passage fixture, simulation only.

Scene truth is for construction/evaluation. Do not import it into perception,
localization, navigation, or skill admission. No robot commands or root motion
are issued here. Uses the retained continuous floor, material palette and lights.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from punktnull.skills.legacy import LEGACY, load

BASE = '/World/ServicePassage'
FIXTURE_CONTRACT = {
    'id': 'service_passage_p01_v1',
    'axis': '+X',
    'roof_x_interval_m': [1.2, 2.0],
    'roof_underside_z_m': .65,
    'roof_top_z_m': 1.03,
    'clear_y_interval_m': [-.775, .775],
    'clear_width_m': 1.55,
    'start_xy_m': [0., 0.],
    'start_yaw_deg': 0.,
    'floor_z_m': 0.,
    'physical_passage_validated': False,
    'fixture_basis': 'Actual imported 29-joint G1 collision FK of reference and retained flat-floor PhysX run003; this is a proposed new collision trial.',
    'geometry_evidence': 'evidence/terrain/crawl_geometry_003/geometry.json',
    'admission_basis': 'No sensor-derived skill admission is implied by fixture geometry.',
}
DISCLOSURES = [
    'Static, deliberately shored ventilation-duct fixture; collapse, yielding, heat, smoke, and rubble dynamics are not modeled.',
    'Solid duct and all shoring members are collision enabled; the robot must physically pass underneath.',
    'Clearance is a configured prototype dimension. Navigation must not use this scene truth as observed terrain.',
    'The surrounding ceiling and south gallery wall are omitted for camera visibility. The passage roof remains solid and visible.',
    'Nominal motion and prior flat-floor body bounds are planning evidence only; a new collision-active run is required.',
    'No manipulation, stairs, physical hardware, or human teleoperation is tested by this fixture.',
]


def _legacy():
    return load('service_passage_base_scene', LEGACY/'scene.py')


def load_layout():
    cfg = deepcopy(_legacy().load_layout())
    cfg.update(title='.NULL Robotics / Service gallery P-01', robot_start_xy_m=[0.,0.], robot_start_yaw_deg=0.)
    cfg['cameras'] = {
        'Isometric': {'eye': [-2.6,-4.4,2.25], 'target': [1.65,0.,.50], 'up': [0.,0.,1.]},
        # Retained camera key/name; deliberately a side-oblique inspection view.
        'TopDown': {'eye': [1.6,-5.8,1.70], 'target': [1.6,0.,.40], 'up': [0.,0.,1.], 'orthographic_width': 7.7},
    }
    cfg['prototype_disclosures'] = DISCLOSURES.copy()
    cfg['fixture_contract'] = deepcopy(FIXTURE_CONTRACT)
    return cfg


def obstacle_specs():
    """The same exact boxes are rendered, collided, and exposed to evaluation."""
    boxes = []
    def add(name, center, size, material):
        boxes.append({'name': name, 'center': list(center), 'size': list(size), 'material': material})
    add('Duct/SolidHousing', [1.6,0.,.84], [.80,1.95,.38], 'steel')
    # Four corner props: the inner faces are exactly y=+/-0.775.
    for xi,x in enumerate((1.27,1.93)):
        for side,sign in (('South',-1),('North',1)):
            add(f'Shoring/{side}Post{xi}', [x,sign*.845,.325], [.14,.14,.65], 'amber')
            # Footings extend outward only; no invisible lip inside the opening.
            add(f'Shoring/{side}Foot{xi}', [x,sign*.91,.035], [.24,.27,.07], 'dark')
    # Recognisable utility gallery; these are real colliders outside the route.
    add('Gallery/RearWall', [2.1,2.45,1.35], [8.2,.18,2.7], 'wall')
    add('Gallery/EndWall', [6.1,.85,1.35], [.18,3.20,2.7], 'wall')
    add('Gallery/Cabinet', [4.65,2.02,.85], [1.05,.62,1.70], 'cabinet')
    add('Gallery/WestCrate', [-1.25,1.72,.33], [.65,.7,.66], 'cabinet')
    add('Gallery/PipeRun', [2.15,2.24,2.16], [6.2,.16,.16], 'pipe')
    return boxes


def collision_boxes():
    result=[]
    for item in obstacle_specs():
        low=[c-s/2 for c,s in zip(item['center'],item['size'])]
        high=[c+s/2 for c,s in zip(item['center'],item['size'])]
        result.append({'prim': BASE+'/'+item['name'], 'center':item['center'], 'size':item['size'],
                       'yaw_degrees':0., 'aabb_min':low, 'aabb_max':high})
    return result


def build_scene(hall=False, course=True):
    if hall: raise ValueError('This local service gallery requires --no-hall')
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    # Exact retained floor/light construction; no robot, material, or physics edits.
    metadata = _legacy().build_scene(hall=False, course=False)
    cfg=load_layout(); stage=omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(BASE): raise RuntimeError('Fresh stage required')
    UsdGeom.Xform.Define(stage,BASE)
    metadata.update(course_enabled=bool(course), course_prim=BASE, table_instances_spawned=0,
        title=cfg['title'], layout_path=str(Path(__file__).resolve()),
        layout_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        camera_eye=cfg['cameras']['Isometric']['eye'],camera_target=cfg['cameras']['Isometric']['target'],
        robot_start_xy_m=[0.,0.],robot_start_yaw_deg=0.,target_present=False,
        target_prim=BASE+'/NoInspectionTarget',target_id='NONE',target_source='No inspection objective in isolated skill fixture',
        prototype_disclosures=DISCLOSURES.copy(),fixture_contract=deepcopy(FIXTURE_CONTRACT),
        configured_camera_views={'top':'Fixed side-oblique passage view (retained UI key)', 'first':'Measured torso camera', 'chase':'Measured root following camera'},
        collision_boxes=collision_boxes() if course else [])

    def box(name,center,size,material,collision=False):
        cube=UsdGeom.Cube.Define(stage,BASE+'/'+name);cube.CreateSizeAttr(1.)
        cube.CreateExtentAttr([Gf.Vec3f(-.5),Gf.Vec3f(.5)])
        xf=UsdGeom.Xformable(cube.GetPrim());xf.AddTranslateOp().Set(Gf.Vec3d(*center));xf.AddScaleOp().Set(Gf.Vec3d(*size))
        mat=UsdShade.Material.Get(stage,'/World/Course/Materials/'+material)
        if not mat:raise RuntimeError('Missing retained material '+material)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        if collision:UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    if course:
        for item in obstacle_specs():box(**item,collision=True)
        # Duct panel seams and hazard bands lie on the outside of solid colliders.
        for i,x in enumerate((1.215,1.405,1.605,1.805,1.985)):
            box(f'Detail/DuctRib{i}',[x,0.,1.033],[.017,1.92,.006],'dark')
        for i,y in enumerate((-.91,-.71,-.51,-.31,-.11,.09,.29,.49,.69,.89)):
            box(f'Detail/EntryHazard{i}',[1.197,y,.85],[.006,.095,.34],'amber' if i%2==0 else 'dark')
        for side,y in (('South',-.983),('North',.983)):
            box(f'Detail/{side}Seam',[1.6,y,.85],[.76,.012,.014],'dark')
        for i,x in enumerate((-1.,0.,1.,2.,3.,4.,5.)):
            box(f'Detail/FloorSeam{i}',[x,.2,.001],[.007,4.4,.002],'dark')
        # Entry/exit markings identify a component test, without route waypoints.
        for name,x in (('Start',0.),('Exit',3.35)):
            for side,y in (('South',-.70),('North',.70)):
                box(f'Detail/{name}{side}',[x,y,.003],[.7,.025,.006],'cyan')
        box('Detail/RearProtection',[2.1,2.352,.20],[8.1,.018,.40],'wall_dark')
        for i,z in enumerate((.45,.52,.59)):
            box(f'Detail/CabinetVent{i}',[4.65,1.703,z],[.72,.016,.018],'dark')
        box('Detail/CabinetHandle',[4.98,1.69,.95],[.025,.035,.22],'dark')
        # Existing locally generated company plaque; no downloaded assets.
        asset=LEGACY/'assets/null.png'
        if not asset.is_file():raise FileNotFoundError(asset)
        path=BASE+'/Signs/Brand';plane=UsdGeom.Mesh.Define(stage,path)
        plane.CreatePointsAttr([Gf.Vec3f(x,2.345,z) for x,z in ((-.85,1.58),(.65,1.58),(.65,2.08),(-.85,2.08))])
        plane.CreateFaceVertexCountsAttr([4]);plane.CreateFaceVertexIndicesAttr([0,1,2,3]);plane.CreateSubdivisionSchemeAttr('none');plane.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(plane).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([(0,0),(1,0),(1,1),(0,1)])
        mat=UsdShade.Material.Define(stage,BASE+'/Materials/Brand');shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Surface');shader.CreateIdAttr('UsdPreviewSurface')
        texture=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Texture');texture.CreateIdAttr('UsdUVTexture');texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(asset));texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
        uv=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/UV');uv.CreateIdAttr('UsdPrimvarReader_float2');uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb');shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.8)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mat)
    metadata['layout_validation']={'roofless':True,'passage_roof_present':bool(course), 'obstacle_count':len(metadata['collision_boxes']), 'geometry_kind':'The exact visible obstacle boxes receive CollisionAPI; decorative bands/seams do not.'}
    return metadata
