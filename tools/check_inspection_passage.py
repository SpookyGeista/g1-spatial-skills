#!/usr/bin/env python3
"""CPU construction preflight only; sampled FK and analytical entry uncertainty."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from punktnull.terrain.inspection_passage import MISSION_CONTRACT,collision_boxes,load_layout,scene_meta
from punktnull.terrain.kinematics import CollisionFK
from punktnull.retained import URDF,LEGACY,load
from punktnull.skills.reference import ReferenceClip
from punktnull.skills.phase import SkillReference


def transformed_envelope(bounds, entry, translation_m=0., heading_deg=0.):
    """Conservative AABB enclosing every entry yaw in ±heading and XY ±translation.

    Encloses rotations of each original per-link AABB, including analytic sine /
    cosine extrema within the heading interval; no Monte Carlo sampling claim.
    """
    b=np.asarray(bounds,float);lo=b[:,:,0];hi=b[:,:,1]
    points=np.stack([np.stack([x,y],axis=-1) for x,y in
         [(lo[:,:,0],lo[:,:,1]),(lo[:,:,0],hi[:,:,1]),(hi[:,:,0],lo[:,:,1]),(hi[:,:,0],hi[:,:,1])]],axis=2)
    phi=np.arctan2(points[:,:,:,1],points[:,:,:,0])+math.radians(entry['heading_deg'])
    r=np.linalg.norm(points,axis=-1);h=math.radians(heading_deg)
    wrap=lambda a:(a+np.pi)%(2*np.pi)-np.pi
    xs=np.stack((r*np.cos(phi-h),r*np.cos(phi+h)),axis=-1)
    ys=np.stack((r*np.sin(phi-h),r*np.sin(phi+h)),axis=-1)
    xmin=np.minimum(xs.min(-1),np.where(abs(wrap(phi-np.pi))<=h,-r,np.inf)).min(-1)
    xmax=np.maximum(xs.max(-1),np.where(abs(wrap(phi))<=h,r,-np.inf)).max(-1)
    ymin=np.minimum(ys.min(-1),np.where(abs(wrap(phi+np.pi/2))<=h,-r,np.inf)).min(-1)
    ymax=np.maximum(ys.max(-1),np.where(abs(wrap(phi-np.pi/2))<=h,r,-np.inf)).max(-1)
    out=b.copy();xy=np.array(entry['approach_xy_m'])
    out[:,:,0,:2]=np.stack([xmin,ymin],axis=-1)+xy-translation_m
    out[:,:,1,:2]=np.stack([xmax,ymax],axis=-1)+xy+translation_m
    return out


def compare(bounds,boxes):
    rows=[]
    for box in boxes:
        low=np.array(box['aabb_min']);high=np.array(box['aabb_max'])
        gap=np.maximum(np.maximum(bounds[:,:,0]-high,low-bounds[:,:,1]),0.)
        distance=np.linalg.norm(gap,axis=-1)
        overlap=np.all(np.minimum(bounds[:,:,1],high)-np.maximum(bounds[:,:,0],low)>1e-9,axis=-1)
        rows.append({'prim':box['prim'],'aabb_min':box['aabb_min'],'aabb_max':box['aabb_max'],
                     'sample_geometry_overlaps':int(overlap.sum()),'minimum_aabb_distance_m':float(distance.min())})
    return {'all_sampled_aabbs_clear':not any(r['sample_geometry_overlaps'] for r in rows),
            'minimum_nonfloor_distance_m':min(r['minimum_aabb_distance_m'] for r in rows),
            'overlaps':[r for r in rows if r['sample_geometry_overlaps']],
            'nearest_obstacles':sorted(rows,key=lambda r:r['minimum_aabb_distance_m'])[:6]}


def all_variant_boxes():
    unique={};variant_rows=[]
    for variant in (0,1,2):
        for room in ('west','east','north'):
            for present in (False,True):
                cfg=load_layout();cfg.update(variant=variant,target_room=room,target_present=present)
                boxes=collision_boxes(cfg)
                variant_rows.append(dict(variant=variant,target_room=room,target_present=present,boxes=len(boxes)))
                for box in boxes:unique[(box['prim'],tuple(box['aabb_min']),tuple(box['aabb_max']))]=box
    return list(unique.values()),variant_rows


def scan_clearance(boxes):
    """Truth-only oracle feasibility. Never writes/injects a navigation grid."""
    xy=np.array(MISSION_CONTRACT['north_scan_xy_m']);grid=.08
    relevant=[b for b in boxes if b['aabb_max'][2]>=.12 and b['aabb_min'][2]<=1.55]
    distances=[]
    for b in relevant:
        d=np.linalg.norm(np.maximum(np.maximum(np.array(b['aabb_min'])[:2]-xy,xy-np.array(b['aabb_max'])[:2]),0))
        distances.append((float(d),b['prim']))
    # Conservative occupied cell if any square intersects a height-band AABB.
    origin=np.array([-10.,-10.]);centers=origin+(np.indices((250,250))[::-1].transpose(1,2,0)+.5)*grid
    occupied=np.zeros((250,250),bool)
    for b in relevant:
        low=np.array(b['aabb_min'])[:2]-grid/2;high=np.array(b['aabb_max'])[:2]+grid/2
        occupied|=np.all((centers>=low)&(centers<=high),axis=-1)
    edt=ndimage.distance_transform_edt(~occupied)*grid
    cell=np.floor((xy-origin)/grid).astype(int)
    # Include all cell centers whose square could contain a pose within 0.10m.
    near=np.linalg.norm(centers-xy,axis=-1)<=.10+grid*math.sqrt(.5)
    return {'obstacle_height_band_m':[.12,1.55],'roof_included_in_occupied_projection':True,
      'nominal_aabb_point_clearance_m':min(d for d,_ in distances),'nearest':sorted(distances)[:6],
      'ideal_truth_grid_nominal_occupied_clearance_m':float(edt[cell[1],cell[0]]),
      'ideal_truth_grid_minimum_occupied_clearance_within_010m_plus_cell_halfdiag_m':float(edt[near].min()),
      'limitations':'Complete geometry oracle only. Actual observed/unknown cells, LIO error and scan admission must be checked during simulation. Do not copy this grid to autonomy.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--traces',type=Path,nargs='*',default=[ROOT/'runs/skills/008_passage_side/sim/trace.npz'])
    p.add_argument('--references',type=Path,nargs='*',default=[])
    a=p.parse_args()
    if a.output.exists():p.error('Existing evidence is never overwritten')
    a.output.mkdir(parents=True)
    scene_path=ROOT/'punktnull/terrain/inspection_passage.py'
    paths=[scene_path,Path(__file__),URDF,ROOT/'punktnull/terrain/kinematics.py']
    refpath=ROOT/'evidence/terrain/crawl_geometry_003/reference_bounds.npz';paths.append(refpath)
    archived=np.load(refpath)
    datasets={'retained_reference':(archived['bounds'],'Every reference frame at50Hz; source conservative URDF FK bounds retained')}
    endpoints={'retained_reference':(archived['bounds'][-1:],archived['root'][-1],'Last reference frame; not measured execution')}
    fk=CollisionFK(URDF,load('_inspection_fixture_bridge',LEGACY/'bridge.py').BODY_JOINT_NAMES)
    provenance={}
    for path in a.references:
        clip=ReferenceClip.load(path);ref=SkillReference(clip)
        reference_bounds=fk.bounds(fk.links(ref.xyz,ref.quat,ref.joints))
        datasets[path.name]=(reference_bounds,
            'Every exact candidate reference frame at50Hz; IsaacLab29 converted using retained SDK permutation')
        endpoints[path.name]=(reference_bounds[-1:],ref.xyz[-1],'Last exact candidate reference frame; not measured execution')
        provenance[path.name]=clip.provenance()
        paths.extend(path/name for name in clip.hashes)
    for path in a.traces:
        z=np.load(path);cols=list(z['columns']);data=z['data'][::4];get=lambda names:data[:,[cols.index(n) for n in names]]
        b=fk.bounds(fk.links(get(['root_'+str(i) for i in range(3)]),get(['root_'+str(i) for i in range(3,7)]),get(['q_'+str(i) for i in range(29)])))
        datasets[path.parent.parent.name]=(b,'Every fourth5ms actual physics record (50Hz); original fixture coordinate offset retained')
        paths.append(path)
        manifest_path=path.parent.parent/'manifest.json';paths.append(manifest_path)
        manifest=json.loads(manifest_path.read_text())
        event=next(e for e in manifest['events'] if e['event']=='measured_skill_return_ready')
        full=z['data'];index=int(np.argmin(abs(full[:,cols.index('sim_time')]-event['sim_time'])))
        row=full[index:index+1];g=lambda names:row[:,[cols.index(n) for n in names]]
        xyz=g(['root_'+str(i) for i in range(3)])
        endb=fk.bounds(fk.links(xyz,g(['root_'+str(i) for i in range(3,7)]),g(['q_'+str(i) for i in range(29)])))
        endpoints[path.parent.parent.name]=(endb,xyz[0],dict(event='measured_skill_return_ready',
            event_sim_time=event['sim_time'],trace_sim_time=float(row[0,cols.index('sim_time')]),trace_row=index))
    boxes,variants=all_variant_boxes();reports=[]
    for name,(bounds,source) in datasets.items():
        for direction in ('ingress','egress'):
            profiles={(0.,0.),(.04,1.5),(.10,3.),
              (MISSION_CONTRACT[direction]['entry_tolerance_m'],MISSION_CONTRACT[direction]['heading_tolerance_deg'])}
            for xy,yaw in sorted(profiles):
                transformed=transformed_envelope(bounds,MISSION_CONTRACT[direction],xy,yaw)
                result=compare(transformed,boxes)
                result['required_margin_m']=MISSION_CONTRACT['minimum_margin_m']
                result['meets_required_margin']=result['all_sampled_aabbs_clear'] and result['minimum_nonfloor_distance_m']>=MISSION_CONTRACT['minimum_margin_m']
                reports.append({'dataset':name,'source':source,'direction':direction,'samples':len(bounds),
                                'entry_translation_bound_each_axis_m':xy,'entry_heading_bound_deg':yaw,**result})
    contract_checks=[r for r in reports if r['entry_translation_bound_each_axis_m']==MISSION_CONTRACT[r['direction']]['entry_tolerance_m']
        and r['entry_heading_bound_deg']==MISSION_CONTRACT[r['direction']]['heading_tolerance_deg']]
    selected=provenance[a.references[0].name] if len(a.references)==1 else None
    minimum=min(r['minimum_nonfloor_distance_m'] for r in contract_checks)
    exit_checks=[]
    for name,(bounds,xyz,source) in endpoints.items():
        for direction in ('ingress','egress'):
            entry=MISSION_CONTRACT[direction];xy=entry['entry_tolerance_m'];yaw=entry['heading_tolerance_deg']
            result=compare(transformed_envelope(bounds,entry,xy,yaw),boxes)
            point=np.stack([xyz,xyz])[None,None]
            root_bounds=transformed_envelope(point,entry,xy,yaw)[0,0,:,:2]
            region=entry['exit_region'];low=np.array([region['xmin'],region['ymin']]);high=np.array([region['xmax'],region['ymax']])
            inside=bool(np.all(root_bounds[0]>=low)&np.all(root_bounds[1]<=high))
            exit_checks.append(dict(dataset=name,direction=direction,source=source,
                transformed_root_uncertainty_bounds_xy_m=root_bounds.tolist(),all_uncertain_root_positions_inside_exit_region=inside,
                measured_or_reference_body_margin_m=result['minimum_nonfloor_distance_m'],
                accepted=bool(inside and result['all_sampled_aabbs_clear'] and result['minimum_nonfloor_distance_m']>=MISSION_CONTRACT['minimum_margin_m'])))
    result={'scope':'New integrated scene geometry preflight only; no new simulation execution or skill admission',
      'reference_source':selected['source'] if selected else None,
      'reference_source_digest':selected['ordered_channel_digest'] if selected else None,
      'scene_source_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),
      'fixture_id':MISSION_CONTRACT['fixture_id'],'contract':MISSION_CONTRACT,
      'required_margin_m':MISSION_CONTRACT['minimum_margin_m'],'minimum_clearance_m':minimum,
      'accepted':bool(selected and a.traces and all(r['meets_required_margin'] for r in contract_checks) and all(r['accepted'] for r in exit_checks)),
      'acceptance_scope':'Sampled geometry preflight for the exact reference plus listed historical measured envelopes; not actual integrated skill acceptance.',
      'fixture':scene_meta(),'variant_conditions':variants,'unique_obstacle_boxes':len(boxes),'checks':reports,'exit_checks':exit_checks,'reference_provenance':provenance,
      'scan_geometry':scan_clearance(boxes),'source_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths},
      'limits':['FK AABBs are conservative and sampled in time; not continuous swept collision or contact force proof.',
                'Entry uncertainty analytically envelopes every heading angle in each interval and independent ±XY translations, not learned response variability.',
                'Ground contact excluded; original reference has below-floor retargeting bounds. Actual simulation resolves contacts.',
                'Earlier standalone passage runs did not pass all skill gates; their motions are planning evidence only.',
                'Actual integrated forward/back traversal, LIO continuity, cameras and planner rejoin remain untested.']}
    (a.output/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checks':[{k:r[k] for k in ['dataset','direction','entry_translation_bound_each_axis_m','entry_heading_bound_deg','all_sampled_aabbs_clear','minimum_nonfloor_distance_m','overlaps']} for r in reports],'scan_geometry':result['scan_geometry']},indent=2))
if __name__=='__main__':main()
