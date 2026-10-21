#!/usr/bin/env python3
"""Read-only retained stair references -> foot-derived candidate geometry.

No reference playback, smoothing, root-height replacement, or skill admission.
Reports a missing stable landing explicitly rather than inventing one.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from punktnull.retained import prepare,URDF,REPO
from punktnull.terrain.kinematics import CollisionFK

FPS=120.
CONTACT_THRESHOLDS={'max_ankle_xy_speed_m_s':.15,'max_ankle_z_speed_m_s':.10,'minimum_duration_s':.10}
SETTLE={'window_s':.30,'max_root_speed_m_s':.10,'max_joint_rms_speed_rad_s':.20,'max_foot_level_difference_m':.025,'max_tilt_deg':12.}


def intervals(mask,minimum):
    edges=np.flatnonzero(np.diff(np.r_[False,np.asarray(mask,bool),False])).reshape(-1,2)
    return [(int(a),int(b)) for a,b in edges if b-a>=minimum]


def read_aligned(source,names):
    header=source.open().readline().strip().split(',')
    if header[7:]!=[n+'_dof' for n in names]:raise ValueError('Raw reference joint order mismatch')
    d=np.loadtxt(source,delimiter=',',skiprows=1)
    if d.shape[1]!=36 or not np.isfinite(d).all() or not np.array_equal(d[:,0],np.arange(len(d))):raise ValueError('Bad motion CSV')
    q=np.deg2rad(d[:,7:]);xyz=d[:,1:4]/100
    r=Rotation.from_euler('xyz',d[:,4:7],degrees=True);yaw=float(r.as_euler('xyz')[0,2]);align=Rotation.from_euler('z',-yaw)
    xyz=(xyz-np.r_[xyz[0,:2],0.])@align.as_matrix().T
    quat=(align*r).as_quat()[:,[3,0,1,2]]
    return xyz,quat,q,{'source_initial_xy_m':(d[0,1:3]/100).tolist(),'applied_yaw_deg':float(np.degrees(-yaw)), 'root_z_unchanged':True}


def analyze(source,fk):
    xyz,quat,q,alignment=read_aligned(source,fk.names);poses=fk.links(xyz,quat,q);bounds=fk.bounds(poses)
    feet={};contacts=[]
    for side in ('left','right'):
        link=side+'_ankle_roll_link';position=poses[link][:,:3,3];v=np.gradient(position,axis=0)*FPS
        ids=[i for i,c in enumerate(fk.collisions) if c['link']==link]
        low=bounds[:,ids,0,:].min(axis=1);high=bounds[:,ids,1,:].max(axis=1)
        feet[side]={'position':position,'velocity':v,'low':low,'high':high}
        mask=(np.linalg.norm(v[:,:2],axis=1)<=CONTACT_THRESHOLDS['max_ankle_xy_speed_m_s'])&(np.abs(v[:,2])<=CONTACT_THRESHOLDS['max_ankle_z_speed_m_s'])
        for a,b in intervals(mask,int(np.ceil(FPS*CONTACT_THRESHOLDS['minimum_duration_s']))):
            contacts.append({'side':side,'frames_half_open':[a,b],'time_s':[a/FPS,b/FPS],
                'ankle_median_xyz_m':np.median(position[a:b],axis=0).tolist(),
                'sole_median_z_m':float(np.median(low[a:b,2])),
                'sole_z_range_m':[float(low[a:b,2].min()),float(low[a:b,2].max())],
                'foot_x_envelope_m':[float(low[a:b,0].min()),float(high[a:b,0].max())],
                'foot_y_envelope_m':[float(low[a:b,1].min()),float(high[a:b,1].max())],
                'basis':'Low-speed FK interval; contact forces unavailable, candidate only'})
    contacts.sort(key=lambda c:sum(c['frames_half_open']))
    start_z=float(np.median(np.r_[feet['left']['low'][:12,2],feet['right']['low'][:12,2]]))
    elevated=[c for c in contacts if c['sole_median_z_m']>start_z+.05]
    previous_height=0.;previous_front=max(float(feet[s]['high'][:12,0].max()) for s in feet)
    steps=[]
    for i,c in enumerate(elevated):
        # Prefer a front face behind the new heel. It must also clear the previous
        # low support's toe envelope; heel overhang is reported, never hidden.
        front=max(previous_front+.010,c['foot_x_envelope_m'][0]-.020)
        height=c['sole_median_z_m']
        steps.append({'index':i+1,'front_x_m':front,'top_z_m':height,
            'rise_from_previous_m':height-previous_height,'width_m':1.55,
            'contact_index':contacts.index(c),'toe_support_length_from_front_m':c['foot_x_envelope_m'][1]-front,
            'heel_overhang_m':max(0.,front-c['foot_x_envelope_m'][0]),
            'top_estimation':'Median minimum capsule/mesh collision bound during low ankle-speed interval; not measured force contact'})
        previous_height=height;previous_front=c['foot_x_envelope_m'][1]
    for i,step in enumerate(steps):
        step['next_riser_x_m']=steps[i+1]['front_x_m'] if i+1<len(steps) else None
        step['tread_depth_m']=steps[i+1]['front_x_m']-step['front_x_m'] if i+1<len(steps) else None
    # Only bounded tread intervals become candidate boxes. The final top
    # landing remains absent because no next contact/stop reference defines it.
    step_boxes=[];nonfoot_overlaps=0;max_foot_depth=0.
    foot_indices=np.array([c['link'].endswith('_ankle_roll_link') for c in fk.collisions])
    for step in steps[:-1]:
        low=np.array([step['front_x_m'],-.775,0.]);high=np.array([step['next_riser_x_m'],.775,step['top_z_m']])
        overlap_depth=np.minimum(bounds[:,:,1],high)-np.maximum(bounds[:,:,0],low)
        overlap=np.all(overlap_depth>1e-9,axis=-1)
        nonfoot_overlaps+=int(overlap[:,~foot_indices].sum())
        foot_depth=np.where(overlap[:,foot_indices],overlap_depth[:,foot_indices,2],0.)
        max_foot_depth=max(max_foot_depth,float(foot_depth.max()))
        step_boxes.append({'name':'CandidateStep'+str(step['index']),'aabb_min':low.tolist(),'aabb_max':high.tolist(),
                           'center':((low+high)/2).tolist(),'size':(high-low).tolist(),'physics_executed':False})
    speed=np.linalg.norm(np.gradient(xyz,axis=0)*FPS,axis=1)
    dq=np.sqrt(np.mean((np.gradient(q,axis=0)*FPS)**2,axis=1))
    tilt=np.degrees(np.arccos(np.clip(Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()[:,2,2],-1,1)))
    dz=np.abs(feet['left']['low'][:,2]-feet['right']['low'][:,2]);n=int(FPS*SETTLE['window_s'])
    def settle(sl):
        metrics={'max_root_speed_m_s':float(speed[sl].max()),'max_joint_rms_speed_rad_s':float(dq[sl].max()),
                 'max_foot_level_difference_m':float(dz[sl].max()),'max_tilt_deg':float(tilt[sl].max())}
        return {**metrics,'meets_diagnostic_settle_window':all(metrics[k]<=SETTLE[k] for k in metrics)}
    report={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frames':len(xyz),'seconds':(len(xyz)-1)/FPS,
       'source_fps_basis':'Retained BONES converter assumes120Hz; source CSV has no FPS field',
       'alignment':alignment,'root_start_end_xyz_m':xyz[[0,-1]].tolist(),'contacts':contacts,'proposed_risers':steps,'bounded_candidate_collision_boxes':step_boxes,
       'candidate_box_aabb_diagnostic':{'nonfoot_sample_geometry_overlaps':nonfoot_overlaps,'maximum_foot_vertical_overlap_m':max_foot_depth,'interpretation':'Conservative collision bounds only; intentional foot/tread contact and retargeting error are not a contact-force test.'},
       'entry':settle(slice(0,n)),'exit':settle(slice(-n,None)),
       'final_foot_bottom_z_m':{s:float(feet[s]['low'][-1,2]) for s in feet},
       'stable_entry_to_exit_reference_established':False,
       'proposal_only':True,'last_landing_length_m':None}
    return report,{'xyz':xyz,'time':np.arange(len(xyz))/FPS,**{s+'_bottom_z':feet[s]['low'][:,2] for s in feet},**{s+'_ankle_x':feet[s]['position'][:,0] for s in feet}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('New output directory required; prior evidence is immutable')
    a.output.mkdir(parents=True)
    prepare();from bridge import BODY_JOINT_NAMES
    fk=CollisionFK(URDF,BODY_JOINT_NAMES)
    sources=sorted((REPO/'data/bones_seed_full/g1/csv/230323').glob('stairs_climbing_up*.csv'))
    sources=[f for f in sources if not f.name.endswith('_M.csv')]
    results=[];selected=None
    for src in sources:
        result,arrays=analyze(src,fk);results.append(result)
        if src.name=='stairs_climbing_up_start_R_001__A300.csv':selected=(result,arrays)
    report={'scope':'Unexecuted reference geometry study. No admitted stair skill or complete ascent demonstration.',
        'urdf':str(URDF),'urdf_sha256':hashlib.sha256(URDF.read_bytes()).hexdigest(),
        'collision_model':'Actual imported29G1 URDF, cylinder→capsule importer setting, conservative mesh AABBs',
        'contact_thresholds':CONTACT_THRESHOLDS,'diagnostic_settle_thresholds':SETTLE,
        'constraints':['Foot low speed is a contact candidate; force/pressure observations are absent.',
          'No source up_stop/up_loop clips were found in the audited climbing collection.',
          'All ordinary ascent sources finish mid-ascent; final landing dimensions and stable ending cannot be derived.',
          'No robot root movement, reference resampling/retargeting, or controller execution performed.',
          'Missing final contacts/terrain must not be filled with fabricated target geometry.'],
        'references':results,'complete_candidate_selected':None,'derivation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'diagnostic_geometry_anchor':selected[0]['source'] if selected else None}
    (a.output/'stair_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected:
        ref,data=selected;np.savez_compressed(a.output/'foot_profiles.npz',**data)
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for side,col in [('left','#0b7a86'),('right','#bf6917')]:
            axes[0].plot(data['time'],data[side+'_bottom_z'],color=col,label=side+' lowest collision bound')
            axes[1].plot(data[side+'_ankle_x'],data[side+'_bottom_z'],color=col,alpha=.8,label=side)
        for c in ref['contacts']:
            axes[0].plot(c['time_s'],[c['sole_median_z_m']]*2,lw=3,color='#323943')
        steps=ref['proposed_risers']
        for s in steps:
            if s['next_riser_x_m'] is not None:axes[1].plot([s['front_x_m'],s['next_riser_x_m']],[s['top_z_m']]*2,'k-',lw=2)
        axes[0].set(xlabel='Source time [s], assumed120Hz',ylabel='Foot collision bottom Z [m]',title='A300-001: low-speed support candidates')
        axes[1].set(xlabel='Aligned forward X [m]',ylabel='Foot collision bottom Z [m]',title='Candidate treads; final landing UNKNOWN')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('REFERENCE ONLY • actual G1 URDF FK • no PhysX stair execution',fontsize=11)
        fig.savefig(a.output/'foot_contact_geometry.png',dpi=150);plt.close(fig)
    print(json.dumps({'output':str(a.output),'references':len(results),'complete_candidate_selected':None,
        'exit_same_level_settle':{Path(r['source']).name:r['exit'] for r in results}},indent=2))

if __name__=='__main__':main()
