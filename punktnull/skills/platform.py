"""BONES first ascent + explicit IK stop on one120mm raised platform."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .legacy import OLD,WORKSPACE,load
from .limits import load_joint_limits
from .reference import FPS,write_skill_reference
from .stairs import SOURCE,URDF,MJCF,LegIK,smooth,LIMIT_MARGIN,StairSession
from punktnull.terrain.kinematics import CollisionFK
TOP=.120

class PlatformSession(StairSession):
    minimum_displacement=.30


def generate(output):
    output=Path(output).resolve();snapshots=output.with_name(output.name+'_sources')
    if output.exists() or snapshots.exists():raise FileExistsError('New reference and provenance directories required')
    if not any(output.is_relative_to(WORKSPACE/p) for p in ('assets','evidence')):raise ValueError('Local assets/evidence required')
    L=load_joint_limits();fk=CollisionFK(URDF,L.names)
    reader=load('platform_raw_reader',WORKSPACE/'tools/derive_stair_fixture.py')
    converter=load('platform_converter',OLD/'tools/convert_bones_to_sonic_reference.py')
    xyz,quat,q,_=reader.read_aligned(SOURCE,L.names);q=np.clip(q,L.lower+LIMIT_MARGIN,L.upper-LIMIT_MARGIN)
    poses=fk.links(xyz,quat,q);right=LegIK(fk,'right')
    # Preserve the experimentally effective60mm first-swing lift correction.
    for i in range(35,156):
        target=poses['right_ankle_roll_link'][i].copy();target[2,3]+=.060*np.sin(np.pi*(i-35)/120)**2
        root=np.eye(4);root[:3,:3]=Rotation.from_quat(quat[i][[1,2,3,0]]).as_matrix();root[:3,3]=xyz[i]
        q[i,right.indices],_=right.solve(root,target,q[i,right.indices])
    poses=fk.links(xyz,quat,q);start=156;end=240;n=end+1
    xyz=xyz[:n].copy();quat=quat[:n].copy();q=q[:n].copy()
    anchor=poses['right_ankle_roll_link'][start].copy();anchor[:3,:3]=np.eye(3);anchor[2,3]=TOP+.039
    left_end=poses['left_ankle_roll_link'][220].copy();left_end[:3,:3]=np.eye(3);left_end[2,3]=TOP+.039
    final=np.array([(anchor[0,3]+left_end[0,3])/2+.065,(anchor[1,3]+left_end[1,3])/2,TOP+.775])
    initial_rotation=Rotation.from_quat(quat[start][[1,2,3,0]])
    root_rotation=Slerp([0,1],Rotation.from_matrix([initial_rotation.as_matrix(),np.eye(3)]))
    legs={side:LegIK(fk,side) for side in ('left','right')};residuals=[]
    for i in range(start,n):
        alpha=float(smooth((i-start)/(end-start)));xyz[i]=xyz[start]*(1-alpha)+final*alpha
        quat[i]=root_rotation(alpha).as_quat()[[3,0,1,2]]
        q[i,12:]=(1-alpha)*q[i,12:]+alpha*q[0,12:]
        root=np.eye(4);root[:3,:3]=root_rotation(alpha).as_matrix();root[:3,3]=xyz[i]
        for side in ('left','right'):
            target=poses[side+'_ankle_roll_link'][i].copy()
            if side=='right':
                b=float(smooth((i-start)/20));target[:3,3]=(1-b)*target[:3,3]+b*anchor[:3,3]
                target[:3,:3]=Slerp([0,1],Rotation.from_matrix([target[:3,:3],np.eye(3)]))(b).as_matrix()
            else:
                # Original second swing becomes a step onto the same first
                # platform height. The obstacle is never moved/removed mid-run.
                target[:3,3]=(1-alpha)*target[:3,3]+alpha*left_end[:3,3]
                target[:3,:3]=Slerp([0,1],Rotation.from_matrix([target[:3,:3],np.eye(3)]))(alpha).as_matrix()
            seed=q[i,legs[side].indices] if i==start else q[i-1,legs[side].indices]
            q[i,legs[side].indices],err=legs[side].solve(root,target,seed);residuals.append(err)
    # Platform001 established top support but left the trailing right heel
    # over the edge; native planner recentering then dropped it. Add one real
    # follow-through step so both feet finish parallel over the long platform.
    start_xyz=xyz[-1].copy();start_q=q[-1].copy()
    footposes=fk.links(xyz[-1:],quat[-1:],q[-1:])
    left_anchor=footposes['left_ankle_roll_link'][0].copy()
    right_start=footposes['right_ankle_roll_link'][0].copy()
    right_finish=right_start.copy();right_finish[0,3]=left_anchor[0,3];right_finish[:3,:3]=np.eye(3)
    final_follow=start_xyz.copy();final_follow[0]=left_anchor[0,3]+.065
    right_leg=legs['right'];left_leg=legs['left'];follow_xyz=[];follow_quat=[];follow_q=[]
    for k in range(1,145):
        t=k/144;blend=float(smooth(t));rootpos=(1-blend)*start_xyz+blend*final_follow
        # Bounded reference COM transfer toward the measured-support-side leg.
        rootpos[1]+=.055*np.sin(np.pi*t)**2
        root=np.eye(4);root[:3,3]=rootpos
        target=right_start.copy();travel=float(smooth((t-.15)/.70))
        target[:3,3]=(1-travel)*right_start[:3,3]+travel*right_finish[:3,3]
        target[2,3]+=.100*np.sin(np.pi*np.clip((t-.15)/.70,0,1))**2
        pose=start_q.copy() if not follow_q else follow_q[-1].copy()
        pose[right_leg.indices],err=right_leg.solve(root,target,pose[right_leg.indices]);residuals.append(err)
        pose[left_leg.indices],err=left_leg.solve(root,left_anchor,pose[left_leg.indices]);residuals.append(err)
        follow_xyz.append(rootpos);follow_quat.append([1.,0.,0.,0.]);follow_q.append(pose)
    xyz=np.r_[xyz,np.array(follow_xyz)];quat=np.r_[quat,np.array(follow_quat)];q=np.r_[q,np.array(follow_q)]
    xyz,quat,q=converter.resample(xyz,quat,q,120.,FPS)
    xyz=np.r_[xyz,np.repeat(xyz[-1:],75,axis=0)];quat=np.r_[quat,np.repeat(quat[-1:],75,axis=0)];q=np.r_[q,np.repeat(q[-1:],75,axis=0)]
    snapshots.mkdir(parents=True);source_files=[SOURCE,URDF,MJCF]
    for p in (Path(__file__),WORKSPACE/'punktnull/skills/stairs.py',WORKSPACE/'tools/derive_stair_fixture.py',WORKSPACE/'punktnull/terrain/platform.py'):
        dest=snapshots/p.name;dest.write_bytes(p.read_bytes());source_files.append(dest)
    modifications=dict(fixture='service_platform_p12_v1',height_m=TOP,source_first_ascent=[0,156],first_swing_extra_clearance_m=.060,
        first_swing_source_interval=[35,155],platform_stop_ik_source_interval=[start,end],constant_end_reference_seconds=1.5,follow_through=dict(side='right',seconds=1.2,swing_height_m=.10,com_lateral_reference_m=.055,reason='Actualplatform001trailingheeloverhang'),
        actual_urdf_leg_ik=True,joint_target_margin_rad=LIMIT_MARGIN,root_target_final_xyz_m=xyz[-1].tolist(),
        root_position_is_never_sent_to_robot=True,physical_support_constraints=False,
        max_ik_position_error_m=float(np.max(np.array(residuals)[:,0])),
        scope='One120mm platform step-up; not successful execution, general stairs or climbing')
    return write_skill_reference(output,xyz,quat,q,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},modifications,skill='stairs')
