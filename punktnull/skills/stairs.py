"""Bounded two-riser reference construction and measured landing contract.

References are inputs to retained SONIC, never robot/root/terrain commands.
Foot transforms below are IK targets, not support constraints in the simulator.
"""
from pathlib import Path
import hashlib,json,uuid
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
from .legacy import OLD,WORKSPACE,load
from .limits import load_joint_limits
from .reference import FPS,WIDTHS,ReferenceClip
from punktnull.terrain.kinematics import CollisionFK

SOURCE=OLD/'data/bones_seed_full/g1/csv/230323/stairs_climbing_up_start_R_003__A300.csv'
URDF=OLD/'gear_sonic/data/assets/robot_description/urdf/g1/main.urdf'
MJCF=OLD/'gear_sonic_deploy/g1/g1_29dof.xml'
TOP=.264
FIRST_TOP=.120
FIRST_FRONT=.270
TOP_FRONT=.506
LIMIT_MARGIN=.025


def smooth(x):
    x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))


class LegIK:
    def __init__(self,fk,side):
        self.fk=fk;self.side=side;self.limits=load_joint_limits()
        self.indices=np.array([i for i,n in enumerate(fk.names) if n.startswith(side+'_') and i<12])
        self.chain=[j for j in fk.joints if j['name'] in [fk.names[i] for i in self.indices]]
        # URDF XML order is not a kinematic ordering guarantee.
        chain=[];parent='pelvis'
        while len(chain)<6:
            j=next(j for j in self.chain if j['parent']==parent);chain.append(j);parent=j['child']
        self.chain=chain
    def forward(self,root,q):
        t=root.copy()
        for j,a in zip(self.chain,q):
            r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(j['axis']*a).as_matrix()
            t=t@j['origin']@r
        return t
    def solve(self,root,target,seed):
        def error(q):
            t=self.forward(root,q)
            return np.r_[(t[:3,3]-target[:3,3])*5,
                         Rotation.from_matrix(target[:3,:3].T@t[:3,:3]).as_rotvec()*.6]
        lower=self.limits.lower[self.indices]+LIMIT_MARGIN;upper=self.limits.upper[self.indices]-LIMIT_MARGIN
        solved=least_squares(error,np.clip(seed,lower+1e-7,upper-1e-7),bounds=(lower,upper),max_nfev=80,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        actual=self.forward(root,solved.x)
        residual=[float(np.linalg.norm(actual[:3,3]-target[:3,3])),float(np.linalg.norm(Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec()))]
        if residual[0]>.004 or residual[1]>.035:raise ValueError('Infeasible landing IK '+str(residual))
        return solved.x,residual


def generate(output):
    """Generate a declared derivative in a NEW local directory; no execution."""
    output=Path(output).resolve()
    if output.exists() or not any(output.is_relative_to(WORKSPACE/p) for p in ('assets','evidence')):
        raise ValueError('Use new local assets/evidence reference directory')
    limits=load_joint_limits();fk=CollisionFK(URDF,limits.names)
    tools=load('stairs_geometry_derivation',WORKSPACE/'tools/derive_stair_fixture.py')
    converter=load('stairs_converter',OLD/'tools/convert_bones_to_sonic_reference.py')
    nativefk=converter.G1Kinematics(MJCF)
    if nativefk.joint_names!=list(limits.names):raise ValueError('Actual native joint order mismatch')
    xyz,quat,q,_=tools.read_aligned(SOURCE,limits.names)
    # Avoid demanding hard-stop joint angles, including source wrists. This
    # changes reference targets only; actual limits are checked at200Hz.
    q=np.clip(q,limits.lower+LIMIT_MARGIN,limits.upper-LIMIT_MARGIN)
    poses=fk.links(xyz,quat,q)
    # Trial001 measured the first toe strike22mm below the120mm riser.
    # Increase only the source swing target clearance with a smooth60mm hump;
    # contact targets, fixture, effort limits and real support remain unchanged.
    # The left second swing gets the same bounded correction.
    swing_residuals=[]
    for side,a,b,height in (('right',35,155,.060),('left',140,238,.120)):
        leg=LegIK(fk,side)
        for i in range(a,b+1):
            phase=(i-a)/(b-a);lift=height*(np.sin(np.pi*phase)**2)
            target=poses[side+'_ankle_roll_link'][i].copy();target[2,3]+=lift
            root=np.eye(4);root[:3,:3]=Rotation.from_quat(quat[i][[1,2,3,0]]).as_matrix();root[:3,3]=xyz[i]
            solved,error=leg.solve(root,target,q[i,leg.indices]);q[i,leg.indices]=solved;swing_residuals.append(error)
    poses=fk.links(xyz,quat,q)
    end=330;start=240;n=end+1
    xyz=xyz[:n].copy();quat=quat[:n].copy();q=q[:n].copy()
    anchor=poses['left_ankle_roll_link'][start].copy()
    right_end=poses['right_ankle_roll_link'][end].copy()
    right_end[:3,:3]=np.eye(3);right_end[2,3]=TOP+.039
    anchor[:3,:3]=np.eye(3);anchor[2,3]=TOP+.039
    final_xyz=np.array([(anchor[0,3]+right_end[0,3])/2+.065,(anchor[1,3]+right_end[1,3])/2,TOP+.775])
    initial_rot=Rotation.from_quat(quat[start][[1,2,3,0]])
    slerp=Slerp([0,1],Rotation.from_matrix([initial_rot.as_matrix(),np.eye(3)]))
    legs={s:LegIK(fk,s) for s in ('left','right')};residuals=[]
    for i in range(start,n):
        alpha=float(smooth((i-start)/(end-start)))
        xyz[i]=xyz[start]*(1-alpha)+final_xyz*alpha
        quat[i]=slerp(alpha).as_quat()[[3,0,1,2]]
        # Keep source arm/waist expression until the final settled posture.
        q[i,12:]=(1-alpha)*q[i,12:]+alpha*q[0,12:]
        root=np.eye(4);root[:3,:3]=Rotation.from_quat(quat[i][[1,2,3,0]]).as_matrix();root[:3,3]=xyz[i]
        for side in ('left','right'):
            target=poses[side+'_ankle_roll_link'][i].copy()
            if side=='left':
                # Stationary supporting left foot; smooth3-frame orientation adaptation.
                b=float(smooth((i-start)/20))
                target[:3,3]=(1-b)*target[:3,3]+b*anchor[:3,3]
                target[:3,:3]=Slerp([0,1],Rotation.from_matrix([target[:3,:3],anchor[:3,:3]]))(b).as_matrix()
            else:
                # Lower the source third footfall onto second landing. Keep
                # original swing lift, progressively remove its extra rise.
                b=float(smooth((i-start)/(end-start)))
                target[:3,3]=(1-b)*target[:3,3]+b*right_end[:3,3]
                target[:3,:3]=Slerp([0,1],Rotation.from_matrix([target[:3,:3],right_end[:3,:3]]))(b).as_matrix()
            seed=q[i,legs[side].indices] if i==start else q[i-1,legs[side].indices]
            solved,residual=legs[side].solve(root,target,seed)
            q[i,legs[side].indices]=solved;residuals.append(residual)
    # Existing120→50Hz converter; extra explicit1s stationary endpoint. No
    # root motion is sent by protocol1, all body motion must arise in PhysX.
    xyz,quat,q=converter.resample(xyz,quat,q,120.,FPS)
    xyz=np.r_[xyz,np.repeat(xyz[-1:],50,axis=0)]
    quat=np.r_[quat,np.repeat(quat[-1:],50,axis=0)]
    q=np.r_[q,np.repeat(q[-1:],50,axis=0)]
    qil=np.empty_like(q);qil[:,converter.MJ_TO_IL]=q
    bp,bq=nativefk.forward(xyz,quat,q)
    dq,bv,bw=converter.calculate_velocities(qil,bp,bq,FPS)
    channels=dict(joint_pos=qil,joint_vel=dq,body_pos=bp,body_quat=bq,body_lin_vel=bv,body_ang_vel=bw)
    actual_dq=dq[:,converter.MJ_TO_IL]
    if any(limits.check(qi,vi,require_effort=False) for qi,vi in zip(q,actual_dq)):
        raise ValueError('Generated reference violates actual position/speed limits')
    bounds=fk.bounds(fk.links(xyz,quat,q));from punktnull.terrain.stairs import collision_boxes
    diagnostic=[]
    for box in collision_boxes():
        depths=np.minimum(bounds[:,:,1],box['aabb_max'])-np.maximum(bounds[:,:,0],box['aabb_min'])
        hit=np.all(depths>1e-9,axis=-1)
        for j,c in enumerate(fk.collisions):
            if hit[:,j].any():diagnostic.append(dict(prim=box['prim'],link=c['link'],frames=int(hit[:,j].sum()),max_minimum_overlap_m=float(depths[hit[:,j],j].min(axis=1).max())))
    output.mkdir(parents=True)
    for key,width in WIDTHS.items():
        np.savetxt(output/(key+'.csv'),channels[key].reshape(len(q),width),delimiter=',',header=','.join(f'{key}_{i}' for i in range(width)),comments='',fmt='%.9f')
    (output/'metadata.txt').write_text(f'Total timesteps: {len(q)}\nFPS: 50\nJoint order: IsaacLab29\nGenerated bounded two-riser derivative, not an executed success.\n')
    source_files=[SOURCE,URDF,MJCF,OLD/'tools/convert_bones_to_sonic_reference.py']
    snapshots=output/'source_snapshot';snapshots.mkdir()
    generator_origins={}
    for p in (Path(__file__),WORKSPACE/'tools/derive_stair_fixture.py'):
        frozen=snapshots/p.name;frozen.write_bytes(p.read_bytes());source_files.append(frozen);generator_origins[str(frozen)]=str(p)
    manifest=dict(schema=1,kind='generated_g1_skill_reference',skill='stairs',fps=50,joint_order='IsaacLab29',
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},generator_original_paths=generator_origins,
        modifications=dict(retained_source_frames=[0,start],landing_transition_source_frames=[start,end],
            generated_landing_hold_seconds=1.,swing_clearance_correction=dict(right_height_m=.060,left_height_m=.120,right_source_interval=[35,155],left_source_interval=[140,238],basis='Actual001first_toe_strike;003first_riser_supported_but_second_toe_strike'),joint_reference_margin_rad=LIMIT_MARGIN,leg_ik='actual URDF6joint constrained least squares; reference targets only',
            final_root_target_m=xyz[-1].tolist(),terrain_height_m=TOP,physical_support_constraints=False),
        channels_sha256={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in [*(k+'.csv' for k in WIDTHS),'metadata.txt']},
        preflight=dict(max_ik_residual_m=float(np.array(residuals)[:,0].max()),max_ik_rotation_residual_rad=float(np.array(residuals)[:,1].max()),
            actual_reference_limits_checked=True,aabb_overlap_diagnostics=diagnostic,physical_execution=False),
        limitations=['Modified trajectory requires actual SONIC/PhysX execution.','AABB overlaps include foot tread contact and conservative envelopes.','Static fixture represents two prototype service steps, not general stairs.'])
    (output/'generated_reference.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return ReferenceClip.load(output)


def stable_landing(state,*,support_height=TOP):
    """Measured actual contact+posture, not world-Z-only endpoint proximity."""
    c=state.get('stair_contacts') or {}
    base=np.asarray(state.get('base',[]),float);dq=np.asarray(state.get('dq',[]),float)
    if base.shape!=(13,) or dq.shape!=(29,):return False
    return bool(not state.get('support',True) and state.get('command_mode')=='active'
        and c.get('both_on_landing',False) and c.get('fresh',False)
        and .70<=base[2]-support_height<=.90 and state.get('tilt_deg',999)<=12
        and np.linalg.norm(base[7:10])<=.10 and np.sqrt(np.mean(dq*dq))<=.20)


class StairSession:
    def __init__(self,clip):
        self.clip=clip;self.phase='DISARMED';self.events=[];self.token=None;self.entered=None
        self.heading=0.;self.ground_displacement=0.;self.initial_xy=None;self.first_support_confirmed=False
    def acquire(self,owner,state,clearance,now):
        from .session import measured_error
        error=measured_error(state,now,upright=True)
        if error or state.get('stable_full_rate_sim_seconds',0)<.7:raise RuntimeError(error or 'Unstable stair entry')
        self.token=uuid.uuid4().hex;self.phase='PRIMING';self.entered=now
        self.heading=np.deg2rad(state['yaw_deg']);self.initial_xy=np.array(state['base'][:2]);return self.token
    def authorization(self,now,stream):
        return dict(owner='skill' if self.token else 'none',owner_token=self.token,phase=self.phase,
            source=str(self.clip.source),source_digest=self.clip.digest,monotonic=now,clearance_ok=True,
            submitted_frame=stream.source_frame,heading_alignment_rad=self.heading)
    def update(self,token,state,clearance,now,*,native_stream_active,submitted_frame,inferred_frame,tracking_error,finished_sending):
        from .session import measured_error
        error=measured_error(state,now)
        if token!=self.token:error='Stair owner token mismatch'
        if error:return self.fail(error)
        if now-self.entered>25:return self.fail('Bounded staircase deadline')
        self.ground_displacement=float(np.linalg.norm(np.array(state['base'][:2])-self.initial_xy))
        if self.phase=='PRIMING':
            if native_stream_active:self.phase='ALIGNING';self.aligned_since=now
            return dict(action='HOLD_REFERENCE')
        if not native_stream_active:return self.fail('Native stair stream lost')
        if self.phase=='ALIGNING':
            # Prime first pose before beginning. No unsupported root reposition.
            if now-self.aligned_since>=.7 and state.get('stable_full_rate_sim_seconds',0)>=.5:self.phase='ENTERING'
            return dict(action='HOLD_REFERENCE')
        if self.phase=='ENTERING':
            if submitted_frame>=55 and not self.first_support_confirmed:
                if state.get('first_riser_contact_seconds',0)<.10:return dict(action='HOLD_REFERENCE')
                self.first_support_confirmed=True
                self.events.append(dict(event='measured_first_riser_support',sim_time=state['sim_time'],source_cursor=submitted_frame,
                    basis='Actual right ankle upward normal force>10N for.10s; native reference has1s future lookahead'))
            if finished_sending:self.phase='SETTLING'
            return dict(action='ADVANCE_REFERENCE')
        if self.phase=='SETTLING':
            if state.get('landing_stable_full_rate_seconds',0)>=.7 and self.ground_displacement>=getattr(self,'minimum_displacement',.50):
                self.phase='RETURN_READY';self.events.append(dict(event='measured_stable_landing',sim_time=state['sim_time'],base=state['base'][:7]))
            return dict(action='HOLD_REFERENCE')
        return dict(action='HOLD_REFERENCE')
    def fail(self,reason):
        self.phase='FAILED';return dict(action='PAUSE_SIM_AND_STOP_NATIVE',reason=reason)
    def release(self,token,*,fresh_map_safe,explicit_release):
        if token!=self.token or self.phase!='RETURN_READY' or not fresh_map_safe or not explicit_release:raise RuntimeError('Stair release conditions missing')
        self.token=None;self.phase='PLANNER_RETURN'
