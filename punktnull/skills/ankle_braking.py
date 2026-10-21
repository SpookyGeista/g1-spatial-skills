"""Explicit experimental ankle end-stop assistance; OFF unless selected.

40ms kinematic lookahead,0.08rad activation,10Nm restoring term plus
4Nm/(rad/s) outward damping, each one-sided contribution capped at20Nm.
Existing50Nm physical effort caps and strict actual-state guard remain required.
This is neither a stopping-distance guarantee nor production robot safety.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
from .limits import load_joint_limits

JOINTS=('left_ankle_pitch_joint','left_ankle_roll_joint','right_ankle_pitch_joint','right_ankle_roll_joint')
LOOKAHEAD_S=.04
ACTIVATION_RAD=.08
RESTORING_NM=10.
DAMPING_NM_PER_RAD_S=4.
ADDITIONAL_CAP_NM=20.


def assistance(q,dq,lower,upper):
    """Signed assistance and all independently auditable scalar terms."""
    values=np.asarray([q,dq,lower,upper],float)
    if not np.isfinite(values).all() or lower>=upper:raise ValueError('Invalid ankle state/limits')
    gap_lower=q-lower;gap_upper=upper-q
    out_lower=max(0.,-dq);out_upper=max(0.,dq)
    predicted_lower=gap_lower-LOOKAHEAD_S*out_lower;predicted_upper=gap_upper-LOOKAHEAD_S*out_upper
    weight_lower=float(np.clip((ACTIVATION_RAD-predicted_lower)/ACTIVATION_RAD,0.,1.))
    weight_upper=float(np.clip((ACTIVATION_RAD-predicted_upper)/ACTIVATION_RAD,0.,1.))
    raw_lower=weight_lower*(RESTORING_NM+DAMPING_NM_PER_RAD_S*out_lower)
    raw_upper=weight_upper*(RESTORING_NM+DAMPING_NM_PER_RAD_S*out_upper)
    lower_assist=min(ADDITIONAL_CAP_NM,raw_lower);upper_assist=min(ADDITIONAL_CAP_NM,raw_upper)
    return dict(gap_lower=gap_lower,gap_upper=gap_upper,outward_speed_lower=out_lower,outward_speed_upper=out_upper,
        predicted_gap_lower=predicted_lower,predicted_gap_upper=predicted_upper,
        weight_lower=weight_lower,weight_upper=weight_upper,raw_assist_lower=raw_lower,raw_assist_upper=raw_upper,
        capped_assist_lower=lower_assist,capped_assist_upper=upper_assist,signed_assist=lower_assist-upper_assist)


class AnkleBraking:
    def __init__(self,*,enabled=False):
        if not isinstance(enabled,bool):raise ValueError('Explicit boolean braking opt-in required')
        self.enabled=enabled;self.limits=load_joint_limits();self.indices=[self.limits.names.index(n) for n in JOINTS]
        if not np.array_equal(self.limits.effort[self.indices],np.full(4,50.)):raise ValueError('Audited ankle50Nm physical effort caps changed')
        self.rows=[];self._last_clock=None;self.failure=None;self.saved=False
        self.terms=list(assistance(0.,0.,-.5,.5))
        self.fields=['q','dq','motor_enabled',*self.terms,'applied_assist','input_effort','output_effort_before_final_clip','expected_effort_after_existing_clip']
        self.columns=['wall_monotonic','sim_time','support','active_supervisor','native_snapshot_index','native_received_at']
        for name in JOINTS:self.columns.extend(name+'_'+field for field in self.fields)
        self.sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('limits.py'))}

    def apply(self,efforts,q,dq,*,wall,sim,support,status,snapshot,command_index=-1):
        if not self.enabled:return efforts
        try:
            before=np.asarray(efforts,float);q=np.asarray(q,float);dq=np.asarray(dq,float)
            if before.shape!=(29,) or q.shape!=(29,) or dq.shape!=(29,) or not np.isfinite(np.r_[before,q,dq,wall,sim]).all():
                raise ValueError('Invalid braking state/effort/clock')
            if not isinstance(support,(bool,np.bool_)) or not isinstance(status,dict):raise ValueError('Explicit support/command status required')
            if self._last_clock is not None and (wall<=self._last_clock[0] or sim<=self._last_clock[1]):raise ValueError('Braking clock repeated or regressed')
            self._last_clock=(float(wall),float(sim));active=not support
            modes=np.zeros(29);stamp=math.nan
            if snapshot is not None:
                modes=np.asarray(snapshot.mode,float);stamp=float(snapshot.received_at)
                if modes.shape!=(29,) or not np.isfinite(modes).all() or not np.isin(modes,[0,1]).all() or not math.isfinite(stamp) or snapshot.mode_pr!=0:
                    raise ValueError('Invalid native motor mode snapshot')
            if active and (status.get('mode')!='active' or status.get('last_rejection') is not None or snapshot is None or not 0<=wall-stamp<=.25):
                raise ValueError('Braking requires fresh accepted native command')
            after=before.copy();records=[]
            for name,i in zip(JOINTS,self.indices):
                terms=assistance(float(q[i]),float(dq[i]),float(self.limits.lower[i]),float(self.limits.upper[i]))
                delta=terms['signed_assist'] if active and modes[i] else 0.
                after[i]+=delta
                records.extend([float(q[i]),float(dq[i]),float(modes[i]),*[terms[k] for k in self.terms],delta,
                    float(before[i]),float(after[i]),float(np.clip(after[i],-self.limits.effort[i],self.limits.effort[i]))])
            if not np.isfinite(after).all():raise ValueError('Nonfinite braking result')
            self.rows.append([float(wall),float(sim),int(support),int(active),int(command_index),stamp,*records])
            return after
        except Exception as exc:
            self.failure=type(exc).__name__+': '+str(exc);raise

    def save(self,output_dir,*,native_snapshot_provenance=None):
        output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
        binary=output/'ankle_braking.npz';metadata=output/'ankle_braking.json'
        if self.saved or binary.exists() or metadata.exists():raise FileExistsError('Braking evidence already exists')
        rows=np.asarray(self.rows,float).reshape(-1,len(self.columns))
        np.savez_compressed(binary,data=rows,columns=np.asarray(self.columns),joint_names=np.asarray(JOINTS))
        effect={}
        for name in JOINTS:
            d=rows[:,self.columns.index(name+'_applied_assist')]
            pre=rows[:,self.columns.index(name+'_output_effort_before_final_clip')]
            final=rows[:,self.columns.index(name+'_expected_effort_after_existing_clip')]
            effect[name]=dict(changed_samples=int(np.count_nonzero(d)),maximum_abs_assist_nm=float(np.max(abs(d))) if len(d) else 0.,
                expected_final_clip_samples=int(np.count_nonzero(abs(pre-final)>1e-9)))
        result=dict(schema=1,kind='explicit optional experimental ankle end-stop restoring/damping supervisor',enabled=self.enabled,
            parameters=dict(lookahead_s=LOOKAHEAD_S,activation_rad=ACTIVATION_RAD,restoring_nm=RESTORING_NM,
                outward_damping_nm_per_rad_s=DAMPING_NM_PER_RAD_S,additional_one_sided_cap_nm=ADDITIONAL_CAP_NM),
            formula='d_pred=gap-.04*outward_speed;w=clip((.08-d_pred)/.08,0,1);assist_side=min(20,w*(10+4*outward_speed));signed=lower-upper',
            physical_effort_caps_nm=self.limits.effort[self.indices].tolist(),actual_position_limits_changed=False,
            final_effort_clipping='Existing engine clamp after all experimental corrections; required and unchanged',
            native_policy_or_gains_changed=False,control_interface_changed=True,additional_publisher_created=False,
            actual_containment_guaranteed=False,physical_robot_validated=False,physics_samples=len(rows),effects=effect,failure=self.failure,
            state_source='Actual current measured q/dq; caller-supplied same-tick native command snapshot/modes',
            snapshot_index_source=native_snapshot_provenance,source_sha256=self.sources,npz_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            absent_command_encoding='native_received_at NaN only in supported startup with no command; inactive supervisor adds no effort',
            limitations=['40ms projection is a heuristic, not dynamics/contact-aware stopping distance.',
                'Restoring assistance remains near a stop even during inward motion; only damping is one-sided.',
                'Actual position/speed/effort and posture guards remain authoritative; no threshold waiver.',
                'Predicted post-clip effort is logged; raw simulation trace must independently confirm applied effort.'])
        metadata.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');self.saved=True
        return result
