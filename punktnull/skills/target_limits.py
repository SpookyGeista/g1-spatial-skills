"""Opt-in experimental native PD target limiter; simulation interface only.

This changes selected desired positions, NOT a model/physical limit. It cannot
promise actual position containment: dynamics/contact/inertia still apply.
The caller must retain its existing final configured effort clipping and every
actual-state guard. No DDS publisher, gain/weight edit or simulator import here.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .limits import load_joint_limits

JOINTS=('left_ankle_roll_joint','right_ankle_roll_joint')
FIELDS=('q','dq','kp','kd','tau','mode')


class AnkleTargetLimiter:
    def __init__(self,*,enabled=False,margin_rad=.05,include_pitch=False,experimental_braking=False):
        if not isinstance(enabled,bool) or not isinstance(include_pitch,bool) or not isinstance(experimental_braking,bool):raise ValueError('Explicit boolean limiter/pitch opt-in required')
        if not isinstance(margin_rad,(int,float)) or not math.isfinite(margin_rad) or not 0<=margin_rad<=.08:
            raise ValueError('Ankle target interior margin must be0..0.08rad')
        from .ankle_braking import AnkleBraking
        self.experimental_braking=experimental_braking;self.braking=AnkleBraking(enabled=experimental_braking)
        self.enabled=enabled;self.include_pitch=include_pitch;self.margin_rad=float(margin_rad);self.limits=load_joint_limits()
        self.joint_names=(('left_ankle_pitch_joint','right_ankle_pitch_joint')+JOINTS) if include_pitch else JOINTS
        self.indices=np.array([self.limits.names.index(name) for name in self.joint_names],int)
        self.lower=self.limits.lower[self.indices]+self.margin_rad
        self.upper=self.limits.upper[self.indices]-self.margin_rad
        if np.any(self.lower>=self.upper):raise ValueError('No feasible desired-position interval')
        self.rows=[];self.snapshots=[];self.snapshot_times=[];self._last_snapshot=None
        self._last_clock=None;self.failure=None;self.saved=False
        self.columns=['wall_monotonic','sim_time','support','active_filter','command_snapshot_index']
        for name in self.joint_names:
            self.columns.extend(name+'_'+x for x in ('raw_desired_q','bounded_desired_q','measured_q','measured_dq',
                'kp','kd','feedforward_tau','motor_enabled','raw_effort','corrected_effort','effort_correction'))
        self.sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('limits.py'))}

    def _command(self,snapshot):
        if snapshot is None:return None,-1
        values=np.asarray([getattr(snapshot,k) for k in FIELDS],float)
        stamp=float(snapshot.received_at)
        if values.shape!=(6,29) or not np.isfinite(values).all() or not math.isfinite(stamp):
            raise ValueError('Invalid exact native command snapshot')
        if getattr(snapshot,'mode_pr',None)!=0 or not np.isin(values[5],[0,1]).all() or np.any(values[2:4]<0):
            raise ValueError('Unsupported native gains/modes')
        if self._last_snapshot is not None:
            last_stamp,last_values=self._last_snapshot
            if stamp<last_stamp:raise ValueError('Native command receipt clock regressed')
            if stamp==last_stamp:
                if not np.array_equal(values,last_values):raise ValueError('Native snapshot changed without new receipt')
                return values,len(self.snapshots)-1
        self.snapshots.append(values.copy());self.snapshot_times.append(stamp);self._last_snapshot=(stamp,values.copy())
        return values,len(self.snapshots)-1

    def apply(self,snapshot,raw_torque,q,dq,*,wall,sim,support,status):
        if not self.enabled and not self.experimental_braking:return raw_torque
        try:
            torque=np.asarray(raw_torque,float);q=np.asarray(q,float);dq=np.asarray(dq,float)
            if torque.shape!=(29,) or q.shape!=(29,) or dq.shape!=(29,) or not np.isfinite(np.r_[torque,q,dq,wall,sim]).all():
                raise ValueError('Invalid limiter input state/effort/clock')
            if self._last_clock is not None and (wall<=self._last_clock[0] or sim<=self._last_clock[1]):
                raise ValueError('Limiter clock repeated or regressed')
            self._last_clock=(float(wall),float(sim))
            if not isinstance(status,dict) or not isinstance(support,(bool,np.bool_)):
                raise ValueError('Explicit measured support and command status required')
            command,index=self._command(snapshot)
            active=not support
            if active:
                if status.get('mode')!='active' or status.get('last_rejection') is not None or command is None:
                    raise ValueError('Unsupported, stale or missing active native command')
                age=float(wall)-float(snapshot.received_at)
                if not 0<=age<=.25:raise ValueError('Native command snapshot stale/future')
                predicted=np.where(command[5]!=0,command[4]+command[2]*(command[0]-q)+command[3]*(command[1]-dq),0.)
                # Detect a wrong snapshot, post-clamp hook, changed gains, or
                # another effort source; do not silently combine controllers.
                if not np.allclose(torque,predicted,rtol=1e-10,atol=1e-8):
                    raise ValueError('Raw effort does not match the exact native PD snapshot')
            corrected=torque.copy();records=[]
            for j,ix in enumerate(self.indices):
                native_q=float(command[0,ix]) if command is not None else math.nan
                kp=float(command[2,ix]) if command is not None else math.nan
                kd=float(command[3,ix]) if command is not None else math.nan
                ff=float(command[4,ix]) if command is not None else math.nan
                enabled=float(command[5,ix]) if command is not None else 0.
                bounded=float(np.clip(native_q,self.lower[j],self.upper[j])) if self.enabled and active and enabled else native_q
                delta=kp*(bounded-native_q) if active and enabled else 0.
                corrected[ix]+=delta
                records.extend([native_q,bounded,float(q[ix]),float(dq[ix]),kp,kd,ff,enabled,float(torque[ix]),float(corrected[ix]),delta])
            if not np.isfinite(corrected).all():raise ValueError('Nonfinite corrected effort')
            self.rows.append([float(wall),float(sim),int(support),int(active),index,*records])
            return self.braking.apply(corrected,q,dq,wall=wall,sim=sim,support=support,status=status,
                snapshot=snapshot,command_index=index) if self.experimental_braking else corrected
        except Exception as exc:
            self.failure=type(exc).__name__+': '+str(exc)
            raise

    def save(self,output_dir):
        """Persist exact inputs and effects once, including failed trials."""
        output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
        binary=output/'ankle_target_limiter.npz';metadata=output/'ankle_target_limiter.json'
        if self.saved or binary.exists() or metadata.exists():raise FileExistsError('Limiter evidence already exists')
        rows=np.asarray(self.rows,float).reshape(-1,len(self.columns))
        commands=np.asarray(self.snapshots,float).reshape(-1,6,29)
        np.savez_compressed(binary,data=rows,columns=np.asarray(self.columns),native_command_snapshots=commands,
            native_command_snapshot_fields=np.asarray(FIELDS),native_command_received_at=np.asarray(self.snapshot_times),
            native_sdk_joint_names=np.asarray(self.limits.names))
        active=rows[:,3]>.5 if len(rows) else np.zeros(0,bool)
        effects={}
        for name in self.joint_names:
            change=rows[:,self.columns.index(name+'_effort_correction')]
            effects[name]=dict(corrected_physics_samples=int(np.count_nonzero(change)),
                max_abs_effort_correction_nm=float(np.max(abs(change))) if len(change) else 0.)
        braking=(self.braking.save(output,native_snapshot_provenance=dict(file=binary.name,sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
                 if self.experimental_braking else None)
        result=dict(schema=1,kind='experimental optional native desired ankle position limiter',enabled=self.enabled,include_pitch=self.include_pitch,experimental_braking=self.experimental_braking,
            corrected_effort_stage='After optional target clamp, BEFORE optional ankle braking and final engine clip',braking_evidence=braking,
            desired_interior_margin_rad=self.margin_rad,joints=list(self.joint_names),sdk_indices=self.indices.tolist(),
            desired_lower_rad=self.lower.tolist(),desired_upper_rad=self.upper.tolist(),
            actual_position_lower_rad=self.limits.lower[self.indices].tolist(),actual_position_upper_rad=self.limits.upper[self.indices].tolist(),
            physics_samples=len(rows),unsupported_active_filter_samples=int(active.sum()),native_snapshots=len(commands),effects=effects,
            failure=self.failure,npz_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),filter_source_sha256=self.sources,
            configured_limits=self.limits.provenance(),equation='corrected_effort = raw_native_effort + kp*(bounded_desired_q - raw_desired_q) for selected enabled motors',
            native_parameters_unchanged=['mode','mode_pr','feedforward tau','desired dq','kp','kd','policy/planner weights'],
            final_configured_effort_clipping='Required in caller after this filter; not increased or removed by this module',
            scope='Simulation-only experimental control-interface change; not original unchanged native behavior or physical-limit proof',
            absent_command_encoding='snapshot_index=-1; raw desired/gains are NaN during supported startup without native snapshot; measured state/efforts remain finite',
            actual_containment_guaranteed=False,physical_robot_validated=False)
        metadata.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');self.saved=True
        return result
