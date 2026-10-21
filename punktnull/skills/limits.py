"""Actual imported G1 simulator limits; no Isaac import, actuators or modifications.

The URDF supplies hard joint position stops. The retained explicit-effort runner
imports G1_CYLINDER_MODEL_12_DEX_CFG: its per-joint actuator overrides supply
speed/effort limits. These are simulator constraints, not a hardware certificate.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import ast
import hashlib
import re
import xml.etree.ElementTree as ET
import numpy as np
from .legacy import OLD, LEGACY, load

CONTRACT_ID='skill_actual_limits_v2'
POSITION_TOLERANCE_RAD=1e-6
SPEED_TOLERANCE_RAD_S=1e-6
EFFORT_TOLERANCE_NM=1e-6
# Preserve the previous global guard in addition to actual per-joint limits.
GLOBAL_SPEED_CEILING_RAD_S=35.

@dataclass(frozen=True)
class JointLimits:
    names: tuple
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    urdf_velocity: np.ndarray
    urdf_effort: np.ndarray
    sources: dict

    def check(self,q,dq,tau=None,*,require_effort=True,check_position=True):
        try:
            q=np.asarray(q,float);dq=np.asarray(dq,float)
            if q.shape!=(29,) or dq.shape!=(29,):return 'invalid_joint_state_shape'
            if not np.isfinite(np.r_[q,dq]).all():return 'nonfinite_joint_state'
            if check_position and (np.any(q<self.lower-POSITION_TOLERANCE_RAD) or np.any(q>self.upper+POSITION_TOLERANCE_RAD)):return 'joint_position_limit'
            if np.any(abs(dq)>self.velocity+SPEED_TOLERANCE_RAD_S):return 'joint_velocity_limit'
            if tau is None:return 'missing_applied_effort' if require_effort else None
            tau=np.asarray(tau,float)
            if tau.shape!=(29,) or not np.isfinite(tau).all():return 'invalid_applied_effort'
            if np.any(abs(tau)>self.effort+EFFORT_TOLERANCE_NM):return 'joint_effort_limit'
        except (TypeError,ValueError):return 'invalid_joint_state'
        return None

    def provenance(self):
        return dict(contract_id=CONTRACT_ID,joint_order='native SDK; same order as q/dq/tau trace',
            joint_names=list(self.names),position_lower_rad=self.lower.tolist(),position_upper_rad=self.upper.tolist(),
            maximum_velocity_rad_s=self.velocity.tolist(),maximum_applied_effort_nm=self.effort.tolist(),
            urdf_velocity_rad_s=self.urdf_velocity.tolist(),urdf_effort_nm=self.urdf_effort.tolist(),
            numerical_tolerances=dict(position_rad=POSITION_TOLERANCE_RAD,speed_rad_s=SPEED_TOLERANCE_RAD_S,effort_nm=EFFORT_TOLERANCE_NM),
            source_sha256=self.sources,existing_global_velocity_ceiling_rad_s=GLOBAL_SPEED_CEILING_RAD_S,
            effort_meaning='Applied/clipped explicit joint effort command; not contact force or independent torque sensor',
            model_discrepancies=[dict(joint=n,urdf_effort_nm=float(self.urdf_effort[i]),configured_effort_nm=float(self.effort[i]),
                urdf_velocity_rad_s=float(self.urdf_velocity[i]),guarded_configured_velocity_rad_s=float(self.velocity[i]))
                for i,n in enumerate(self.names) if self.urdf_effort[i]!=self.effort[i] or self.urdf_velocity[i]!=self.velocity[i]],
            limitation='Configured simulation model only; existing overrides are disclosed, never increased here; physical validity unresolved')


def _configured_actuators(path):
    tree=ast.parse(Path(path).read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='G1_CYLINDER_MODEL_12_DEX_CFG' for t in n.targets))
    config=assignment.value
    actuators=next(k.value for k in config.keywords if k.arg=='actuators')
    result=[]
    for call in actuators.values:
        values={k.arg:ast.literal_eval(k.value) for k in call.keywords if k.arg in ('joint_names_expr','velocity_limit_sim','effort_limit_sim')}
        if set(values)!= {'joint_names_expr','velocity_limit_sim','effort_limit_sim'}:raise ValueError('Incomplete actual actuator limits')
        result.append(values)
    return result

@lru_cache(maxsize=1)
def load_joint_limits():
    urdf=OLD/'gear_sonic/data/assets/robot_description/urdf/g1/main.urdf'
    config=OLD/'gear_sonic/envs/manager_env/robots/g1.py'
    bridge=load('limits_bridge',LEGACY/'bridge.py')
    names=tuple(bridge.BODY_JOINT_NAMES)
    if len(names)!=29 or len(set(names))!=29:raise ValueError('Unexpected native joint order')
    joints={j.get('name'):j for j in ET.parse(urdf).getroot().findall('joint')}
    acts=_configured_actuators(config);lower=[];upper=[];speed=[];effort=[];uv=[];ue=[]
    for name in names:
        j=joints[name];limit=j.find('limit')
        if j.get('type')!='revolute' or limit is None:raise ValueError('Missing physical joint stop '+name)
        lower.append(float(limit.get('lower')));upper.append(float(limit.get('upper')))
        uv.append(float(limit.get('velocity')));ue.append(float(limit.get('effort')))
        matching=[a for a in acts if any(re.fullmatch(pattern,name) for pattern in a['joint_names_expr'])]
        if len(matching)!=1:raise ValueError('Actuator mapping ambiguous for '+name)
        a=matching[0]
        def value(key):
            x=a[key]
            if isinstance(x,dict):
                values=[v for pattern,v in x.items() if re.fullmatch(pattern,name)]
                if len(values)!=1:raise ValueError('Limit mapping ambiguous for '+name)
                x=values[0]
            if not np.isfinite(x) or x<=0:raise ValueError('Invalid configured limit')
            return float(x)
        speed.append(min(GLOBAL_SPEED_CEILING_RAD_S,value('velocity_limit_sim')))
        effort.append(value('effort_limit_sim'))
    arrays=[np.asarray(v,float) for v in (lower,upper,speed,effort,uv,ue)]
    if not all(np.isfinite(v).all() for v in arrays) or np.any(arrays[0]>=arrays[1]):raise ValueError('Invalid model limit arrays')
    for a in arrays:a.setflags(write=False)
    return JointLimits(names,*arrays,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (urdf,config,LEGACY/'bridge.py')})


def acceptance_contract():
    return dict(id=CONTRACT_ID,version=2,
        authorization='User authorized reference position/joint tracking deviations only when actual limits remain respected',
        reference_tracking='Height and joint-reference residuals diagnostic only; no offline p95 tracking acceptance threshold',
        retained_orientation_guard='45deg relative orientation discrepancy for0.15s; actual phase permission and upright recovery retained',
        measured_progress_minimum_m=.50,stable_upright_seconds=.70,state_max_age_s=.15,
        live_human_acceptance=False,physical_robot_validation=False,limits=load_joint_limits().provenance())
