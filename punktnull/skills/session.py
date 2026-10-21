"""Bounded measured-state skill contract; no motor publication or scene edits."""
from dataclasses import dataclass
import math
import secrets
import json
from pathlib import Path
import numpy as np
from .legacy import OLD, WORKSPACE, ground_module
from .limits import load_joint_limits, CONTRACT_ID
from .phase import SkillReference
from .reference import ReferenceClip
from .platform120_spec import PLATFORM120_SPEC


@dataclass(frozen=True)
class SkillSpec:
    name: str
    status: str
    source: str | None
    permitted_backend: str | None
    evidence: str


REGISTRY = {
    'walk': SkillSpec('walk','validated_bounded_upright_sim',None,'retained_sonic_planner',
                     'Retained B10/10 upright RGB-D tasks; not terrain-skill validation'),
    'crawl': SkillSpec('crawl','experimental_actual_limits_v2',
        str(OLD/'data/bones_seed_crawl_extension_2026-09-16/02_crawl_only'),'native_pose_v1',
        'Prospective user-approved actual-limits v2; reference tracking diagnostic; archived passage position-stop excursions retained as failures; physical validation absent'),
    'stairs': SkillSpec('stairs','failed_two_riser_sim',None,None,
        'Actual stairs001-004 failed120/144mm two-riser contract; separate platform120 is a bounded single-riser component only'),
    'platform120': SkillSpec('platform120',PLATFORM120_SPEC['status'],
        PLATFORM120_SPEC['source'],PLATFORM120_SPEC['permitted_backend'],
        'Actual platform002-004:3/3 isolated120mm ascent+full-foot landing+native planner return, timing/uncutPASS; no autonomous mission admission; evidence/platform/three_run_summary.json'),
}


@dataclass(frozen=True)
class ClearanceEvidence:
    monotonic: float
    source: str
    corridor_verified: bool
    entry_and_exit_verified: bool
    minimum_margin_m: float
    fixture_id: str

    def check(self, now):
        if self.source not in ('measured_sensor_geometry','disclosed_simulation_fixture'):
            return 'clearance_source_unidentified'
        if not math.isfinite(self.monotonic) or not -.02 <= now-self.monotonic <= .25:
            return 'clearance_evidence_stale'
        if not self.fixture_id or not self.corridor_verified or not self.entry_and_exit_verified:
            return 'corridor_or_standing_exit_not_verified'
        if not math.isfinite(self.minimum_margin_m) or self.minimum_margin_m < .08:
            return 'insufficient_validated_clearance_margin'
        return None


def measured_error(state, now, *, upright=False):
    try:
        stamp=float(state['monotonic']);base=np.asarray(state['base'],float)
        q=np.asarray(state['q'],float);dq=np.asarray(state['dq'],float)
        if base.shape != (13,) or q.shape != (29,) or dq.shape != (29,): return 'invalid_measured_shape'
        if not np.isfinite(np.r_[stamp,base,q,dq]).all(): return 'nonfinite_measured_state'
        if not -.02 <= now-stamp <= .15:return 'stale_measured_state'
        if state['support']:return 'startup_support_not_released'
        if abs(np.linalg.norm(base[3:7])-1)>.02:return 'invalid_measured_quaternion'
        limits_error=load_joint_limits().check(q,dq,state.get('tau'),require_effort=True)
        if limits_error:return limits_error
        if base[2]<.025:return 'absolute_state_limit'
        if upright and (base[2]<.60 or float(ground_module().tilt_degrees(base[3:7]))>25
                        or np.linalg.norm(base[7:10])>.10 or np.sqrt(np.mean(dq*dq))>.20):
            return 'not_stable_upright'
    except (KeyError,TypeError,ValueError):return 'missing_or_invalid_state'
    return None


class SkillSession:
    """Explicit ownership + measured ground traversal and upright recovery.

    The caller feeds the actual simulator/native guard result, not a mock timer.
    It must pause physics and stop the controller on FAILED. There is deliberately
    no IDLE fallback or automatic ownership hand-back during a ground phase.
    """
    def __init__(self, clip, *, skill='crawl', experimental=False, allow_generated=False):
        spec=REGISTRY[skill]
        if spec.status=='references_only' or spec.permitted_backend!='native_pose_v1':
            raise ValueError('Skill has no executable validated backend')
        if not experimental:raise ValueError('Experimental crawl must be explicitly selected for this trial')
        # Always re-read and verify channel/source hashes, including local derivatives.
        clip=ReferenceClip.load(clip.source)
        if str(clip.source)==spec.source:
            if clip.digest!='7b068d903e6049d16a956b19b867bb4d59ca4775fe2c32dc3e23445113b27191':
                raise ValueError('Registered reference content changed')
            self.generated_reference=None
        else:
            if not allow_generated or skill!='crawl':raise ValueError('Reference differs from registered capability')
            if not any(clip.source.is_relative_to(WORKSPACE/p) for p in ('assets','evidence')):
                raise ValueError('Generated crawl outside authorized workspace')
            generated=json.loads((clip.source/'generated_reference.json').read_text())
            baseline=ReferenceClip.load(spec.source)
            if generated.get('kind')!='generated_g1_skill_reference' or generated.get('skill')!='crawl' or generated.get('physical_execution') is not False:
                raise ValueError('Not an explicitly unvalidated generated crawl derivative')
            expected={str(baseline.source/name):sha for name,sha in baseline.hashes.items()}
            if any(generated.get('sources_sha256',{}).get(name)!=sha for name,sha in expected.items()):
                raise ValueError('Generated crawl lacks complete registered source provenance')
            limits=load_joint_limits();perm=ground_module().MJ_TO_IL
            if any(limits.check(q,dq,require_effort=False) for q,dq in zip(clip.data['joint_pos'][:,perm],clip.data['joint_vel'][:,perm])):
                raise ValueError('Generated crawl reference exceeds actual position/speed limits')
            self.generated_reference=generated
        self.clip=clip;self.reference=SkillReference(clip)
        self.phase='READY';self.token=None;self.started=None;self.heading=None
        self.ground_observed=False;self.ground_start_xy=None;self.ground_displacement=0.
        self.stable_since=None;self.error=None;self.last_state_time=None;self.last_inferred_frame=-1
        self.events=[];self.clearance=None
        self.native_ack_time=None
        self.aligned_since=None;self.released_token=None

    def acquire(self, owner, state, clearance, now):
        if self.phase!='READY' or owner!='autonomy':raise RuntimeError('Acquire requires exclusive autonomous owner')
        error=measured_error(state,now,upright=True) or clearance.check(now)
        if error:raise ValueError(error)
        if float(state.get('stable_full_rate_sim_seconds',0))<.70:raise ValueError('Stable full-rate entry window required')
        self.token=secrets.token_hex(16);self.phase='PRIMING';self.started=now
        self.heading=float(ground_module().yaw(state['base'][3:7])-ground_module().yaw(self.clip.quaternion[0]))
        self.clearance=clearance;self.events.append({'event':'skill_acquired','monotonic':now,'source':clearance.source})
        return self.token

    def fail(self, reason, now):
        if self.phase!='FAILED':self.events.append({'event':'skill_failed','monotonic':now,'reason':reason})
        self.phase='FAILED';self.error=reason
        return {'action':'PAUSE_SIM_AND_STOP_NATIVE','reason':reason,'return_owner':False}

    def update(self, token, state, clearance, now, *, native_stream_active, submitted_frame,
               inferred_frame=None, tracking_error=None, guard_failure=None, finished_sending=False):
        if self.phase=='FAILED':return self.fail(self.error,now)
        if token!=self.token or not self.token:return self.fail('skill_ownership_mismatch',now)
        error=measured_error(state,now) or clearance.check(now) or guard_failure
        if error:return self.fail(error,now)
        if now-self.started>self.clip.seconds+20:return self.fail('skill_timeout',now)
        self.clearance=clearance
        if self.phase=='PRIMING':
            if native_stream_active:
                self.phase='ALIGNING';self.events.append({'event':'native_reference_stream_acknowledged','monotonic':now})
                self.native_ack_time=now
            elif now-self.started>5:return self.fail('native_stream_ack_timeout',now)
            return {'action':'PRIME_START_POSE'}
        if not native_stream_active:return self.fail('native_skill_source_lost',now)
        if self.phase=='ALIGNING':
            if submitted_frame!=0:return self.fail('reference_advanced_before_measured_alignment',now)
            initial_errors=np.asarray(self.reference.errors(0,state['base'][:3],state['base'][3:7],state['q'],self.heading))
            aligned=(measured_error(state,now,upright=True) is None and initial_errors[1]<=12.)
            if aligned:
                if self.aligned_since is None:self.aligned_since=now
                if now-self.aligned_since>=.70:
                    self.phase='ENTERING';self.events.append({'event':'measured_reference_entry_aligned','monotonic':now,'errors':initial_errors.tolist()})
                    return {'action':'ADVANCE_REFERENCE'}
            else:self.aligned_since=None
            if now-self.native_ack_time>8:return self.fail('reference_entry_alignment_timeout',now)
            return {'action':'PRIME_START_POSE'}
        if inferred_frame is None or inferred_frame<0 or tracking_error is None:
            if self.phase=='ENTERING' and now-self.native_ack_time<=.20:return {'action':'PRIME_START_POSE'}
            return self.fail('missing_measured_tracking_evidence',now)
        if not (0 <= inferred_frame < self.clip.frames) or inferred_frame>submitted_frame+4:
            return self.fail('invalid_measured_reference_phase',now)
        values=np.asarray(tracking_error,float)
        if values.shape!=(3,) or not np.isfinite(values).all():return self.fail('invalid_tracking_error',now)
        self.last_inferred_frame=max(self.last_inferred_frame,int(inferred_frame))
        base=np.asarray(state['base']);tilt=float(ground_module().tilt_degrees(base[3:7]))
        if self.reference.ground[inferred_frame] and base[2]<.45 and tilt>45:
            self.phase='GROUND';self.ground_observed=True
            if self.ground_start_xy is None:self.ground_start_xy=base[:2].copy()
            self.ground_displacement=max(self.ground_displacement,float(np.linalg.norm(base[:2]-self.ground_start_xy)))
        elif self.ground_observed and inferred_frame>np.flatnonzero(self.reference.ground)[-1]:self.phase='EXITING'
        if finished_sending:
            self.phase='SETTLING'
            if not self.ground_observed:return self.fail('reference_sent_without_observed_ground_motion',now)
            stable=(measured_error(state,now,upright=True) is None and float(state.get('stable_full_rate_sim_seconds',0))>=.70)
            if stable:
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=.70 and self.ground_displacement>=.50:
                    self.phase='RETURN_READY';return {'action':'CHECK_FRESH_MAP_AND_EXPLICIT_REJOIN','return_owner':False}
            else:self.stable_since=None
            return {'action':'HOLD_FINAL_REFERENCE','return_owner':False}
        return {'action':'ADVANCE_REFERENCE','return_owner':False}

    def release(self, token, *, fresh_map_safe, explicit_release):
        if token!=self.token or self.phase!='RETURN_READY' or not fresh_map_safe or not explicit_release:
            raise RuntimeError('Measured completion, fresh map and explicit ownership release required')
        self.released_token=self.token;self.phase='COMPLETE';self.token=None
        return 'autonomy'

    def authorization(self, now, stream):
        return {'schema':2,'acceptance_contract':CONTRACT_ID,'monotonic':now,'owner':'skill' if self.token else 'none','owner_token':self.token,
                'phase':self.phase,'released_owner_token':self.released_token,'source_digest':self.clip.digest,'source':str(self.clip.source),
                'heading_alignment_rad':self.heading,'submitted_frame':stream.source_frame,
                'cursor_basis':'submitted, not native execution frame','clearance_ok':bool(self.clearance and not self.clearance.check(now)),
                'clearance_source':self.clearance.source if self.clearance else None,'failure':self.error}
