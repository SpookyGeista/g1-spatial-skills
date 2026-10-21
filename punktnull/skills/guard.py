"""Drop-in measured guard for the retained 200Hz loop during owned POSE skills."""
import json
from pathlib import Path
import numpy as np
from .legacy import ground_module
from .reference import ReferenceClip
from .limits import load_joint_limits, CONTRACT_ID
from .phase import SkillReference

_old=ground_module()
LIMITS={**_old.LIMITS,'submitted_to_measured_search_frames_before':12,'submitted_to_measured_search_frames_after':4,
        'acceptance_contract':CONTRACT_ID,'reference_height_and_joint_error_acceptance':'diagnostic_only',
        'actual_position_numeric_tolerance_rad':1e-6,
        'supported_initialization_seconds_max':120.,'supported_initialization_position_excess_max_rad':1e-4}
ACTIVE_GUARD=None


class NativeStreamFeedback:
    def __init__(self,folder):
        self.tails={key:_old.CsvTail(Path(folder)/(key+'.csv')) for key in ('motion_name','motion_playing','encoder_mode')}
        self.pending={key:{} for key in self.tails};self.latest=None;self.last_index=None

    def poll(self, now):
        for key,tail in self.tails.items():
            for row in tail.rows():self.pending[key][int(row['index'])]=row
        joined=set.intersection(*(set(v) for v in self.pending.values()))
        for index in sorted(joined):
            rows={key:items.pop(index) for key,items in self.pending.items()}
            if self.last_index is not None and index!=self.last_index+1:raise ValueError('Native feedback gap')
            stamps=[float(r['time_monotonic_ms'])/1000 for r in rows.values()]
            playing=float(rows['motion_playing']['playing_0']);mode=float(rows['encoder_mode']['encoder_mode_0'])
            if not np.isfinite([*stamps,playing,mode]).all():raise ValueError('Nonfinite native feedback')
            if max(stamps)-min(stamps)>1e-5:raise ValueError('Native feedback mismatched timestamps')
            if self.latest and stamps[0]<=self.latest['monotonic']:raise ValueError('Native feedback clock regressed')
            if playing not in (0.,1.) or mode not in (0.,1.,2.):raise ValueError('Invalid native mode flags')
            self.latest={'monotonic':stamps[0],'motion':rows['motion_name']['motion_name'],
                         'playing':playing>.5,'encoder_mode':int(mode),'index':index}
            self.last_index=index
        return self.latest


class GroundGuard:
    columns=_old.GroundGuard.columns
    def __init__(self,reference,controller_logs):
        global ACTIVE_GUARD
        self.clip=ReferenceClip.load(reference);self.reference=SkillReference(self.clip);self.joint_limits=load_joint_limits()
        self.feedback=NativeStreamFeedback(controller_logs)
        self.authorization_path=Path(controller_logs).parent/'skill_stream.json'
        self.last=[-1,0,0,0,0];self.bad_since=None;self.failure=None;self.native=None
        self._last_poll=-float('inf');self._authorization=None;self._token=None
        self._startup_started=None;self._startup_closed=False;self.startup_position_diagnostic=None
        self._last_owned_phase=None;self._release_seen=None;self._retired_tokens=set()
        ACTIVE_GUARD=self

    def assess(self,now,xyz,quat,joints,velocities,support=False,*,efforts=None,state_monotonic=None,command_status=None):
        self.last=[-1,0,0,0,0]
        try:
            xyz=np.asarray(xyz,float);quat=np.asarray(quat,float)
            if xyz.shape!=(3,) or quat.shape!=(4,):return 'invalid_ground_state_shape'
            if not np.isfinite(np.r_[now,xyz,quat]).all():return 'nonfinite_ground_state'
            if abs(np.linalg.norm(quat)-1)>.02:return 'invalid_ground_quaternion'
            if self._startup_started is None:self._startup_started=now
            status=command_status if isinstance(command_status,dict) else {}
            count=status.get('accepted_count');mode=status.get('mode')
            initializing=(status.get('rejected_count')==0 and isinstance(count,int)
                and ((mode=='waiting' and count==0) or (mode=='active' and count>0)))
            if not support or (command_status is not None and not initializing):self._startup_closed=True
            startup=(support and initializing and not self._startup_closed and 0<=now-self._startup_started<=120.)
            # Visible supported initialization may show tiny PhysX stop residuals
            # before the strict pre-release check. This bounded exception cannot
            # extend into any unsupported step or mask a gross position error.
            strict_error=self.joint_limits.check(joints,velocities,efforts,require_effort=not support or startup)
            self.startup_position_diagnostic=strict_error if startup else None
            tiny_position_residual=(startup and strict_error=='joint_position_limit'
                and np.max(np.maximum(self.joint_limits.lower-np.asarray(joints,float),
                                      np.asarray(joints,float)-self.joint_limits.upper))<=1e-4)
            error=(self.joint_limits.check(joints,velocities,efforts,require_effort=True,check_position=False)
                   if tiny_position_residual else strict_error)
            if error:return error
            if not support and (state_monotonic is None or not np.isfinite(state_monotonic) or not -.02<=now-state_monotonic<=.15):
                return 'stale_or_missing_measured_state'
            if xyz[2]<LIMITS['absolute_min_root_height_m']:return 'absolute_ground_state_limit'
        except (ValueError,TypeError):return 'invalid_ground_state'
        if now-self._last_poll>=.01:
            self._last_poll=now
            try:
                self._authorization=json.loads(self.authorization_path.read_text())
                if not isinstance(self._authorization,dict):return 'invalid_skill_authorization'
            except (OSError,ValueError):self._authorization=None
            try:self.native=self.feedback.poll(now)
            except (ValueError,KeyError):return 'invalid_native_skill_feedback'
        if support:return None
        auth=self._authorization
        upright=xyz[2]>=LIMITS['upright_min_height_m'] and _old.tilt_degrees(quat)<=LIMITS['upright_max_tilt_deg']
        if not auth or auth.get('owner')!='skill':
            # A missing/stale file never relinquishes the previous owner. A
            # bound COMPLETE release must be observed while still fresh, then
            # actual native planner acknowledgement retires that session.
            if auth and self._token is not None and auth.get('phase')=='COMPLETE':
                stamp=auth.get('monotonic')
                if (self._release_seen is None and self._last_owned_phase=='RETURN_READY'
                    and auth.get('released_owner_token')==self._token
                    and auth.get('source_digest')==self.clip.digest and auth.get('source')==str(self.clip.source)
                    and isinstance(stamp,(int,float)) and np.isfinite(stamp) and -.02<=now-stamp<=.15):
                    self._release_seen=float(stamp)
                native=self.native
                stable_upright=xyz[2]>=.60 and _old.tilt_degrees(quat)<=25 and np.sqrt(np.mean(np.asarray(velocities,float)**2))<=.20
                if (self._release_seen is not None and stable_upright and native
                    and -.02<=now-native['monotonic']<=.15 and native['monotonic']>=self._release_seen
                    and native['motion']=='planner_motion' and native['encoder_mode']==0 and native['playing']):
                    self._retired_tokens.add(self._token);self._token=None
                    self._last_owned_phase=None;self._release_seen=None;self.bad_since=None
            return None if upright else 'ground_without_skill_owner'
        if auth.get('phase')=='FAILED':return 'skill_supervisor_failed'
        if auth.get('source_digest')!=self.clip.digest or auth.get('source')!=str(self.clip.source):return 'skill_reference_identity_mismatch'
        token=auth.get('owner_token')
        if not isinstance(token,str) or not token:return 'invalid_skill_owner_token'
        try:authorization_age=now-float(auth.get('monotonic',0))
        except (TypeError,ValueError):return 'invalid_skill_authorization_clock'
        if not -.02<=authorization_age<=.15:return 'stale_skill_supervisor'
        if not auth.get('clearance_ok'):return 'skill_clearance_not_confirmed'
        if token in self._retired_tokens or (self._release_seen is not None and token==self._token):return 'retired_skill_owner_token'
        if self._token is None:
            if self._retired_tokens and auth.get('phase')!='PRIMING':return 'new_skill_owner_requires_priming'
            self._token=token
        elif token!=self._token:return 'skill_owner_changed_mid_execution'
        self._last_owned_phase=auth.get('phase')
        if auth.get('phase')=='PRIMING':return None if upright else 'ground_before_native_skill_ack'
        if auth.get('phase') not in ('ALIGNING','ENTERING','GROUND','EXITING','SETTLING','RETURN_READY'):
            return 'invalid_skill_phase'
        native=self.native
        if not native or not -.02<=now-native['monotonic']<=.15:return 'stale_native_skill_feedback'
        if native['motion']!='streamed' or native['encoder_mode']!=0 or not native['playing']:
            return 'native_reference_stream_not_active'
        frame=auth.get('submitted_frame');heading=auth.get('heading_alignment_rad')
        if type(frame) is not int or not 0<=frame<self.clip.frames or not isinstance(heading,(int,float)) or not np.isfinite(heading):
            return 'invalid_owned_reference_cursor'
        # This is bounded measured phase inference, NOT an invented native frame
        # counter. Full raw state and submitted cursor remain available for audit.
        candidates=range(max(0,frame-12),min(self.clip.frames,frame+5))
        errors=[self.reference.errors(i,xyz,quat,joints,heading) for i in candidates]
        scores=np.sum((np.asarray(errors)/np.array([.20,45.,.65]))**2,axis=1)
        selected=int(np.argmin(scores));index=list(candidates)[selected];err=errors[selected]
        permission=bool(self.reference.permission[index]);self.last=[index,int(permission),*err]
        # Height/joint reference residuals remain logged, not acceptance gates.
        # Keep the orientation discrepancy guard to reject a sustained wrong-way phase.
        excessive=err[1]>LIMITS['online_orientation_error_deg']
        if excessive:
            if self.bad_since is None:self.bad_since=now
            if now-self.bad_since>=.15:return 'sustained_stream_reference_orientation_error'
        else:self.bad_since=None
        if not permission and not upright:return 'unexpected_ground_outside_reference_phase'
        return None
