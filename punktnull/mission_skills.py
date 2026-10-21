"""Known-fixture skill gates around the retained observed-space navigator.

Fixture coordinates select a supported skill. They are not obstacle perception
or general terrain understanding. All walking still uses the observed RGB-D map.
"""
from copy import deepcopy
from functools import lru_cache
import json
import math
import os
import time
import numpy as np
from .inspection import InspectionExplorer
from .retained import prepare
prepare()
from navigation import Explorer, NavCommand


class CrawlInspectionExplorer(InspectionExplorer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from .terrain.inspection_passage import MISSION_CONTRACT
        self.skill_contract=deepcopy(MISSION_CONTRACT)
        self.inspection_plan=deepcopy(self.inspection_plan)
        for room in self.inspection_plan['rooms']:
            if room['id']=='north':room['observation_xy_m']=self.skill_contract['north_scan_xy_m']
        self.pending_skill=None;self.completed_skills=[];self._return_after_skill=False

    def finish_scan(self):
        super().finish_scan()
        if (self.room_index<len(self.inspection_plan['rooms']) and self.room_index>=0 and
                self.inspection_plan['rooms'][self.room_index]['id']=='north' and
                not any(s['id']==self.skill_contract['ingress']['id'] for s in self.completed_skills)):
            self._select_gate('ingress')

    def _select_gate(self,direction):
        self.pending_skill=deepcopy(self.skill_contract[direction])
        self.pending_skill['direction']=direction
        self.set_target(self.pending_skill['approach_xy_m'])
        self.room_scan=False

    def return_to_start(self):
        if (any(s['id']==self.skill_contract['ingress']['id'] for s in self.completed_skills) and
                not any(s['id']==self.skill_contract['egress']['id'] for s in self.completed_skills)):
            self._return_after_skill=True;self._select_gate('egress')
        else:super().return_to_start()

    def step(self,position_xy,yaw,now,pose_time=None):
        if self.pending_skill is None:return super().step(position_xy,yaw,now,pose_time)
        if (self.inject_review_after_room and not self.injected_review_resolved and
                any(r['id']==self.inject_review_after_room for r in self.completed_rooms)):
            return super().step(position_xy,yaw,now,pose_time)
        command=Explorer.step(self,position_xy,yaw,now,pose_time)
        if command.status=='TARGET_REACHED':
            self.status='CRAWL_REQUIRED';self.reason='Known low service passage; measured entry and native skill ownership required'
            return NavCommand(float(yaw),status=self.status,reason=self.reason)
        return command

    def finish_skill(self,position_xy,now,evidence):
        if not self.pending_skill:raise RuntimeError('No pending crawl gate')
        if evidence.get('gate_id')!=self.pending_skill['id'] or not evidence.get('observed_exit'):
            raise RuntimeError('Skill cannot complete from cursor or waypoint alone')
        if not Explorer.reanchor_from_handover(self,position_xy,now):
            raise RuntimeError('No fresh observed map connection after crawl')
        self.completed_skills.append(deepcopy(evidence));self.pending_skill=None
        if self._return_after_skill:
            self._return_after_skill=False;super().return_to_start()
        else:self.set_target(self.inspection_plan['rooms'][self.room_index]['observation_xy_m'])
        self.room_scan=False;self.scan_pending=False

    def reanchor_from_handover(self,position_xy,now):
        gate=deepcopy(self.pending_skill)
        okay=super().reanchor_from_handover(position_xy,now)
        if okay and gate:
            self.pending_skill=gate;self.set_target(gate['approach_xy_m'])
        return okay


def entry_errors(state,gate):
    xy=np.asarray(state['base'][:2],float)
    distance=float(np.linalg.norm(xy-np.asarray(gate['approach_xy_m'],float)))
    difference=(float(state['yaw_deg'])-gate['heading_deg']+180)%360-180
    return {'position_error_m':distance,'heading_error_deg':abs(difference),
            'accepted':distance<=gate['entry_tolerance_m'] and abs(difference)<=gate['heading_tolerance_deg']}


def observed_exit(state,gate):
    x,y=map(float,state['base'][:2]);region=gate['exit_region']
    return region['xmin']<=x<=region['xmax'] and region['ymin']<=y<=region['ymax']


@lru_cache(maxsize=1)
def _collision_fk():
    from .terrain.kinematics import CollisionFK
    from .retained import URDF
    from .skills.limits import load_joint_limits
    return CollisionFK(URDF,load_joint_limits().names)


def configured_exit_colliders(variant,target_room,target_absent):
    """Called before worker creation; mirror the retained child scene options."""
    from .terrain.inspection_passage import collision_boxes
    values={'NULL_SCENE_VARIANT':str(variant),'NULL_TARGET_ROOM':target_room,
            'NULL_TARGET_PRESENT':'0' if target_absent else '1'}
    prior={k:os.environ.get(k) for k in values}
    try:
        os.environ.update(values)
        return collision_boxes()
    finally:
        for k,v in prior.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def observed_exit_geometry(state,gate,boxes,minimum_margin=.08):
    """Actual joints/pose versus declared fixed colliders; a simulator veto."""
    fk=_collision_fk();base=np.asarray(state['base'],float);q=np.asarray(state['q'],float)
    bounds=fk.bounds(fk.links(base[None,:3],base[None,3:7],q[None,:]))[0]
    margin=float('inf');closest=None
    for box in boxes:
        gap=np.maximum(np.maximum(bounds[:,0]-box['aabb_max'],np.asarray(box['aabb_min'])-bounds[:,1]),0.)
        distance=float(np.linalg.norm(gap,axis=-1).min())
        if distance<margin:margin=distance;closest=box['prim']
    inside=observed_exit(state,gate)
    return dict(accepted=inside and margin>=minimum_margin,inside_exit_region=inside,
        minimum_nonfloor_aabb_clearance_m=margin,required_margin_m=minimum_margin,closest_collider=closest,
        source='Measured PhysX root/joints + retained G1 collision FK against disclosed fixed scene; no perceived terrain')


def wrapped_angle(value):
    return (value+math.pi)%(2*math.pi)-math.pi


class CrawlMissionMixin:
    """Use the mission's one existing publisher for autonomous reference skills."""
    def _skill_native(self,name):
        state=self.truth_state or {}
        feedback=state.get('native_feedback')
        return bool(feedback and -.02<=time.monotonic()-feedback['monotonic']<=.15 and
                    feedback['motion']==name and feedback['encoder_mode']==0 and feedback['playing'])

    def _align_skill_entry(self,gate):
        from .skills.session import measured_error
        self.transition('SKILL_APPROACH','Standplatz und Blickrichtung vor niedriger Passage prüfen')
        heading=math.radians(gate['heading_deg']);until=time.monotonic()+30
        stable_since=None;self.command_stabilizer.reset()
        while time.monotonic()<until:
            self.tick()
            if 'takeover' in self.commands:
                self.commands.remove('takeover');self.takeover()
                self.transition('SKILL_APPROACH','Nach Bedienerrückgabe Standplatz und Richtung erneut prüfen')
                until=time.monotonic()+30;stable_since=None;self.command_stabilizer.reset()
            estimated=entry_errors(self.state,gate)
            actual=entry_errors(self.truth_state,gate)
            # All motion targets use FAST-LIO. Truth is a separate veto for the
            # explicitly configured physical fixture, never a pose correction.
            offset=np.asarray(gate['approach_xy_m'])-np.asarray(self.state['base'][:2])
            yaw=math.radians(self.state['yaw_deg'])
            if estimated['position_error_m']>gate['entry_tolerance_m']*.8:
                angle=float(math.atan2(offset[1],offset[0]))
                if abs(wrapped_angle(angle-yaw))>.20:
                    self.command_stabilizer.reset()
                    self.client.hold(yaw+float(np.clip(1.6*wrapped_angle(angle-yaw),-.55,.55)))
                else:
                    navigation_now=max(self.state['sim_time'],self.last_sensor['sim_time'])
                    command=self.command_stabilizer.filter(
                        NavCommand(angle,.20,status='NAVIGATING',reason='Observed final skill approach'),
                        self.state['base'][:2],navigation_now)
                    if command.status!='NAVIGATING':
                        self.client.hold(yaw)
                        raise RuntimeError('Crawl approach refused: '+command.status+': '+command.reason)
                    self.client.set_command(command.world_heading,command.speed)
                    self.log_command(command.world_heading,command.speed,'SKILL_APPROACH',command.to_dict())
                stable_since=None
            else:
                self.command_stabilizer.reset()
                self.client.hold(yaw+float(np.clip(1.6*wrapped_angle(heading-yaw),-.55,.55)))
                stable=measured_error(self.truth_state,time.monotonic(),upright=True) is None
                if actual['accepted'] and estimated['accepted'] and stable:
                    if stable_since is None:stable_since=time.monotonic()
                    if time.monotonic()-stable_since>=.7:
                        self.event('skill_entry_admitted',gate_id=gate['id'],estimated=estimated,actual_fixture_veto=actual,
                                   steering_source='FAST-LIO',clearance_source='disclosed fixed simulation fixture')
                        return
                else:stable_since=None
        self.client.hold(math.radians(self.state['yaw_deg']))
        raise RuntimeError('Crawl entry position/heading/stability not achieved within30s; no skill launched')

    def run_crawl_gate(self):
        from .skills.reference import ReferenceClip,ReferencePoseStream,attach_reference_validation
        from .skills.session import SkillSession,ClearanceEvidence,measured_error
        from .skills.limits import load_joint_limits
        gate=deepcopy(self.explorer.pending_skill)
        if gate is None:raise RuntimeError('Missing observed crawl approach')
        self._align_skill_entry(gate)
        clip=ReferenceClip.load(self.crawl_reference.source)
        session=SkillSession(clip,experimental=True,allow_generated=self.crawl_generated);stream=ReferencePoseStream(clip)
        contract=self.explorer.skill_contract
        def clearance():
            return ClearanceEvidence(time.monotonic(),'disclosed_simulation_fixture',True,True,
                                     contract['minimum_margin_m'],contract['fixture_id'])
        def publish_authorization():
            authorization=session.authorization(time.monotonic(),stream)
            self._skill_atomic(self.output/'skill_stream.json',authorization)
            with (self.output/('skill_authorization_'+gate['id']+'.jsonl')).open('a') as out:
                out.write(json.dumps(authorization,allow_nan=False)+'\n')
        prior_validator=self.client._validate_pose
        attach_reference_validation(self.client)
        token=session.acquire('autonomy',self.truth_state,clearance(),time.monotonic())
        self.skill_session=session;self.skill_active=True;self.skill_gate=gate
        publish_authorization()
        self.client.request_pose()
        self.event('mission_skill_started',gate_id=gate['id'],reference=clip.provenance(),
                   limits=load_joint_limits().provenance(),source='autonomous selected existing reference',
                   scene_reset=False,reference_tracking_is_acceptance_gate=False)
        next_packet=0.;started=time.monotonic()
        trace_file=self.output/('submitted_reference_'+gate['id']+'.jsonl')
        while session.phase!='RETURN_READY':
            self.transition('SKILL_'+session.phase,'Kriechpassage: '+gate['direction'],'autonomous crawl / native reference stream')
            self.tick();now=time.monotonic()
            last=self.truth_state.get('ground_guard_last') or [-1,0,0,0,0]
            action=session.update(token,self.truth_state,clearance(),now,
                native_stream_active=self._skill_native('streamed'),submitted_frame=stream.source_frame,
                inferred_frame=last[0],tracking_error=last[2:],finished_sending=stream.finished_sending)
            publish_authorization()
            if action['action']=='PAUSE_SIM_AND_STOP_NATIVE':raise RuntimeError(action['reason'])
            if now>=next_packet:
                self.client.offer_pose(stream.packet(now,advance=action['action']=='ADVANCE_REFERENCE'))
                next_packet=now+.02
                with trace_file.open('a') as out:out.write(json.dumps(stream.status(),allow_nan=False)+'\n')
        exit_geometry=observed_exit_geometry(self.truth_state,gate,self.skill_collision_boxes)
        if not exit_geometry['accepted']:raise RuntimeError('Standing recovery failed exit/geometry check: '+json.dumps(exit_geometry))
        self.event('mission_skill_standing_exit',gate_id=gate['id'],base=self.truth_state['base'][:7],
                   ground_displacement_m=session.ground_displacement,session_events=session.events,exit_geometry=exit_geometry)
        footprint=self.grid.begin_stable_episode(self.state['base'][:2],self.state['sim_time'],
            gate_id=gate['id'],exit_geometry=exit_geometry,
            stable_seconds=self.truth_state.get('stable_full_rate_sim_seconds',0),
            support_free=not self.truth_state['support'])
        self.event('skill_stable_footprint_episode',**footprint)
        # Continue the final endpoint while new sensor observations establish a
        # safe standing map connection. No map reset or invented free cells.
        self.skill_mapping_enabled=True;until=time.monotonic()+8
        while True:
            self.tick();now=time.monotonic()
            last=self.truth_state.get('ground_guard_last') or [-1,0,0,0,0]
            action=session.update(token,self.truth_state,clearance(),now,
                native_stream_active=self._skill_native('streamed'),submitted_frame=stream.source_frame,
                inferred_frame=last[0],tracking_error=last[2:],finished_sending=True)
            publish_authorization()
            if action['action']=='PAUSE_SIM_AND_STOP_NATIVE':raise RuntimeError(action['reason'])
            if now>=next_packet:
                self.client.offer_pose(stream.packet(now,advance=False));next_packet=now+.02
                with trace_file.open('a') as out:out.write(json.dumps(stream.status(),allow_nan=False)+'\n')
            pose=np.asarray(self.state['base'][:2]);cell=self.grid.world_to_cell(pose)
            fresh=(self.last_sensor is not None and time.monotonic()-self.last_sensor['monotonic']<=.8 and
                   0<=self.state['sim_time']-self.grid.last_update<=self.grid.config.stale_after)
            if (session.phase=='RETURN_READY' and fresh and self.grid.in_bounds(cell) and
                    self.grid.layers()['safe'][cell[1],cell[0]]):break
            if time.monotonic()>until:raise RuntimeError('No fresh observed safe footprint after crawl; planner rejoin refused')
        release_error=measured_error(self.truth_state,time.monotonic(),upright=True)
        exit_geometry=observed_exit_geometry(self.truth_state,gate,self.skill_collision_boxes)
        if (release_error or self.truth_state.get('stable_full_rate_sim_seconds',0)<.70 or
                not exit_geometry['accepted']):
            raise RuntimeError('Crawl planner rejoin refused: current stable exit lost: '+str(release_error))
        self.event('mission_skill_release_checked',gate_id=gate['id'],exit_geometry=exit_geometry,
                   stable_full_rate_sim_seconds=self.truth_state['stable_full_rate_sim_seconds'])
        session.release(token,fresh_map_safe=True,explicit_release=True)
        publish_authorization()
        self.transition('SKILL_RETURN_CHECK','Aufgerichtet; native Planerübernahme prüfen','transition hold')
        requested=time.monotonic();self.client.resume_planner(math.radians(self.state['yaw_deg']))
        while not self._skill_native('planner_motion'):
            self.tick()
            if time.monotonic()-requested>12:raise RuntimeError('Native planner did not acknowledge skill rejoin')
        self.client.acknowledge_started()
        with self.client._lock:self.client._validate_pose=prior_validator
        self.skill_active=False;self.skill_mapping_enabled=False
        self.settle()
        exit_geometry=observed_exit_geometry(self.truth_state,gate,self.skill_collision_boxes)
        if not exit_geometry['accepted']:
            raise RuntimeError('Native planner settled outside configured passage exit; gate not completed')
        evidence=dict(id=gate['id'],gate_id=gate['id'],observed_exit=True,
            physical_exit_base=self.truth_state['base'][:7],estimated_exit_base=self.state['base'][:7],
            measured_ground_displacement_m=session.ground_displacement,wall_seconds=time.monotonic()-started,
            exit_geometry=exit_geometry,
            completion_source='measured ground progress + standing exit + fresh map + native planner acknowledgement',
            scene_reset=False,operator_intervention=False)
        self.explorer.finish_skill(self.state['base'][:2],max(self.state['sim_time'],self.last_sensor['sim_time']),evidence)
        self.command_stabilizer.reset();self.scan_yaw=None
        self.manifest.setdefault('completed_mission_skills',[]).append(evidence)
        self.event('mission_skill_completed',**evidence)
        self.skill_session=None;self.skill_gate=None

    @staticmethod
    def _skill_atomic(path,value):
        temporary=path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(value,allow_nan=False)+'\n');temporary.replace(path)
