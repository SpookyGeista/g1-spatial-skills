"""Observed-path navigation to explicitly supplied room-inspection objectives."""
import json
import numpy as np
from .retained import ROOT,prepare
prepare()
from navigation import Explorer,OccupancyGrid,NavCommand

class LioOccupancyGrid(OccupancyGrid):
    def begin_stable_episode(self,position_xy,now,*,gate_id,exit_geometry,stable_seconds,support_free):
        """Record a new measured upright footprint after a checked ground skill.

        Keep every old pose/cell. Do not connect the endpoints or change depth
        observations; the existing conservative single-footprint method runs
        with a new continuity anchor only after explicit exit admission.
        """
        episodes=getattr(self,'stable_episodes',[])
        if (not gate_id or any(e['gate_id']==gate_id for e in episodes) or
                not isinstance(exit_geometry,dict) or exit_geometry.get('accepted') is not True or
                exit_geometry.get('inside_exit_region') is not True or
                not np.isfinite(stable_seconds) or stable_seconds<.70 or support_free is not True):
            raise ValueError('New footprint episode requires unique checked stable skill exit')
        margin=exit_geometry.get('minimum_nonfloor_aabb_clearance_m',float('nan'))
        if not np.isfinite(margin) or margin<.08:raise ValueError('Exit geometry margin is not verified')
        prior=self.proprioceptive_poses
        if prior and now<=prior[-1]['time']:raise ValueError('New footprint episode must advance time')
        self.proprioceptive_poses=[]
        try:
            accepted=super().record_stable_pose(position_xy,now,stable=True,support_free=True)
            added=self.proprioceptive_poses
        finally:
            self.proprioceptive_poses=prior+self.proprioceptive_poses
        if not accepted:raise ValueError(self.last_proprioceptive_rejection)
        evidence=dict(gate_id=gate_id,sim_time=float(now),xy_m=list(added[-1]['xy']),
            previous_xy_m=prior[-1]['xy'].copy() if prior else None,
            stable_seconds=float(stable_seconds),exit_geometry=exit_geometry,
            source='FAST-LIO position at measured stable unsupported skill exit; flat-floor footprint assumption',
            interpolated_path=False,depth_cells_changed=False,old_map_reset=False)
        self.stable_episodes=[*episodes,evidence]
        return evidence

    def proprioceptive_evidence(self):
        result=super().proprioceptive_evidence()
        result['separate_stable_episodes']=getattr(self,'stable_episodes',[])
        return result

    def diagnostics(self):
        result=super().diagnostics();result['localization']='FAST-LIO IMU estimate + measured waist FK; one-time initial registration'
        return result

class InspectionExplorer(Explorer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.inspection_plan=json.loads((ROOT/'inspection_plan.json').read_text())
        self.room_index=-1;self.completed_rooms=[];self.room_scan=False;self.inspection_failure=None
        self.inject_review_after_room=None;self.injected_review_resolved=False
        self.room_arrival=None;self.settled_admission=None;self.arrival_retries={}

    def admit_settled_room_scan(self,position_xy,sim_time):
        """Recheck the actual post-stop pose; arrival alone cannot admit a scan."""
        if not self.room_scan:raise RuntimeError('No pending room scan to admit')
        room=self.inspection_plan['rooms'][self.room_index]
        xy=np.asarray(position_xy,dtype=float)
        if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Invalid settled scan position')
        distance=float(np.linalg.norm(xy-np.asarray(room['observation_xy_m'])))
        admitted=distance<=self.inspection_plan['arrival_radius_m']
        record=dict(room=room['id'],arrival=self.room_arrival,settled_xy_m=xy.tolist(),
                    settled_sim_time=float(sim_time),distance_m=distance,
                    radius_m=self.inspection_plan['arrival_radius_m'],admitted=admitted)
        self.settled_admission=record
        if not admitted:
            self.arrival_retries[room['id']]=self.arrival_retries.get(room['id'],0)+1
            record['rejected_attempt']=self.arrival_retries[room['id']]
            self.room_scan=False;self.scan_origin=None;self.room_arrival=None
            self.set_target(room['observation_xy_m'])
            if record['rejected_attempt']>=3:
                self.inspection_failure='Three post-stop room scan admissions outside the configured arrival radius'
        return record

    def finish_scan(self):
        if not self.scan_pending:
            raise RuntimeError('Cannot complete a scan that was not requested')
        advance=self.room_index==-1 or self.room_scan
        if self.room_scan:
            requested=np.array(self.inspection_plan['rooms'][self.room_index]['observation_xy_m'])
            if np.linalg.norm(self.scan_origin-requested)>self.inspection_plan['arrival_radius_m']:
                raise RuntimeError('Room scan occurred outside the requested observation zone')
            self.completed_rooms.append(dict(self.inspection_plan['rooms'][self.room_index],
                observed_scan_origin_m=self.scan_origin.tolist(),scan_number=self.scan_count+1,
                arrival=self.room_arrival,settled_admission=self.settled_admission))
        super().finish_scan();self.room_scan=False
        if not advance:return
        self.room_index+=1
        if self.room_index<len(self.inspection_plan['rooms']):
            self.set_target(self.inspection_plan['rooms'][self.room_index]['observation_xy_m'])
        else:self.mode='inspection_complete';self.target=None;self.path=[]

    def step(self,position_xy,yaw,now,pose_time=None):
        if (self.inject_review_after_room and not self.injected_review_resolved and
                any(r['id']==self.inject_review_after_room for r in self.completed_rooms)):
            self.status='OPERATOR_REVIEW_REQUIRED'
            self.reason='Explicitly injected test review after room scan; takeover and checked explicit release required'
            return NavCommand(float(yaw),status=self.status,reason=self.reason)
        if self.inspection_failure:
            self.status='INSPECTION_SCAN_BLOCKED';self.reason=self.inspection_failure
            return NavCommand(float(yaw),status='INSPECTION_SCAN_BLOCKED',reason=self.inspection_failure)
        if self.mode=='inspection_complete':
            self.status='EXPLORATION_EXHAUSTED';self.reason='All configured room scans physically completed'
            return NavCommand(float(yaw),status='EXPLORATION_EXHAUSTED',reason='All configured room scans physically completed')
        command=super().step(position_xy,yaw,now,pose_time)
        if command.status=='TARGET_REACHED':
            self.room_scan=True;self.scan_pending=True;self.scan_origin=np.asarray(position_xy).copy()
            requested=np.asarray(self.inspection_plan['rooms'][self.room_index]['observation_xy_m'])
            self.room_arrival=dict(xy_m=self.scan_origin.tolist(),sim_time=float(now),
                                   distance_m=float(np.linalg.norm(self.scan_origin-requested)))
            self.settled_admission=None
            self.status='SCAN_REQUIRED';self.reason='Configured room reached; observed full scan required'
            return NavCommand(float(yaw),status='SCAN_REQUIRED',reason='Configured room reached using estimated pose; observed full scan required')
        return command

    def reject_scan(self,measured_xy,now,reason):
        super().reject_scan(measured_xy,now,reason)
        self.inspection_failure=reason
        return True

    def reanchor_from_handover(self,position_xy,now):
        okay=super().reanchor_from_handover(position_xy,now)
        if okay:
            if self.status=='OPERATOR_REVIEW_REQUIRED':self.injected_review_resolved=True
            self.inspection_failure=None
            # The operator may have moved away from a requested room. The
            # retained owner requests a fresh scan after handover; that scan
            # must not count as this room's inspection unless we first arrive.
            self.room_scan=False
            self.room_arrival=None;self.settled_admission=None
            if self.mode!='return' and 0<=self.room_index<len(self.inspection_plan['rooms']):
                self.set_target(self.inspection_plan['rooms'][self.room_index]['observation_xy_m'])
        return okay
