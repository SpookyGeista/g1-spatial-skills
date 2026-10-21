"""Independent saved-evidence audit for both integrated crawl gates.

No controller input, simulator import, ground-truth correction or file mutation.
This can replace ONLY an upright-only physics predicate after accepted=True.
It cannot replace mission, LIO, perception, timing, video or human acceptance.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
from .skills.limits import load_joint_limits,acceptance_contract,POSITION_TOLERANCE_RAD,SPEED_TOLERANCE_RAD_S,EFFORT_TOLERANCE_NM
from .terrain.kinematics import CollisionFK
from .retained import URDF


def _jsonl(path):
    lines=Path(path).read_text().splitlines()
    if not lines:return []
    return [json.loads(line) for line in lines]


def native_rows(folder):
    """Require all three complete, synchronized streams; no intersection hiding."""
    tables={}
    for name in ('motion_name','encoder_mode','motion_playing'):
        with (Path(folder)/(name+'.csv')).open() as stream:rows=list(csv.DictReader(stream))
        table={int(r['index']):r for r in rows}
        if not table or len(rows)!=len(table):raise ValueError('Missing/duplicate native '+name+' rows')
        tables[name]=table
    keys=set(tables['motion_name'])
    if any(set(t)!=keys for t in tables.values()):raise ValueError('Incomplete native synchronized indices')
    result=[]
    for index in sorted(keys):
        values={k:t[index] for k,t in tables.items()}
        stamps=[float(v['time_monotonic_ms'])/1000 for v in values.values()]
        mode=float(values['encoder_mode']['encoder_mode_0']);playing=float(values['motion_playing']['playing_0'])
        if not np.isfinite([*stamps,mode,playing]).all() or max(stamps)-min(stamps)>1e-5:
            raise ValueError('Native nonfinite/mismatched timestamps')
        if mode not in (0,1,2) or playing not in (0,1):raise ValueError('Invalid native flags')
        result.append((index,stamps[0],values['motion_name']['motion_name'],int(mode),bool(playing)))
    if any(b[0]!=a[0]+1 or not 0<b[1]-a[1]<=.15 for a,b in zip(result,result[1:])):
        raise ValueError('Native index/clock/freshness gap')
    return result


def audit_authorization(rows,source,digest,begin,finish):
    """An explicit owner release is necessary; absence/staleness never releases."""
    if len(rows)<3:raise ValueError('Missing skill authorization sequence')
    stamps=np.array([r['monotonic'] for r in rows],float)
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0) or np.max(np.diff(stamps))>.15:
        raise ValueError('Authorization clock/freshness gap')
    if not -.15<=stamps[0]-begin<=.15 or not begin<stamps[-1]<=finish:
        raise ValueError('Authorization does not cover declared skill')
    if any(r.get('source')!=source or r.get('source_digest')!=digest for r in rows):
        raise ValueError('Authorization source changed')
    releases=[i for i,r in enumerate(rows) if r.get('owner')=='none' and r.get('phase')=='COMPLETE']
    if releases!=[len(rows)-1] or rows[-1].get('owner_token') is not None:
        raise ValueError('No unique final explicit COMPLETE release')
    owned=rows[:-1];tokens={r.get('owner_token') for r in owned}
    if len(tokens)!=1 or not isinstance(next(iter(tokens)),str) or not next(iter(tokens)):
        raise ValueError('Owner changed or missing token')
    if rows[-1].get('released_owner_token')!=next(iter(tokens)) or not rows[-1].get('clearance_ok'):
        raise ValueError('Final release is not bound to validated owner/clearance')
    if any(r.get('owner')!='skill' or not r.get('clearance_ok') for r in owned):
        raise ValueError('Unowned/unverified clearance interval')
    phases=[r.get('phase') for r in owned]
    allowed={'PRIMING','ALIGNING','ENTERING','GROUND','EXITING','SETTLING','RETURN_READY'}
    if phases[0]!='PRIMING' or phases[-1]!='RETURN_READY' or not {'ALIGNING','GROUND','RETURN_READY'}<=set(phases) or not set(phases)<=allowed:
        raise ValueError('Incomplete measured skill phase sequence')
    return dict(token=next(iter(tokens)),release=float(stamps[-1]),samples=len(rows),maximum_age_s=float(np.max(np.diff(stamps))))


def audit_submission(rows,digest,frames,begin,release):
    if len(rows)<3:raise ValueError('Missing submitted stream evidence')
    stamps=np.array([r['monotonic'] for r in rows],float)
    source=np.array([r['source_frame'] for r in rows]);wire=np.array([r['wire_index'] for r in rows])
    if (not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0) or np.max(np.diff(stamps))>.15
        or not -.02<=stamps[0]-begin<=.15 or not -.02<=release-stamps[-1]<=.15):
        raise ValueError('Submitted stream fails full owned-interval freshness')
    if source.dtype.kind not in 'iu' or wire.dtype.kind not in 'iu':raise ValueError('Invalid stream indices')
    if source[0]!=0 or source[-1]!=frames-1 or np.any(np.diff(source)<0) or np.any(np.diff(source)>1) or np.any(np.diff(wire)!=1):
        raise ValueError('Reference/wire cursor restarted, skipped or incomplete')
    if any(r.get('source_digest')!=digest for r in rows):raise ValueError('Submitted source changed')
    return dict(samples=len(rows),maximum_age_s=float(np.max(np.diff(stamps))),last_frame=int(source[-1]))


def passage_geometry(bounds,xyz,boxes,roof_y,roof_z,direction):
    """Conservative full-link AABBs at actual200Hz samples, never pelvis alone."""
    bounds=np.asarray(bounds,float);xyz=np.asarray(xyz,float)
    if bounds.ndim!=4 or bounds.shape[2:]!=(2,3) or len(bounds)!=len(xyz) or not len(bounds) or not np.isfinite(bounds).all():
        raise ValueError('Invalid actual geometry bounds')
    overlaps=[];minimum=math.inf
    for box in boxes:
        low=np.asarray(box['aabb_min'],float);high=np.asarray(box['aabb_max'],float)
        if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite(np.r_[low,high]).all() or np.any(high<=low):
            raise ValueError('Invalid instantiated collision geometry')
        gap=np.maximum(np.maximum(bounds[:,:,0]-high,low-bounds[:,:,1]),0)
        minimum=min(minimum,float(np.linalg.norm(gap,axis=-1).min()))
        hit=np.all(np.minimum(bounds[:,:,1],high)-np.maximum(bounds[:,:,0],low)>1e-9,axis=-1)
        if hit.any():overlaps.append(dict(prim=box['prim'],sample_body_overlaps=int(hit.sum()),first_sample=int(np.flatnonzero(hit.any(axis=1))[0])))
    y0,y1=map(float,roof_y)
    cross=(bounds[:,:,0,1]<=y1)&(bounds[:,:,1,1]>=y0)
    ceiling=float(bounds[:,:,1,2][cross].max()) if cross.any() else None
    if direction=='ingress':whole_enter=float(bounds[0,:,1,1].max())<y0;whole_exit=float(bounds[-1,:,0,1].min())>y1
    elif direction=='egress':whole_enter=float(bounds[0,:,0,1].min())>y1;whole_exit=float(bounds[-1,:,1,1].max())<y0
    else:raise ValueError('Invalid passage direction')
    ground=(xyz[:,2]<.45)
    displacement=float(np.linalg.norm(xyz[ground,:2]-xyz[np.flatnonzero(ground)[0],:2],axis=1).max()) if ground.any() else 0.
    return dict(sampled_aabb_overlaps=overlaps,minimum_sampled_nonfloor_aabb_distance_m=minimum,
        roof_crossing_body_samples=int(cross.sum()),roof_crossing_max_body_z_m=ceiling,
        every_crossing_body_below_roof=bool(ceiling is not None and ceiling<roof_z),
        whole_body_entered_and_exited=bool(whole_enter and whole_exit),measured_low_root_displacement_m=displacement)


def _inside(xy,region):
    return bool(region['xmin']<=xy[0]<=region['xmax'] and region['ymin']<=xy[1]<=region['ymax'])


def evaluate(run):
    run=Path(run);report=dict(schema=1,kind='integrated two-pass actual-limits crawl evidence',accepted=False,checks={},gates=[],
        acceptance_scope='Only replacement for upright-only physical predicate; all mission/LIO/sensor/timing/video gates remain required',
        live_human_tested=False,physical_robot_tested=False,contract=acceptance_contract(),limitations=[
            'Known fixed fixture and simulator state are disclosed skill admission/evaluation inputs; no general terrain recognition.',
            'Reference height/joint residuals are diagnostic; actual configured position/speed/applied-effort limits remain strict.',
            'Conservative collision AABBs sampled at200Hz; no contact-force or continuous swept collision proof.',
            'Applied/clipped effort command is not an independent measured torque sensor.'])
    checks=report['checks'];paths=[run/'manifest.json',run/'events.jsonl',run/'sim/metadata.json',run/'sim/trace.npz',run/'skill_geometry_preflight.json']
    try:
        manifest=json.loads(paths[0].read_text());events=_jsonl(paths[1]);meta=json.loads(paths[2].read_text());proof=json.loads(paths[4].read_text())
        contract=manifest['skill_fixture_contract'];scene=meta['scene'];actual=scene['inspection_passage'];reference=manifest['crawl_reference']
        boxes=scene['collision_boxes'];source=reference['source'];digest=reference['ordered_channel_digest'];frames=reference['frames']
        checks['declared_two_pass_crawl_contract']=bool(contract['two_passes_required'] and len({contract[x]['id'] for x in ('ingress','egress')})==2)
        checks['scene_and_reference_preflight_bound']=bool(proof.get('accepted') and proof.get('reference_source')==source
            and proof.get('reference_source_digest')==digest and proof.get('contract')==contract
            and proof.get('scene_source_sha256')==scene.get('scene_source_sha256')
            and scene.get('fixture_id')==contract['fixture_id'] and actual.get('mission_contract')==contract
            and proof.get('minimum_clearance_m',0)>=.08 and proof.get('required_margin_m')==.08)
        checks['reference_snapshot_intact']=all(hashlib.sha256((run/'source_snapshot/crawl_reference'/name).read_bytes()).hexdigest()==sha for name,sha in reference['source_sha256'].items())
        saved_sources=manifest['extension_source_sha256']
        checks['geometry_method_source_bound']=all(saved_sources.get(name)==hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest()
            and saved_sources[name]==hashlib.sha256((run/'source_snapshot/extension'/name).read_bytes()).hexdigest()
            for name in ('punktnull/terrain/kinematics.py','punktnull/skills/limits.py'))
        with np.load(paths[3],allow_pickle=False) as z:data=z['data'];cols=list(z['columns'])
        if data.ndim!=2 or data.shape[1]!=len(cols) or len(cols)!=len(set(cols)) or not len(data) or not np.isfinite(data).all():
            raise ValueError('Missing/nonfinite physical trace')
        get=lambda name:data[:,cols.index(name)]
        wall=get('wall_monotonic');sim=get('sim_time');support=get('support');step=get('physics_step_index')
        root=data[:,[cols.index('root_'+str(i)) for i in range(13)]];q=data[:,[cols.index('q_'+str(i)) for i in range(29)]]
        dq=data[:,[cols.index('dq_'+str(i)) for i in range(29)]];tau=data[:,[cols.index('tau_'+str(i)) for i in range(29)]]
        tilt=get('tilt_deg');limits=load_joint_limits();unsupported=support==0
        checks['continuous_full_rate_state']=bool(np.all(np.diff(step)==1) and np.all(np.diff(wall)>0)
            and np.max(np.diff(wall)[unsupported[1:]&unsupported[:-1]])<=.15 and np.max(abs(np.diff(sim)-.005))<1e-6)
        checks['single_support_release_no_reattach']=bool(np.count_nonzero(np.diff(support)==-1)==1 and np.all(np.diff(support)<=0))
        checks['valid_actual_orientation']=bool(np.max(abs(np.linalg.norm(root[:,3:7],axis=1)-1))<=.001)
        position=np.maximum(limits.lower-q,q-limits.upper);velocity=abs(dq)-limits.velocity;effort=abs(tau)-limits.effort
        checks['recorded_actual_model_limits_match']=bool(meta.get('joint_names')==list(limits.names) and np.array_equal(np.asarray(meta.get('effort_limits_nm',[])),limits.effort))
        checks['actual_position_limits_all_unsupported']=bool(unsupported.any() and np.max(position[unsupported])<=POSITION_TOLERANCE_RAD)
        checks['actual_speed_limits_all_unsupported']=bool(unsupported.any() and np.max(velocity[unsupported])<=SPEED_TOLERANCE_RAD_S)
        checks['actual_effort_limits_all_unsupported']=bool(unsupported.any() and np.max(effort[unsupported])<=EFFORT_TOLERANCE_NM)
        checks['absolute_root_height']=bool(np.min(root[unsupported,2])>=.025)
        report['actual_limits']=dict(unsupported_samples=int(unsupported.sum()),max_position_excess_rad=float(np.max(position[unsupported])),
            max_speed_excess_rad_s=float(np.max(velocity[unsupported])),max_effort_excess_nm=float(np.max(effort[unsupported])))
        natives=native_rows(run/'controller');paths.extend(run/'controller'/(k+'.csv') for k in ('motion_name','encoder_mode','motion_playing'))
        fk=CollisionFK(URDF,limits.names);inside_skill=np.zeros(len(data),bool);tokens=[];intervals=[]
        expected_ids={contract[x]['id'] for x in ('ingress','egress')}
        starts=[e for e in events if e.get('event')=='mission_skill_started'];ends=[e for e in events if e.get('event')=='mission_skill_completed']
        checks['exactly_two_actual_completed_gates']=len(starts)==len(ends)==2 and {e.get('gate_id') for e in starts}==expected_ids and {e.get('gate_id') for e in ends}==expected_ids
        for direction in ('ingress','egress'):
            gate=contract[direction];gid=gate['id'];gc={};row=dict(gate_id=gid,direction=direction,checks=gc);report['gates'].append(row)
            def event(name):
                match=[e for e in events if e.get('event')==name and e.get('gate_id')==gid]
                if len(match)!=1:raise ValueError('Missing/duplicate '+name+' for '+gid)
                return match[0]
            begin=event('mission_skill_started');standing=event('mission_skill_standing_exit');end=event('mission_skill_completed')
            b=float(begin['monotonic']);s=float(standing['monotonic']);e=float(end['monotonic']);intervals.append((b,e));gc['ordered_measured_transitions']=b<s<e
            authpath=run/('skill_authorization_'+gid+'.jsonl');streampath=run/('submitted_reference_'+gid+'.jsonl');paths.extend([authpath,streampath])
            authorization=audit_authorization(_jsonl(authpath),source,digest,b,e);tokens.append(authorization['token']);release=authorization['release']
            row['authorization']=authorization;gc['explicit_exclusive_owned_authorization']=True
            row['submitted_stream']=audit_submission(_jsonl(streampath),digest,frames,b,release);gc['complete_fresh_reference_submission']=True
            mask=(wall>=b)&(wall<=e);inside_skill|=mask
            if mask.sum()<141:raise ValueError('Truncated measured skill interval')
            begin_index=int(np.searchsorted(wall,b));end_index=int(np.searchsorted(wall,e,side='right')-1)
            gc['trace_covers_observed_events']=bool(abs(wall[begin_index]-b)<=.15 and abs(e-wall[end_index])<=.15 and not np.any(support[mask]))
            elapsed=sim[mask];gc['strict_state_clock_during_skill']=bool(np.max(np.diff(wall[mask]))<=.15 and np.max(abs(np.diff(elapsed)-.005))<1e-6)
            ground=mask&(root[:,2]<.45)&(tilt>45);ground_i=np.flatnonzero(ground)
            progress=float(np.linalg.norm(root[ground,:2]-root[ground_i[0],:2],axis=1).max()) if len(ground_i) else 0.
            gc['actual_ground_progress']=progress>=.5;row['actual_ground_displacement_m']=progress
            low=mask&((root[:,2]<.25)|(tilt>65));gc['low_posture_only_in_allowed_phase']=bool(np.all(get('guard_ground_permission')[low]>.5))
            for label,t in (('standing',s),('release',release),('planner_return',e)):
                ix=int(np.searchsorted(wall,t,side='right')-1);window=mask&(sim>=sim[ix]-.700001)&(sim<=sim[ix])
                stable=(support==0)&(root[:,2]>=.60)&(tilt<=25)&(np.linalg.norm(root[:,7:10],axis=1)<=.10)&(np.sqrt(np.mean(dq*dq,axis=1))<=.20)
                gc[label+'_stable_full_rate_window']=bool(window.sum()>=140 and np.all(stable[window]))
                gc[label+'_actual_exit_region']=_inside(root[ix,:2],gate['exit_region'])
            nested=standing['session_events'];ack=next(float(x['monotonic']) for x in nested if x.get('event')=='native_reference_stream_acknowledged')
            owned=[r for r in natives if ack<=r[1]<=release];returned=[r for r in natives if release<r[1]<=e]
            gc['actual_native_exclusive_joint_stream']=bool(len(owned)>100 and all(r[2]=='streamed' and r[3]==0 and r[4] for r in owned)
                and owned[0][1]-ack<=.15 and release-owned[-1][1]<=.15)
            settled=[r for r in returned if e-.70<=r[1]<=e]
            gc['actual_native_planner_return']=bool(len(settled)>=25 and all(r[2]=='planner_motion' and r[3]==0 and r[4] for r in settled)
                and e-settled[-1][1]<=.15)
            row['native_owned_rows']=len(owned);row['native_settled_planner_rows']=len(settled)
            bounds=fk.bounds(fk.links(root[mask,:3],root[mask,3:7],q[mask]))
            geometry=passage_geometry(bounds,root[mask,:3],boxes,actual['roof_y_interval_m'],actual['clear_height_m'],direction)
            row['geometry']=geometry
            gc['physical_full_body_passage']=geometry['whole_body_entered_and_exited']
            gc['all_crossing_body_envelopes_below_roof']=geometry['every_crossing_body_below_roof']
            gc['no_sampled_nonfloor_body_overlap']=not geometry['sampled_aabb_overlaps']
            gc['end_event_not_reset_or_human_claim']=bool(end.get('scene_reset') is False and end.get('operator_intervention') is False and end.get('observed_exit') is True)
            row['wall_seconds']=e-b;row['accepted']=all(gc.values())
        outside=unsupported&~inside_skill
        checks['upright_guard_outside_skill_intervals']=bool(np.all(root[outside,2]>=.38) and np.all(tilt[outside]<=50))
        checks['distinct_sequential_owner_sessions']=bool(len(set(tokens))==2 and intervals[0][1]<intervals[1][0])
        checks['both_measured_skill_gates_accepted']=len(report['gates'])==2 and all(g.get('accepted',False) for g in report['gates'])
        checks['no_simulator_or_cleanup_error']=not meta.get('error') and not manifest.get('cleanup_errors') and meta.get('end_reason') not in ('fall','release_refused','nonfinite_state','motor_command_watchdog','supervisor_watchdog')
        report['sources_sha256']={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception as exc:
        checks['complete_integrated_skill_evidence']=False;report.setdefault('errors',[]).append(type(exc).__name__+': '+str(exc))
    report['failed_checks']=[k for k,v in checks.items() if not v]
    report['failed_gate_checks']={g['gate_id']:[k for k,v in g['checks'].items() if not v] for g in report['gates'] if not g.get('accepted',False)}
    report['accepted']=bool(checks and all(checks.values()) and not report.get('errors'))
    return report
