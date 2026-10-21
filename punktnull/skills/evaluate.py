"""Post-run evidence only. Does not provide ground truth to a controller."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np


def _evaluate_v1(run):
    run=Path(run);manifest=json.loads((run/'manifest.json').read_text())
    report={'schema':1,'kind':'actual native SONIC protocol1 + Isaac/PhysX skill trial',
            'live_human_tested':False,'physical_robot_tested':False,'accepted':False,
            'outcome':manifest['outcome'],'checks':{},'limitations':[
                'Isolated disclosed flat floor, not a low passage or complete exploration mission.',
                'Online reference phase is inferred in a bounded neighborhood of submitted cursor; native absolute reference frame is not logged.',
                'Tracking statistics use that retained per-physics-step phase inference, not the old keyboard exact-playback metric.',
                'No contact-force, real-hardware, external-user or general reliability claim.']}
    paths=[run/'manifest.json',run/'sim/metadata.json',run/'sim/trace.npz',run/'submitted_reference.jsonl']
    missing=[str(p) for p in paths if not p.exists()]
    if missing:report['missing_evidence']=missing;return report
    meta=json.loads(paths[1].read_text());z=np.load(paths[2],allow_pickle=False);a=z['data'];columns=list(z['columns'])
    idx=lambda k:columns.index(k)
    get=lambda k:a[:,idx(k)]
    submitted=[json.loads(line) for line in paths[3].read_text().splitlines()]
    checks=report['checks'];events=manifest['events'];getevent=lambda k:next((e for e in events if e['event']==k),None)
    begin=getevent('skill_started');ready=getevent('measured_skill_return_ready');resumed=getevent('native_planner_rejoined')
    checks['runtime_completed']=manifest['outcome']=='STREAMED_CRAWL_RETURN_EXECUTED_PENDING_TRACE_EVALUATION' and manifest.get('session_phase')=='COMPLETE'
    checks['ordered_observed_transitions']=bool(begin and ready and resumed and begin['monotonic']<ready['monotonic']<resumed['monotonic'])
    checks['finite_trace']=bool(np.isfinite(a).all())
    checks['physics_steps_contiguous']=bool(np.all(np.diff(get('physics_step_index'))==1)) if 'physics_step_index' in columns else False
    checks['no_simulator_error']=not meta.get('error') and meta.get('end_reason') not in ('fall','command_watchdog','supervisor_watchdog')
    checks['no_cleanup_error']=not manifest.get('cleanup_errors')
    support=get('support');checks['single_support_release_no_reattach']=bool(np.count_nonzero(np.diff(support)==-1)==1 and not np.any(np.diff(support)>0))
    active=(get('wall_monotonic')>=begin['monotonic']) if begin else np.zeros(len(a),bool)
    if ready:active&=get('wall_monotonic')<=ready['monotonic']
    checks['unsupported_motion']=bool(active.any() and not np.any(support[active]))
    timing=[e for e in meta.get('events',[]) if e.get('name')=='timing' and begin and ready
            and begin['monotonic']<=e['monotonic']<=ready['monotonic']]
    ratios=[e['physics_to_wall_ratio'] for e in timing]
    checks['controller_physics_realtime_band']=bool(ratios and all(.95<=r<=1.05 for r in ratios))
    report['timing']={'source':'retained measured5s physics/wall windows during skill',
        'required_band':[.95,1.05],'ratios':ratios}
    frames=np.array([p['source_frame'] for p in submitted],dtype=int)
    checks['all_reference_frames_submitted_once_in_order']=bool(len(frames) and frames[0] in (0,1) and frames[-1]==manifest['reference']['frames']-1 and np.all((np.diff(frames)>=0)&(np.diff(frames)<=1)))
    checks['no_stream_restart']=bool(len(submitted) and np.all(np.diff([p['wire_index'] for p in submitted])==1))
    ground=active & (get('guard_ground_permission')>.5)
    metric_names=['guard_height_error_m','guard_orientation_error_deg','guard_joint_rms_error_rad']
    p95=[float(np.percentile(get(k)[ground],95)) for k in metric_names] if ground.any() else None
    checks['bounded_inferred_phase_tracking']=bool(p95 is not None and np.all(np.asarray(p95)<=np.array([.12,25.,.35])))
    xyz=a[:,[idx('root_'+str(i)) for i in range(3)]]
    report['measured']={'trace_samples':len(a),'ground_tracking_samples':int(ground.sum()),
                        'ground_tracking_p95_height_m_orientation_deg_joint_rms_rad':p95,
                        'ground_displacement_m':ready.get('ground_displacement_m') if ready else None,
                        'minimum_root_height_m':float(xyz[active,2].min()) if active.any() else None,
                        'skill_wall_seconds':ready['monotonic']-begin['monotonic'] if begin and ready else None,
                        'return_observed':bool(resumed)}
    tables={}
    for key in ['motion_name','encoder_mode','motion_playing']:
        path=run/'controller'/(key+'.csv');paths.append(path)
        if not path.exists():continue
        tables[key]={int(r['index']):r for r in csv.DictReader(path.open())}
    joined=set.intersection(*(set(t) for t in tables.values())) if len(tables)==3 else set()
    rows=[]
    if begin and ready:
        for i in sorted(joined):
            stamp=float(tables['motion_name'][i]['time_monotonic_ms'])/1000
            if begin['monotonic']<=stamp<=ready['monotonic']:
                rows.append((tables['motion_name'][i]['motion_name'],int(float(tables['encoder_mode'][i]['encoder_mode_0'])),float(tables['motion_playing'][i]['playing_0'])>.5))
    streamed=[r for r in rows if r[0]=='streamed']
    checks['actual_native_joint_stream']=bool(len(streamed)>100 and all(r[1]==0 and r[2] for r in streamed))
    checks['no_smpl_substitution']=bool(not any(r[1]==2 for r in rows))
    report['measured']['native_streamed_rows']=len(streamed)
    if manifest.get('terrain')=='passage':
        from punktnull.terrain.kinematics import CollisionFK
        from punktnull.terrain.service_passage import collision_boxes,FIXTURE_CONTRACT
        from punktnull.retained import URDF,prepare
        prepare()
        from bridge import BODY_JOINT_NAMES
        fixture=meta.get('scene',{}).get('fixture_contract')
        scene=meta.get('scene',{})
        # Renderer/motor-loop metadata, not just a requested CLI flag, must
        # prove that the configured collision scene was actually instantiated.
        checks['passage_scene_instantiated']=bool(fixture==FIXTURE_CONTRACT and scene.get('course_enabled')
                                                  and scene.get('collision_boxes')==collision_boxes())
        fk=CollisionFK(URDF,BODY_JOINT_NAMES)
        bounds=fk.bounds(fk.links(xyz,a[:,[idx('root_'+str(i)) for i in range(3,7)]],
                                   a[:,[idx('q_'+str(i)) for i in range(29)]]))
        overlaps=0;minimum=float('inf')
        for box in collision_boxes():
            low=np.array(box['aabb_min']);high=np.array(box['aabb_max'])
            gap=np.maximum(np.maximum(bounds[:,:,0]-high,low-bounds[:,:,1]),0.)
            minimum=min(minimum,float(np.linalg.norm(gap,axis=-1).min()))
            overlaps+=int(np.all(np.minimum(bounds[:,:,1],high)-np.maximum(bounds[:,:,0],low)>1e-9,axis=-1).sum())
        x0,x1=FIXTURE_CONTRACT['roof_x_interval_m']
        crossing=(bounds[:,:,0,0]<=x1)&(bounds[:,:,1,0]>=x0)
        below=bool(crossing.any() and bounds[:,:,1,2][crossing].max()<FIXTURE_CONTRACT['roof_underside_z_m'])
        entered=bool(active.any() and np.min(xyz[active,0])<x0-.5)
        exited=bool(ready and np.min(bounds[get('wall_monotonic')>=ready['monotonic'],:,0,0])>x1)
        checks.update(physical_passage_entered_and_exited=entered and exited,
                      measured_body_stayed_below_roof=below,
                      no_sampled_conservative_body_obstacle_overlap=overlaps==0)
        report['measured']['passage']={'sampled_aabb_overlaps':overlaps,'minimum_sampled_aabb_distance_m':minimum,
            'roof_crossing_max_body_z_m':float(bounds[:,:,1,2][crossing].max()) if crossing.any() else None,
            'geometry_samples':len(a),'fixture':FIXTURE_CONTRACT['id']}
        report['limitations'][0]='Isolated collision-active configured passage; no integrated building mission or sensor-selected skill.'
        report['limitations'].append('Conservative body bounds at each physics sample; no force-sensor or between-sample collision proof.')
    report['source_sha256']={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    report['accepted']=bool(checks and all(checks.values()))
    return report


def evaluate(run, *, contract_version=2):
    """Version2 is prospective user-approved actual-state acceptance.

    Re-reading an old trace is explicitly retrospective, not a new physical run.
    Call contract_version=1 only for historical-method reproduction. Existing
    evidence files are never changed by this function.
    """
    if contract_version not in (1,2):raise ValueError('Unsupported skill contract version')
    if contract_version==1:return _evaluate_v1(run)
    from .limits import (load_joint_limits,acceptance_contract,POSITION_TOLERANCE_RAD,
                         SPEED_TOLERANCE_RAD_S,EFFORT_TOLERANCE_NM)
    run=Path(run)
    try:report=_evaluate_v1(run)
    except Exception as exc:
        return dict(schema=2,accepted=False,acceptance_contract=acceptance_contract(),
                    checks={'complete_parseable_evidence':False},errors=[repr(exc)])
    report['schema']=2;report['acceptance_contract']=acceptance_contract()
    report['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['helper_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('limits.py','phase.py')}
    checks=report['checks']
    legacy_tracking=checks.pop('bounded_inferred_phase_tracking',None)
    report['tracking_diagnostic']={'legacy_p95_tracking_check_would_pass':legacy_tracking,
        'acceptance_effect':'none; height and joint-reference errors are not actual joint-limit violations',
        'p95_height_m_orientation_deg_joint_rms_rad':report.get('measured',{}).get('ground_tracking_p95_height_m_orientation_deg_joint_rms_rad')}
    if report.get('missing_evidence'):return report
    try:
        limits=load_joint_limits();manifest=json.loads((run/'manifest.json').read_text())
        meta=json.loads((run/'sim/metadata.json').read_text())
        report['retrospective_new_contract']=manifest.get('acceptance',{}).get('id')!=report['acceptance_contract']['id']
        report['limitations'] += [
            'Actual-limit v2 was authorized after the earlier runs; retrospective re-evaluation is not a new execution or a rewrite of original acceptance.',
            'Effort is the clipped command logged by the explicit motor loop, not measured contact force or an independent motor-torque sensor.',
            'URDF positions and active configuration speeds/efforts are source-derived; old metadata records effort limits but not all runtime position/velocity limit arrays.',
            'Full-rate state and native mode history are retained. Historical per-tick ownership tokens and clearance authorizations were not retained; their original online enforcement is not reconstructed.']
        with np.load(run/'sim/trace.npz',allow_pickle=False) as z:
            a=z['data'];columns=list(z['columns'])
        col=lambda key:a[:,columns.index(key)]
        get=lambda prefix,n:a[:,[columns.index(prefix+str(i)) for i in range(n)]]
        events=manifest['events'];event=lambda name:next((e for e in events if e['event']==name),None)
        begin=event('skill_started');ready=event('measured_skill_return_ready');resume=event('native_planner_rejoined')
        if not(begin and ready and resume):raise ValueError('Missing measured completion/return evidence')
        wall=col('wall_monotonic');sim=col('sim_time')
        active=(wall>=begin['monotonic'])&(wall<=ready['monotonic'])
        if not active.any():raise ValueError('Empty skill interval')
        q=get('q_',29);dq=get('dq_',29);tau=get('tau_',29);base=get('root_',13)
        lower=np.maximum(limits.lower-q,0.);upper=np.maximum(q-limits.upper,0.);position_error=np.maximum(lower,upper)
        velocity_error=np.maximum(abs(dq)-limits.velocity,0.);effort_error=np.maximum(abs(tau)-limits.effort,0.)
        finite=np.isfinite(np.c_[wall,sim,base,q,dq,tau]).all()
        checks['actual_position_limits_respected']=bool(finite and np.max(position_error[active])<=POSITION_TOLERANCE_RAD)
        checks['actual_velocity_limits_respected']=bool(finite and np.max(velocity_error[active])<=SPEED_TOLERANCE_RAD_S)
        checks['actual_applied_effort_limits_respected']=bool(finite and np.max(effort_error[active])<=EFFORT_TOLERANCE_NM)
        checks['recorded_model_efforts_match_contract']=bool(meta.get('joint_names')==list(limits.names) and
            np.array_equal(np.asarray(meta.get('effort_limits_nm',[])),limits.effort))
        checks['monotonic_full_rate_state']=bool(finite and np.all(np.diff(wall)>0) and np.all(np.diff(sim)>0) and
            np.max(np.diff(wall[active]))<=.15 and np.max(abs(np.diff(sim)-.005))<1e-7)
        checks['valid_measured_quaternions']=bool(np.max(abs(np.linalg.norm(base[:,3:7],axis=1)-1))<=.02)
        tilt=col('tilt_deg')
        permission=col('guard_ground_permission')>.5
        checks['ground_only_in_authorized_reference_phase']=bool(np.all(permission[active] | ((base[active,2]>=.25)&(tilt[active]<=65))))
        checks['absolute_root_height_respected']=bool(np.min(base[active,2])>=.025)
        ground=active&permission&(base[:,2]<.45)&(tilt>45)
        travelled=float(np.max(np.linalg.norm(base[ground,:2]-base[ground,:2][0],axis=1))) if ground.any() else 0.
        checks['actual_ground_progress']=bool(ground.sum()>=2 and travelled>=.50)
        # Independent full-rate recovery: no reference-height/joint-match gate.
        end=int(np.flatnonzero(wall<=ready['monotonic'])[-1])
        stable_window=(wall<=wall[end])&(sim>=sim[end]-.70-1e-8)
        stable=(base[:,2]>=.60)&(tilt<=25)&(np.linalg.norm(base[:,7:10],axis=1)<=.10)&(np.sqrt(np.mean(dq*dq,axis=1))<=.20)&(col('support')==0)
        checks['measured_stable_upright_return_window']=bool(stable_window.sum()>=140 and stable[stable_window].all())
        submitted=[json.loads(line) for line in (run/'submitted_reference.jsonl').read_text().splitlines()]
        stamps=np.array([r['monotonic'] for r in submitted]);expected=manifest['reference']['ordered_channel_digest']
        checks['fresh_identified_reference_stream']=bool(len(stamps)>2 and np.isfinite(stamps).all() and np.all(np.diff(stamps)>0)
            and np.max(np.diff(stamps))<=.15 and all(r.get('source_digest')==expected for r in submitted))
        # Native selected source after acknowledgement, then actual planner return.
        nested=ready.get('session_events',[])
        ack=next((e['monotonic'] for e in nested if e.get('event')=='native_reference_stream_acknowledged'),None)
        tables={key:{int(r['index']):r for r in csv.DictReader((run/'controller'/(key+'.csv')).open())}
                for key in ['motion_name','encoder_mode','motion_playing']}
        joined=sorted(set.intersection(*(set(t) for t in tables.values())))
        rows=[]
        for i in joined:
            rr={k:t[i] for k,t in tables.items()};times=[float(v['time_monotonic_ms'])/1000 for v in rr.values()]
            if max(times)-min(times)>1e-5:raise ValueError('Native feedback clocks disagree')
            rows.append((i,times[0],rr['motion_name']['motion_name'],float(rr['encoder_mode']['encoder_mode_0']),float(rr['motion_playing']['playing_0'])))
        owned=[r for r in rows if ack is not None and ack<=r[1]<=ready['monotonic']]
        returned=[r for r in rows if ready['monotonic']<=r[1]<=resume['monotonic']+.03 and r[2]=='planner_motion' and r[3]==0 and r[4]>.5]
        checks['exclusive_fresh_native_skill_source']=bool(len(owned)>100 and all(r[2]=='streamed' and r[3]==0 and r[4]>.5 for r in owned)
            and all(b[0]==a[0]+1 and 0<b[1]-a[1]<=.15 for a,b in zip(owned,owned[1:]))
            and owned[0][1]-ack<=.15 and ready['monotonic']-owned[-1][1]<=.15)
        checks['actual_native_planner_rejoined']=bool(returned)
        violations=[]
        for i,name in enumerate(limits.names):
            bad=active&(position_error[:,i]>POSITION_TOLERANCE_RAD)
            if not bad.any():continue
            index=int(np.argmax(np.where(active,position_error[:,i],-1)))
            runs=[];indices=np.flatnonzero(bad)
            for segment in np.split(indices,np.where(np.diff(indices)>1)[0]+1):
                runs.append(dict(first_wall=float(wall[segment[0]]),last_wall=float(wall[segment[-1]]),
                    first_sim=float(sim[segment[0]]),last_sim=float(sim[segment[-1]]),samples=len(segment),
                    sampled_duration_sim_s=float(sim[segment[-1]]-sim[segment[0]]+.005),
                    inferred_reference_frames=[int(col('guard_reference_frame')[segment].min()),int(col('guard_reference_frame')[segment].max())]))
            violations.append(dict(joint=name,lower=float(limits.lower[i]),upper=float(limits.upper[i]),
                minimum_actual=float(q[active,i].min()),maximum_actual=float(q[active,i].max()),
                maximum_excess_rad=float(position_error[index,i]),samples=int(bad.sum()),percentage_of_active_samples=float(bad.sum()/active.sum()*100),
                total_sampled_duration_sim_s=float(bad.sum()*.005),at_maximum=dict(wall=float(wall[index]),sim=float(sim[index]),actual_q=float(q[index,i]),
                    inferred_reference_frame=int(col('guard_reference_frame')[index])),segments=runs))
        report['actual_limits_observation']=dict(active_samples=int(active.sum()),joint_position_violations=violations,
            maximum_position_excess_rad=float(position_error[active].max()),maximum_velocity_excess_rad_s=float(velocity_error[active].max()),
            maximum_applied_effort_excess_nm=float(effort_error[active].max()),
            maximum_abs_joint_speed_rad_s=np.max(abs(dq[active]),axis=0).tolist(),maximum_abs_applied_effort_nm=np.max(abs(tau[active]),axis=0).tolist(),
            measured_ground_progress_m=travelled,stable_return_samples=int(stable_window.sum()),stable_return_fraction=float(stable[stable_window].mean()),
            native_owned_rows=len(owned),native_planner_return_rows=len(returned))
    except Exception as exc:
        checks['complete_actual_limits_evidence']=False;report.setdefault('errors',[]).append(repr(exc))
    report['failed_checks']=[k for k,v in checks.items() if not v]
    report['accepted']=bool(checks and all(checks.values()) and not report.get('errors'))
    report['acceptance_scope']='Version2 actual configured limits + measured isolated skill execution; no original-report changes, integrated mission or real-hardware claim'
    return report
