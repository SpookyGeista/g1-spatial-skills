#!/usr/bin/env python3
"""Inspect actual200Hz limits in completed OR incomplete saved skill trials.

Writes a NEW report only. A diagnostic is never a replacement for acceptance.
Current model comparison is refused unless the recorded limits and source
hashes exactly match the independently parsed active simulator configuration.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from punktnull.skills.limits import load_joint_limits,acceptance_contract,POSITION_TOLERANCE_RAD,SPEED_TOLERANCE_RAD_S,EFFORT_TOLERANCE_NM


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []


def table(path):
    if not Path(path).exists():return []
    with Path(path).open() as f:return list(csv.DictReader(f))


def causal(rows,t,key='monotonic'):
    candidates=[x for x in rows if float(x[key])<=t]
    return max(candidates,key=lambda x:float(x[key])) if candidates else None


def verify_contract(manifest,meta,limits):
    recorded=manifest.get('acceptance',{}).get('limits')
    if not isinstance(recorded,dict):raise ValueError('Missing recorded actual-limit provenance; comparison refused')
    current=limits.provenance()
    for key in ('joint_names','position_lower_rad','position_upper_rad','maximum_velocity_rad_s','maximum_applied_effort_nm','numerical_tolerances','source_sha256'):
        if recorded.get(key)!=current[key]:raise ValueError('Recorded/current limit contract mismatch: '+key)
    if meta.get('joint_names')!=list(limits.names) or meta.get('effort_limits_nm')!=limits.effort.tolist():
        raise ValueError('Simulator metadata does not match actual joint order/effort configuration')
    for path,digest in recorded['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Model/bridge provenance source changed: '+path)
    return dict(matched=True,source_sha256=recorded['source_sha256'],numerical_tolerances=recorded['numerical_tolerances'])


def audit_braking(run,manifest,trace,trace_columns):
    """Verify separate braking telemetry without hiding raw-limit diagnostics."""
    run=Path(run);paths=[run/'sim/ankle_braking.npz',run/'sim/ankle_braking.json']
    expected=bool(manifest.get('experimental_ankle_braking',False))
    result=dict(requested=expected,available=all(p.exists() for p in paths),verified=False,checks={},errors=[])
    if not expected and not any(p.exists() for p in paths):return dict(result,verified=None),None
    cache=None;checks=result['checks']
    try:
        if not all(p.exists() for p in paths):raise ValueError('Requested braking telemetry missing')
        if not expected:raise ValueError('Undeclared braking telemetry')
        bm=json.loads(paths[1].read_text());lm=json.loads((run/'sim/ankle_target_limiter.json').read_text())
        parent=run/'sim/ankle_target_limiter.npz';parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
        checks['recorded_hashes_and_parent_chain']=bool(hashlib.sha256(paths[0].read_bytes()).hexdigest()==bm.get('npz_sha256')
            and parent_sha==lm.get('npz_sha256') and bm.get('snapshot_index_source')==dict(file=parent.name,sha256=parent_sha)
            and lm.get('braking_evidence')==bm and lm.get('experimental_braking') is True and bm.get('enabled') is True)
        parameters=dict(lookahead_s=.04,activation_rad=.08,restoring_nm=10.,outward_damping_nm_per_rad_s=4.,additional_one_sided_cap_nm=20.)
        checks['exact_declared_parameters_and_caps']=bm.get('parameters')==parameters and bm.get('physical_effort_caps_nm')==[50.]*4 and bm.get('actual_position_limits_changed') is False
        source_ok=True
        for source,digest in bm.get('source_sha256',{}).items():
            path=Path(source);relative=path.relative_to(ROOT);saved=run/'source_snapshot'/relative
            source_ok&=saved.exists() and hashlib.sha256(saved.read_bytes()).hexdigest()==digest and hashlib.sha256(path.read_bytes()).hexdigest()==digest
        checks['recorded_source_snapshots_match']=bool(source_ok and len(bm.get('source_sha256',{}))>=2)
        if not all(checks.values()):raise ValueError('Braking metadata/hash/source mismatch')
        with np.load(paths[0],allow_pickle=False) as z:bd=z['data'];bc=list(z['columns']);names=list(z['joint_names'])
        with np.load(parent,allow_pickle=False) as z:td=z['data'];tc=list(z['columns']);commands=z['native_command_snapshots'];ct=z['native_command_received_at']
        names_expected=['left_ankle_pitch_joint','left_ankle_roll_joint','right_ankle_pitch_joint','right_ankle_roll_joint']
        if names!=names_expected or bd.ndim!=2 or bd.shape[1]!=len(bc) or not len(bd) or len(set(bc))!=len(bc):raise ValueError('Invalid braking trace schema')
        bg=lambda k:bd[:,bc.index(k)];tg=lambda k:td[:,tc.index(k)];pg=lambda k:trace[:,trace_columns.index(k)]
        times=bg('sim_time');support=bg('support')>.5;active=bg('active_supervisor')>.5;ci=bg('native_snapshot_index').astype(int)
        if not np.isfinite(bd[:,[i for i,n in enumerate(bc) if n!='native_received_at']]).all():raise ValueError('Nonfinite braking trace')
        if not np.all(np.diff(times)>0) or not np.all(np.diff(bg('wall_monotonic'))>0):raise ValueError('Braking clock gap/reset')
        if not np.array_equal(active,~support):raise ValueError('Incorrect supported activation')
        ti=np.searchsorted(tg('sim_time'),times);pi=np.searchsorted(pg('sim_time'),times)
        if np.any(ti>=len(td)) or np.any(pi>=len(trace)) or not np.allclose(tg('sim_time')[ti],times,rtol=0,atol=1e-8) or not np.allclose(pg('sim_time')[pi],times,rtol=0,atol=1e-8):
            raise ValueError('Braking has no same-tick target/physical trace match')
        if np.any(ci<-1) or np.any(ci>=len(commands)) or np.any(ci[active]<0) or not np.array_equal(ci,tg('command_snapshot_index')[ti].astype(int)):
            raise ValueError('Braking native snapshot index mismatch')
        present=ci>=0
        if not np.allclose(bg('native_received_at')[present],ct[ci[present]],rtol=0,atol=1e-9) or not np.isnan(bg('native_received_at')[~present]).all() or np.any(~present&active):
            raise ValueError('Braking command receipt clock mismatch')
        limits=load_joint_limits();joint_rows=[]
        # Exact native-to-target-to-braking-to-final-clip chain for active rows.
        aq=np.column_stack([pg('q_'+str(j))[pi[active]] for j in range(29)])
        av=np.column_stack([pg('dq_'+str(j))[pi[active]] for j in range(29)])
        cmd=commands[ci[active]];chain=np.where(cmd[:,5]!=0,cmd[:,4]+cmd[:,2]*(cmd[:,0]-aq)+cmd[:,3]*(cmd[:,1]-av),0.)
        for k,name in enumerate(lm['joints']):
            j=limits.names.index(name);raw=cmd[:,0,j];mode=cmd[:,5,j]!=0
            lower=limits.lower[j]+lm['desired_interior_margin_rad'];upper=limits.upper[j]-lm['desired_interior_margin_rad']
            if abs(lower-lm['desired_lower_rad'][k])>1e-10 or abs(upper-lm['desired_upper_rad'][k])>1e-10:raise ValueError('Target bounds differ from actual model margin')
            bounded=np.where(mode,np.clip(raw,lower,upper),raw);delta=np.where(mode,cmd[:,2,j]*(bounded-raw),0.)
            if not np.allclose(delta,tg(name+'_effort_correction')[ti[active]],rtol=0,atol=1e-8):raise ValueError('Target correction mismatch before braking')
            chain[:,j]+=delta
        for name in names:
            j=limits.names.index(name);q=bg(name+'_q');v=bg(name+'_dq')
            if not np.array_equal(q,pg('q_'+str(j))[pi]) or not np.array_equal(v,pg('dq_'+str(j))[pi]):raise ValueError('Braking measured q/dq differs from same physics tick')
            lo=q-limits.lower[j];hi=limits.upper[j]-q;vo=np.maximum(0,-v);vu=np.maximum(0,v)
            pl=lo-.04*vo;pu=hi-.04*vu;wl=np.clip((.08-pl)/.08,0,1);wu=np.clip((.08-pu)/.08,0,1)
            rl=wl*(10+4*vo);ru=wu*(10+4*vu);cl=np.minimum(20,rl);cu=np.minimum(20,ru)
            terms=dict(gap_lower=lo,gap_upper=hi,outward_speed_lower=vo,outward_speed_upper=vu,predicted_gap_lower=pl,predicted_gap_upper=pu,
                weight_lower=wl,weight_upper=wu,raw_assist_lower=rl,raw_assist_upper=ru,capped_assist_lower=cl,capped_assist_upper=cu,signed_assist=cl-cu)
            if any(not np.allclose(bg(name+'_'+term),value,rtol=0,atol=1e-9) for term,value in terms.items()):raise ValueError('Braking formula terms mismatch for '+name)
            modes=np.zeros(len(bd));modes[present]=commands[ci[present],5,j]
            if not np.array_equal(bg(name+'_motor_enabled'),modes):raise ValueError('Braking enabled mode mismatch')
            delta=np.where(active&(modes!=0),cl-cu,0.)
            inp=bg(name+'_input_effort');post=inp+delta;final=np.clip(post,-limits.effort[j],limits.effort[j])
            if (not np.allclose(delta,bg(name+'_applied_assist'),rtol=0,atol=1e-9)
                or not np.allclose(inp,tg(name+'_corrected_effort')[ti],rtol=0,atol=1e-8)
                or not np.allclose(post,bg(name+'_output_effort_before_final_clip'),rtol=0,atol=1e-8)
                or not np.allclose(final,bg(name+'_expected_effort_after_existing_clip'),rtol=0,atol=1e-8)
                or not np.allclose(final,pg('tau_'+str(j))[pi],rtol=0,atol=1e-8)):
                raise ValueError('Braking pre/post/final actual effort mismatch for '+name)
            if not np.allclose(chain[:,j],inp[active],rtol=0,atol=1e-8):raise ValueError('Braking input differs from exact native+target PD')
            chain[:,j]+=delta[active]
            joint_rows.append(dict(joint=name,changed_samples=int(np.count_nonzero(delta)),maximum_abs_assist_nm=float(abs(delta).max()),
                final_clip_samples=int(np.count_nonzero(abs(post-final)>1e-9)),same_tick_state_and_effect_verified=True))
        final_chain=np.clip(chain,-limits.effort,limits.effort)
        actual_tau=np.column_stack([pg('tau_'+str(j))[pi[active]] for j in range(29)])
        checks['full29joint_native_target_braking_applied_chain']=bool(np.allclose(final_chain,actual_tau,rtol=0,atol=1e-8))
        checks['same_tick_state_and_all_formula_stages']=True
        result.update(verified=all(checks.values()),samples=len(bd),active_samples=int(active.sum()),joints=joint_rows,
            source_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,parent,run/'sim/ankle_target_limiter.json']})
        if result['verified']:cache=dict(data=bd,columns=bc)
    except Exception as exc:result['errors'].append(type(exc).__name__+': '+str(exc))
    return result,cache


def diagnose(run):
    run=Path(run).resolve();manifest=json.loads((run/'manifest.json').read_text());meta=json.loads((run/'sim/metadata.json').read_text())
    report=dict(schema=1,kind='incomplete-capable saved actual-limit diagnostic',run=str(run),diagnostic_complete=False,
        recorded_outcome_unchanged=manifest.get('outcome'),recorded_error=manifest.get('error'),acceptance_result='NOT_REEVALUATED_BY_THIS_DIAGNOSTIC',
        physical_robot_tested=False,live_human_tested=False,violations=[],errors=[],limitations=[
            'Observed absence of violations before abort is not completion, future containment or reliability.',
            'Trace tau is the clipped applied effort command, not measured contact force or independent motor torque.',
            'Reference cursor is submitted/inferred; actual native absolute reference-frame index is unavailable.',
            'Nearest earlier native log rows are contextual until an exact cached command snapshot is available.'])
    inputs=[run/'manifest.json',run/'sim/metadata.json',run/'sim/trace.npz'];limits=load_joint_limits()
    try:
        report['contract_match']=verify_contract(manifest,meta,limits)
        with np.load(inputs[2],allow_pickle=False) as z:a=z['data'];columns=list(z['columns'])
        if a.ndim!=2 or len(a)<1 or a.shape[1]!=len(columns) or len(columns)!=len(set(columns)) or not np.isfinite(a).all():raise ValueError('Invalid/nonfinite raw physical trace')
        col=lambda k:a[:,columns.index(k)]
        block=lambda k:a[:,[columns.index(k+str(i)) for i in range(29)]]
        wall=col('wall_monotonic');sim=col('sim_time');support=col('support')>.5;q=block('q_');dq=block('dq_');tau=block('tau_')
        if np.any(np.diff(wall)<=0) or np.any(np.diff(sim)<=0) or np.any(np.diff(col('physics_step_index'))!=1):raise ValueError('Physical trace clock/index reset or gap')
        report['trace']=dict(samples=len(a),supported_samples=int(support.sum()),unsupported_samples=int((~support).sum()),
            first_sim_s=float(sim[0]),last_sim_s=float(sim[-1]),maximum_sim_step_error_s=float(np.max(abs(np.diff(sim)-.005))) if len(a)>1 else None,
            maximum_wall_step_s=float(np.max(np.diff(wall))) if len(a)>1 else None,
            minimum_root_height_m=float(col('root_2').min()),maximum_tilt_deg=float(col('tilt_deg').max()))
        events=lines(run/'events.jsonl');submitted=lines(run/'submitted_reference.jsonl')
        inputs.extend(p for p in (run/'events.jsonl',run/'submitted_reference.jsonl') if p.exists())
        native={k:table(run/'controller'/(k+'.csv')) for k in ('action','motion_name','motion_playing','encoder_mode')}
        for key,rows in native.items():
            for row in rows:row['monotonic']=float(row['time_monotonic_ms'])/1000
            path=run/'controller'/(key+'.csv')
            if path.exists():inputs.append(path)
        report['requested_filter']=dict(ankle_roll_margin_rad=manifest.get('experimental_ankle_roll_target_margin_rad',0.),include_pitch=manifest.get('experimental_limit_ankle_pitch',False),source='Recorded trial manifest request; not measured filter telemetry')
        limiter_files=[run/'sim/ankle_target_limiter.npz',run/'sim/ankle_target_limiter.json']
        ldata=lcols=commands=None
        if all(p.exists() for p in limiter_files):
            lm=json.loads(limiter_files[1].read_text())
            if hashlib.sha256(limiter_files[0].read_bytes()).hexdigest()!=lm['npz_sha256']:raise ValueError('Limiter telemetry hash mismatch')
            with np.load(limiter_files[0],allow_pickle=False) as z:
                ldata=z['data'];lcols=list(z['columns']);commands=z['native_command_snapshots'];command_times=z['native_command_received_at']
                if list(z['native_sdk_joint_names'])!=list(limits.names):raise ValueError('Limiter native joint order mismatch')
            report['limiter_telemetry']=dict(available=True,metadata=lm,npz=str(limiter_files[0].relative_to(run)))
            inputs+=limiter_files
        else:
            selected=manifest.get('simulator_command',[])
            report['limiter_telemetry']=dict(available=False,requested_cli=selected,
                missing=[str(p.relative_to(run)) for p in limiter_files if not p.exists()],
                interpretation='No exact per-step filter commands/effects can be reconstructed from missing telemetry. In011 root diagnosed teardown persistence after Kit immediate shutdown; runtime hook fixed prospectively for012, not backfilled.')
        braking,brake_cache=audit_braking(run,manifest,a,columns)
        report['braking_telemetry']=braking
        inputs.extend(p for p in (run/'sim/ankle_braking.npz',run/'sim/ankle_braking.json') if p.exists())
        reference=manifest.get('reference',{});clip_data=None
        source=Path(reference.get('source','/nonexistent'))
        if source.exists() and reference.get('source_sha256') and all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in reference['source_sha256'].items()):
            from punktnull.skills.legacy import ground_module
            clip_data=np.loadtxt(source/'joint_pos.csv',delimiter=',',skiprows=1,ndmin=2)[:,ground_module().MJ_TO_IL]
        def context(index,j):
            t=float(wall[index]);r=dict(sample_index=int(index),wall_monotonic=t,sim_time_s=float(sim[index]),support=bool(support[index]),
                q_rad=float(q[index,j]),dq_rad_s=float(dq[index,j]),applied_effort_nm=float(tau[index,j]),root_height_m=float(col('root_2')[index]),tilt_deg=float(col('tilt_deg')[index]))
            r['prior_manifest_event']=causal(events,t)
            statuses=[v for v in meta.get('events',[]) if v.get('name')=='command_status']
            r['native_command_status_event']=causal(statuses,t)
            logged={k:causal(rows,t) for k,rows in native.items()}
            r['native_logged_state']={k:(dict(index=int(v['index']),age_s=t-v['monotonic'],monotonic=v['monotonic'],
                value=v.get({'motion_name':'motion_name','motion_playing':'playing_0','encoder_mode':'encoder_mode_0','action':'act_'+str(j)}[k])) if v else None) for k,v in logged.items()}
            stream=causal(submitted,t);r['submitted_reference']=stream
            inference=col('guard_reference_frame');previous=np.flatnonzero((np.arange(len(a))<=index)&(inference>=0))
            r['current_guard_inferred_frame']=int(inference[index]);r['last_valid_guard_inference']=None
            if len(previous):
                pi=int(previous[-1]);frame=int(inference[pi]);r['last_valid_guard_inference']=dict(frame=frame,age_s=t-float(wall[pi]),sim_time_s=float(sim[pi]),
                    reference_q_rad=float(clip_data[frame,j]) if clip_data is not None and frame<len(clip_data) else None,
                    basis='Previous measured-neighborhood phase inference, not native execution index')
            if clip_data is not None and stream and 0<=stream['source_frame']<len(clip_data):r['submitted_reference_q_rad']=float(clip_data[stream['source_frame'],j])
            if ldata is not None:
                ii=int(np.argmin(abs(ldata[:,lcols.index('sim_time')]-sim[index])))
                if abs(ldata[ii,lcols.index('sim_time')]-sim[index])<=1e-6:
                    li=int(ldata[ii,lcols.index('command_snapshot_index')])
                    if 0<=li<len(commands):
                        c=commands[li];raw=float(c[4,j]+c[2,j]*(c[0,j]-q[index,j])+c[3,j]*(c[1,j]-dq[index,j])) if c[5,j] else 0.
                        desired=float(c[0,j]);correction=0.;name=limits.names[j]
                        if name+'_bounded_desired_q' in lcols:
                            desired=float(ldata[ii,lcols.index(name+'_bounded_desired_q')]);correction=float(ldata[ii,lcols.index(name+'_effort_correction')])
                        brake_delta=0.;complete=not braking['requested'] and not braking['available']
                        if brake_cache is not None:
                            bd=brake_cache['data'];bc=brake_cache['columns'];bi=int(np.argmin(abs(bd[:,bc.index('sim_time')]-sim[index])))
                            if abs(bd[bi,bc.index('sim_time')]-sim[index])<=1e-6:
                                brake_delta=float(bd[bi,bc.index(name+'_applied_assist')]) if name+'_applied_assist' in bc else 0.
                                complete=True
                        final=float(np.clip(raw+correction+brake_delta,-limits.effort[j],limits.effort[j])) if complete else None
                        r['exact_native_pd_snapshot']=dict(received_at=float(command_times[li]),raw_desired_q_rad=float(c[0,j]),effective_desired_q_rad=desired,
                            desired_dq_rad_s=float(c[1,j]),kp=float(c[2,j]),kd=float(c[3,j]),feedforward_nm=float(c[4,j]),motor_enabled=bool(c[5,j]),
                            raw_pd_effort_nm=raw,filter_correction_nm=correction,target_stage_effort_nm=raw+correction,
                            braking_assist_nm=brake_delta if complete else None,complete_control_chain_verified=complete,
                            corrected_before_clip_nm=raw+correction+brake_delta if complete else None,expected_clipped_effort_nm=final,
                            matches_recorded_applied_effort=bool(abs(final-tau[index,j])<=1e-6) if complete else None)
            return r
        metrics={'position':(np.maximum(np.maximum(limits.lower-q,q-limits.upper),0),POSITION_TOLERANCE_RAD,'rad'),
                 'speed':(np.maximum(abs(dq)-limits.velocity,0),SPEED_TOLERANCE_RAD_S,'rad/s'),
                 'applied_effort':(np.maximum(abs(tau)-limits.effort,0),EFFORT_TOLERANCE_NM,'Nm')}
        counts={}
        for metric,(excess,tolerance,unit) in metrics.items():
            bad=excess>tolerance;counts[metric]=dict(supported_sample_joint_crossings=int(bad[support].sum()),unsupported_sample_joint_crossings=int(bad[~support].sum()))
            for j,name in enumerate(limits.names):
                for supported in (True,False):
                    mask=bad[:,j]&(support==supported)
                    if not mask.any():continue
                    ii=np.flatnonzero(mask);peak=int(ii[np.argmax(excess[ii,j])]);segments=[]
                    for seg in np.split(ii,np.where(np.diff(ii)>1)[0]+1):
                        segments.append(dict(first_sample=int(seg[0]),last_sample=int(seg[-1]),samples=len(seg),
                            first_sim_s=float(sim[seg[0]]),last_sim_s=float(sim[seg[-1]]),sampled_duration_s=float(len(seg)*.005)))
                    row=dict(joint=name,sdk_index=j,metric=metric,unit=unit,support=supported,samples=int(mask.sum()),
                        maximum_excess=float(excess[peak,j]),numerical_tolerance=tolerance,
                        lower_bound=float(limits.lower[j]) if metric=='position' else None,
                        upper_or_magnitude_bound=float(limits.upper[j] if metric=='position' else limits.velocity[j] if metric=='speed' else limits.effort[j]),
                        peak=context(peak,j),segments=segments,
                        last50ms_before_peak=[context(i,j) for i in range(max(0,peak-10),peak+1)])
                    report['violations'].append(row)
        report['violation_counts']=counts
        report['recorded_guard_thresholds']=meta.get('ground_guard',{}).get('thresholds')
        report['recorded_simulator_termination']={k:meta.get(k) for k in ('end_reason','error')}
        report['recorded_simulator_terminal_events']=meta.get('events',[])[-4:]
        report['diagnostic_complete']=True
    except Exception as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
    report['diagnostic_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'punktnull/skills/limits.py')}
    report['evidence_source_sha256']={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.exists()}
    return report


def markdown(report):
    out=['# Saved actual-limit diagnostic — '+Path(report['run']).name,'',
         '**Recorded outcome remains '+str(report['recorded_outcome_unchanged'])+'.** Diagnostic completion: '+str(report['diagnostic_complete'])+'. This report does not grant mission acceptance.','',
         '| Joint | Metric | Supported | Samples | Maximum excess | Peak simulation time |','|---|---|---:|---:|---:|---:|']
    for v in report['violations']:out.append(f"| {v['joint']} | {v['metric']} | {v['support']} | {v['samples']} | {v['maximum_excess']:.9g} {v['unit']} | {v['peak']['sim_time_s']:.6f}s |")
    if not report['violations']:out.append('| No comparisons/violations listed; inspect provenance/errors before interpreting | | | | | |')
    out+=['','Limiter telemetry available: '+str(report.get('limiter_telemetry',{}).get('available',False))+'. Missing telemetry is never reconstructed as measured filter behavior.','',
          'Full JSON includes50ms histories, actual raw/native context and provenance hashes. Reference phase remains inferred/submitted; contact forces are not measured.']
    if report['errors']:out+=['','Errors: '+repr(report['errors'])]
    return '\n'.join(out)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output-name',default='actual_limit_diagnosis_001.json');a=p.parse_args()
    if Path(a.output_name).name!=a.output_name or not a.output_name.endswith('.json'):p.error('Use a simple new.json output name')
    output=a.run/a.output_name;md=output.with_suffix('.md')
    if output.exists() or md.exists():p.error('Existing reports are never overwritten')
    result=diagnose(a.run);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');md.write_text(markdown(result))
    print(json.dumps(dict(run=str(a.run),diagnostic_complete=result['diagnostic_complete'],recorded_outcome=result['recorded_outcome_unchanged'],
        violations=[{k:v[k] for k in ('joint','metric','support','samples','maximum_excess')} for v in result['violations']],
        limiter_available=result.get('limiter_telemetry',{}).get('available'),braking_verified=result.get('braking_telemetry',{}).get('verified'),errors=result['errors']),indent=2))
    return 0 if result['diagnostic_complete'] else 2

if __name__=='__main__':raise SystemExit(main())
