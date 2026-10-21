#!/usr/bin/env python3
"""Read-only census of every G1 LIO/skill run; existing audits, never re-evaluation.

Only explicit numeric report versions are ordered. Missing, malformed, ambiguous
or stale reports cannot confer success. Earlier results remain in each history.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    raw=Path(path).read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def versioned_reports(folder, stem, pass_key, manifest_hash):
    pattern=re.compile(re.escape(stem)+r'(?:_v([1-9][0-9]*))?\.json$')
    candidates=[]
    for path in folder.glob(stem+'*.json'):
        match=pattern.fullmatch(path.name)
        if match:candidates.append((int(match.group(1) or 1),path))
    candidates.sort(key=lambda pair:(pair[0],pair[1].name))
    history=[];data_by_path={}
    for version,path in candidates:
        record=dict(file=path.name,version=version,path=str(path),verdict='INVALID_REPORT',failed_checks=[])
        try:
            data,sha=load(path);record['sha256']=sha;data_by_path[str(path)]=data
            checks=data.get('checks',{})
            bad=[k for k,v in checks.items() if v is False]
            bad=list(dict.fromkeys(bad+data.get('failed_checks',[])))
            errors=data.get('errors') or data.get('error')
            value=data.get(pass_key)
            declared_hash=(data.get('input_sha256',{}).get('manifest.json') or data.get('source_sha256',{}).get('manifest.json'))
            record.update(failed_checks=bad,errors=errors,declared_pass=value,
                          manifest_hash_matches=declared_hash==manifest_hash if declared_hash else None)
            if declared_hash and declared_hash!=manifest_hash:record['verdict']='STALE_MANIFEST_REPORT'
            elif value is True and not bad and not errors:record['verdict']='PASS'
            elif value is False:record['verdict']='FAIL'
            elif value is True:record['verdict']='INCONSISTENT_REPORT'
            else:record['error']='Missing explicit boolean '+pass_key
        except Exception as exc:record['error']=repr(exc)
        history.append(record)
    if not history:return dict(verdict='MISSING',selected=None,history=[]),{}
    latest=history[-1]
    if sum(r['version']==latest['version'] for r in history)>1:
        return dict(verdict='AMBIGUOUS_VERSION',selected=None,history=history),{}
    return dict(verdict=latest['verdict'],selected=latest['path'],history=history,
                earlier_failed_reports=[r['path'] for r in history[:-1] if r['verdict']!='PASS'],
                failed_checks=latest['failed_checks']),data_by_path.get(latest['path'],{})


def status_for(outcome, completed, report_verdict):
    if not completed:return 'IN_PROGRESS_OR_UNFINALIZED'
    if outcome in ('FAILED','ABORTED','FAILURE'):
        return 'CONFLICTING_EXECUTION_AND_REPORT' if report_verdict=='PASS' else 'EXECUTION_FAILED'
    if report_verdict=='PASS':return 'AUDITED_PASS_IN_REPORTED_SCOPE'
    if report_verdict=='FAIL':return 'AUDITED_FAIL'
    return 'UNVERIFIED_'+report_verdict


def optional_json(path, errors):
    if not path.is_file():return {},None
    try:return load(path)
    except Exception as exc:errors.append(dict(path=str(path),error=repr(exc)));return {},None


def row_for(run, family):
    row=dict(id=run.name,family=family,path=str(run),errors=[])
    manifest,mhash=optional_json(run/'manifest.json',row['errors'])
    outcome=manifest.get('outcome');completed=bool(outcome and outcome not in ('RUNNING','PENDING','STARTING','INITIALIZING'))
    row.update(manifest_sha256=mhash,declared_outcome=outcome,completed=completed,
               execution_failure=manifest.get('failure') or manifest.get('error'),
               live_human_tested=manifest.get('live_human_tested'),full_prototype_acceptance='NOT_CONFERRED_BY_THIS_CENSUS')
    primary,data=versioned_reports(run,'evaluation_g1_lio' if family=='lio' else 'skill_evaluation',
                                    'audit_pass' if family=='lio' else 'accepted',mhash)
    target,target_data=versioned_reports(run,'target_evidence_audit','pass_all',mhash)
    review,review_data=versioned_reports(run/'mission_review_replay','summary','native_replay_review_pass',mhash)
    row['primary_audit']=primary;row['target_audit']=target;row['recorded_handover_audit']=review
    row['negative_fault_audits']=[]
    for folder in sorted(run.glob('fault_test_*')):
        if not folder.is_dir():continue
        fault,fault_data=versioned_reports(folder,'summary','negative_test_pass',mhash)
        expected_fault=folder.name.removeprefix('fault_test_')
        semantic_checks=dict(declared_fault_matches_folder=fault_data.get('fault')==expected_fault,
            mission_failed_in_manifest=outcome in ('FAILED','ABORTED','FAILURE'),
            mission_outcome_matches_manifest=fault_data.get('mission_outcome')==outcome,
            explicitly_not_completed=fault_data.get('mission_completed') is False,
            explicitly_not_live_acceptance=fault_data.get('live_human_acceptance') is False)
        if fault['verdict']=='PASS' and not all(semantic_checks.values()):
            fault['verdict']='INCONSISTENT_REPORT'
            fault['failed_checks']+=['census_'+k for k,v in semantic_checks.items() if not v]
        row['negative_fault_audits'].append(dict(fault=expected_fault,audit=fault,semantic_checks=semantic_checks,
            observed_physics=fault_data.get('physics',{}),scope='Negative fault response only; never a completed mission or human recovery'))
    row['declared_core_runtime_sha256']={k:v for k,v in manifest.get('extension_source_sha256',{}).items()
        if k in ('run_mission.py','run_isaac.py','tools/lio_bridge.py') or k.startswith(('punktnull/','ros2_ws/config/'))}
    row['status']=status_for(outcome,completed,primary['verdict'])
    if row['errors'] and row['status']=='AUDITED_PASS_IN_REPORTED_SCOPE':row['status']='UNVERIFIED_INPUT_ERROR'
    metadata,_=optional_json(run/'sim/metadata.json',row['errors'])
    sensor,_=optional_json(run/'sim/mid360/metadata.json',row['errors'])
    recording,_=optional_json(run/'sim/recording.json',row['errors'])
    source,_=optional_json(run/'mission_review_replay/source.json',row['errors'])
    events=manifest.get('events',[])
    if (run/'events.jsonl').is_file():
        try:events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines() if line.strip()]
        except Exception as exc:row['errors'].append(dict(path=str(run/'events.jsonl'),error=repr(exc)))
    counts=dict(assistance_requests=sum(e.get('event')=='transition' and e.get('state')=='ASSISTANCE_REQUESTED' for e in events),
                operator_control_intervals=sum(e.get('event')=='transition' and e.get('state')=='OPERATOR_CONTROL' for e in events),
                explicit_release_commands=sum(e.get('event')=='operator_command' and e.get('command')=='release' for e in events),
                autonomy_resumed_events=sum(e.get('event')=='autonomy_resumed' for e in events))
    row['interventions']=dict(counts,pose_source=source.get('source'),count_source='retained events; does not authenticate a live person',
        native_handover_review_pass=review['verdict']=='PASS' if review['selected'] else None,
        useful_operator_recovery_demonstrated=review_data.get('useful_operator_recovery_demonstrated'))
    physics=data.get('physics',{});measured=data.get('measured',{})
    ratios=data.get('timing',{}).get('ratios',[])
    finite_ratios=[float(v) for v in ratios if isinstance(v,(int,float)) and math.isfinite(v)]
    lio_timing=physics.get('timing',{})
    windows=lio_timing.get('windows',[])
    row['timing']=dict(manifest_mission_wall_s=manifest.get('mission_seconds'),total_wall_s=manifest.get('total_wall_seconds'),
        audit_mission_wall_s=physics.get('mission_wall_s'),skill_wall_s=measured.get('skill_wall_seconds'),
        simulation_to_wall_ratio=lio_timing.get('overall_ratio'),
        observed_window_ratio_min=min([r['ratio'] for r in windows]+finite_ratios) if windows or finite_ratios else None,
        observed_window_ratio_max=max([r['ratio'] for r in windows]+finite_ratios) if windows or finite_ratios else None,
        timing_gate_pass=data.get('checks',{}).get('mission_loop_ratio_in_retained_band',data.get('checks',{}).get('controller_physics_realtime_band')))
    row['return']=dict(physical_distance_to_mission_start_m=physics.get('physical_return_distance_m'),
        estimated_distance_manifest_m=manifest.get('return_distance_m'),
        stable_terminal_fraction=physics.get('terminal_stability_fraction'),
        skill_planner_return_observed=measured.get('return_observed'))
    row['conditions']=dict(test=manifest.get('test'),target_room=manifest.get('target_room_for_evaluation_only'),
        target_present=manifest.get('target_present_for_evaluation_only'),scene_variant=manifest.get('scene_variant_for_evaluation_only'),
        initial_variation=metadata.get('initial_variation'),lidar_rays_per_second=sensor.get('config',{}).get('rays_per_second'),
        sensor_model=manifest.get('sensor_model'),terrain=manifest.get('terrain','flat' if family=='skills' else None),
        skill_fixture=measured.get('passage',{}).get('fixture'),clearance_source=manifest.get('clearance_source'),
        localization=manifest.get('localization'),autonomous_skill_admission=manifest.get('autonomous_skill_admission'))
    trajectory=data.get('trajectory',{})
    row['lio']=dict(integration_pass=data.get('sensor_lio_integration_pass'),translation_rmse_m=trajectory.get('translation_rmse_m'),
        translation_max_m=trajectory.get('translation_max_m'),rotation_max_deg=trajectory.get('rotation_max_deg'),
        matched_scans=trajectory.get('matched_post_initialization_scans'),post_initialization_scans=trajectory.get('post_initialization_scans'))
    row['rooms']=dict(physical_entries=[r['id'] for r in physics.get('rooms',[]) if r.get('pelvis_center_entered')],
        declared_completed=[r.get('id') for r in manifest.get('completed_room_scans',[])],
        failed_scans=[{k:v for k,v in r.items() if k not in ('retained_camera_frames',)} for r in data.get('room_scans',{}).get('rooms',[]) if not r.get('pass')])
    row['skill']=dict(native_reference_stream_validated=manifest.get('native_reference_stream_validated'),
        phase=manifest.get('session_phase'),tracking_p95_height_m_angle_deg_joint_rms_rad=measured.get('ground_tracking_p95_height_m_orientation_deg_joint_rms_rad'),
        ground_displacement_m=measured.get('ground_displacement_m'),passage=measured.get('passage'),
        actual_native_joint_stream=data.get('checks',{}).get('actual_native_joint_stream'))
    row['recording']=dict(file_present=(run/'sim/uncut_runtime.mp4').is_file(),declared_frames=recording.get('frames'),
        declared_fps=recording.get('fps'),full_decode_verified=data.get('checks',{}).get('recording_complete_decodable'),
        decoded_frames=data.get('recording',{}).get('decode',{}).get('decoded_frames'),
        time_basis=recording.get('time_basis'))
    # Audit validity can depend on late-read inputs; never let a parse error confer pass.
    if row['errors'] and row['status']=='AUDITED_PASS_IN_REPORTED_SCOPE':row['status']='UNVERIFIED_INPUT_ERROR'
    return row


def series_summary(rows, specification):
    name, separator, references=specification.partition('=')
    if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('Series syntax NAME=010,011,012')
    references=references.split(',')
    if not references or len(references)!=len(set(references)) or any(not re.fullmatch(r'[0-9]{3}',x) for x in references):raise ValueError('Use unique three-digit LIO run references')
    selected=[];missing=[]
    for ref in references:
        matches=[r for r in rows if r['family']=='lio' and r['id'].startswith('lio_'+ref+'_')]
        if len(matches)>1:raise ValueError('Ambiguous series run '+ref)
        if matches:selected.append(matches[0])
        else:missing.append(ref)
    revisions={};snapshot_changes=[]
    for row in selected:
        core=row.get('declared_core_runtime_sha256',{})
        fingerprint=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest() if core else 'MISSING'
        revisions.setdefault(fingerprint,[]).append(row['id'])
        for source_name,expected in core.items():
            base=Path(row['path'])/'source_snapshot/extension';path=base/source_name
            actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.resolve().is_relative_to(base.resolve()) and path.is_file() else None
            if actual!=expected:snapshot_changes.append(dict(run=row['id'],file=source_name,actual_sha256=actual,declared_sha256=expected))
    return dict(name=name,explicit_run_references=references,expected_trials=len(references),missing_runs=missing,
        rows=[dict(id=r['id'],completed=r['completed'],status=r['status'],primary_report=r['primary_audit']['selected']) for r in selected],
        completed=sum(r['completed'] for r in selected),bounded_audit_passes=sum(r['completed'] and r['status']=='AUDITED_PASS_IN_REPORTED_SCOPE' for r in selected),
        lio_and_timing_passes=sum(r['completed'] and r['lio']['integration_pass'] is True and r['timing']['timing_gate_pass'] is True for r in selected),
        separate_target_audit_passes=sum(r['target_audit']['verdict']=='PASS' for r in selected),
        all_bounded_audits_pass=bool(len(selected)==len(references) and all(r['completed'] and r['status']=='AUDITED_PASS_IN_REPORTED_SCOPE' for r in selected)),
        core_runtime_revision_groups=revisions,saved_core_snapshot_changes=snapshot_changes,
        saved_core_snapshots_match=bool(selected and 'MISSING' not in revisions and not snapshot_changes),
        declared_core_runtime_same=bool(len(selected)==len(references) and len(revisions)==1 and 'MISSING' not in revisions),
        revision_comparison_scope='Declared run_mission/run_isaac, punktnull Python, ROS configuration and lio_bridge hashes, with saved snapshots rehashed by this census; tools/tests and scenario arguments differ independently. Does not attest an unhashed executed binary.')


def census(root, series=()):
    paths=[(p,'lio') for p in sorted(root.glob('lio_*')) if p.is_dir()]
    paths += [(p,'skills') for p in sorted((root/'skills').glob('*')) if p.is_dir()]
    rows=[row_for(p,family) for p,family in paths]
    completed=[r for r in rows if r['completed']]
    return dict(schema_version=2,generated_at_utc=datetime.now(timezone.utc).isoformat(),runs_root=str(root),
        scope='Census of existing reports and manifest outcomes, not a fresh physics/vision/native audit or a full-prototype acceptance decision.',
        selection_policy='Highest explicit numeric _vN version per named report family, base filename=version1. No mtime ordering or fallback after malformed latest report. Every earlier result remains in history.',
        missing_report_policy='A claimed successful execution without an applicable audit is UNVERIFIED. Running/unfinalized entries are separate from completed totals.',
        counts=dict(discovered=len(rows),completed=len(completed),in_progress_or_unfinalized=len(rows)-len(completed),
            completed_statuses=dict(Counter(r['status'] for r in completed)),
            separate_negative_fault_audits=dict(Counter(f['audit']['verdict'] for r in completed for f in r['negative_fault_audits'])),
            by_family={family:dict(Counter(r['status'] for r in completed if r['family']==family)) for family in ('lio','skills')}),
        no_pooled_reliability_claim='These include startup failures, steering diagnostics, different revisions, sensor budgets, recorded-input tests and isolated skills. Status counts are not reliability estimates.',
        rows=rows,explicit_series=[series_summary(rows,spec) for spec in series],tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def fmt(value, digits=3):
    if value is None:return '—'
    if isinstance(value,float):return f'{value:.{digits}f}'
    return str(value).replace('|','\\|').replace('\n',' ')


def markdown(report):
    lines=['# G1 LIO and skill run census','',f"Snapshot: {report['generated_at_utc']}",'',report['scope'],'',report['no_pooled_reliability_claim'],'',
        f"{report['counts']['completed']} completed entries; {report['counts']['in_progress_or_unfinalized']} running/unfinalized. Missing reports never count as passes.",'']
    if report.get('explicit_series'):
        lines+=['## Explicit series (never pooled)','','| Series | Completed / expected | Bounded audit pass | LIO + timing pass | Separate target audit pass | Same declared core runtime |','|---|---|---|---|---|---|']
        for group in report['explicit_series']:
            lines.append(f"| {group['name']} | {group['completed']}/{group['expected_trials']} | {group['bounded_audit_passes']}/{group['expected_trials']} | {group['lio_and_timing_passes']}/{group['expected_trials']} | {group['separate_target_audit_passes']}/{group['expected_trials']} | {group['declared_core_runtime_same'] and group['saved_core_snapshots_match']} |")
        lines.append('')
    for family,title in [('lio','LIO runs'),('skills','Isolated skill trials')]:
        lines+=['## '+title,'','| Run | Manifest outcome | Latest audit / census status | Condition | Wall s / ratio | Assistance / POSE / resume | Return | Failure / scope |','|---|---|---|---|---|---|---|---|']
        for r in report['rows']:
            if r['family']!=family:continue
            c=r['conditions'];t=r['timing'];i=r['interventions'];ret=r['return'];primary=r['primary_audit']
            condition=(f"{c['lidar_rays_per_second'] or '?'} rays/s; target {c['target_room']}; variant {c['scene_variant']}; yaw {(c['initial_variation'] or {}).get('yaw_deg','?')}°" if family=='lio' else f"{c['terrain']}; {c['skill_fixture'] or 'fixture not audited'}")
            seconds=t['manifest_mission_wall_s'] if family=='lio' else t['skill_wall_s']
            if seconds is None:seconds=t['total_wall_s']
            timing=fmt(seconds)+' / '+fmt(t['simulation_to_wall_ratio'])
            if family=='skills' and t['observed_window_ratio_min'] is not None:timing+=f" [{fmt(t['observed_window_ratio_min'])}–{fmt(t['observed_window_ratio_max'])}]"
            returned=(fmt(ret['physical_distance_to_mission_start_m'])+' m' if ret['physical_distance_to_mission_start_m'] is not None else ('planner '+fmt(ret['skill_planner_return_observed']) if family=='skills' else 'not audited'))
            failure=', '.join(primary.get('failed_checks',[])) or r['execution_failure'] or ('bounded audit only' if primary['verdict']=='PASS' else 'no applicable verified pass')
            link=f"[{r['id']}]({primary['selected'] or str(Path(r['path'])/'manifest.json')})"
            lines.append('| '+' | '.join([link,fmt(r['declared_outcome']),fmt(primary['verdict']+' / '+r['status']),fmt(condition),timing,
                f"{i['assistance_requests']} / {i['operator_control_intervals']} / {i['autonomy_resumed_events']}",returned,fmt(failure)])+' |')
        lines.append('')
    lines+=['## Separate target and recorded-input audits','',
        '| Run | RGB-D target audit | Recorded native handover audit | Source / live-human flag |','|---|---|---|---|']
    for r in report['rows']:
        if not r['target_audit']['selected'] and not r['recorded_handover_audit']['selected']:continue
        cells=[]
        for key in ['target_audit','recorded_handover_audit']:
            audit=r[key];cells.append(f"[{audit['verdict']}]({audit['selected']})" if audit['selected'] else audit['verdict'])
        lines.append(f"| {r['id']} | {' | '.join(cells)} | {fmt(r['interventions']['pose_source'])}; live={fmt(r['live_human_tested'])} |")
    lines+=['','## Separate negative fault diagnostics','',
        'These checks test bounded fault handling. Their mission outcomes remain FAILED; their passes do not enter a completed-mission or human-recovery numerator.','',
        '| Run | Injected fault | Negative diagnostic | Mission outcome | Last physics after abort |','|---|---|---|---|---|']
    for r in report['rows']:
        for f in r['negative_fault_audits']:
            a=f['audit'];link=f"[{a['verdict']}]({a['selected']})" if a['selected'] else a['verdict']
            lines.append(f"| {r['id']} | {f['fault']} | {link} | {r['declared_outcome']} | {fmt(f['observed_physics'].get('last_physics_minus_abort_wall_s'),6)} wall s |")
    lines+=['','## Earlier report history retained','']
    for r in report['rows']:
        audit_items=[(key,r[key]) for key in ['primary_audit','target_audit','recorded_handover_audit']]
        audit_items += [('negative_fault_'+f['fault'],f['audit']) for f in r['negative_fault_audits']]
        for key,audit in audit_items:
            history=audit['history']
            if len(history)>1:
                lines.append('- '+r['id']+' / '+key+': '+' → '.join(f"[{h['file']}]({h['path']}) {h['verdict']}" for h in history)+'.')
    lines+=['',report['selection_policy'],'',
        'Mission durations exclude startup when reported; skill durations cover the reference execution interval. “Resume” counts retained autonomy_resumed events; native validation remains a separate audit. Every video retains its own simulation-time encoding disclosure. A finite map, reported success, recorded POSE or isolated passage traversal does not establish live useful intervention, integrated terrain navigation, physical validation or external-user acceptance.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,default=ROOT/'runs');p.add_argument('--output',type=Path,required=True);p.add_argument('--series',action='append',default=[],help='Explicit grouping, e.g. initial10k=010,011,012; repeatable')
    a=p.parse_args();target=a.output.resolve();md=target.with_suffix('.md')
    if target.suffix!='.json' or target.exists() or md.exists():p.error('Use a new .json output; companion Markdown must not exist')
    if not target.parent.is_dir():p.error('Output parent must exist')
    result=census(a.runs.resolve(),a.series);body=markdown(result)
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with md.open('x') as f:f.write(body)
    print(json.dumps(dict(output=str(target),markdown=str(md),counts=result['counts']),indent=2))

if __name__=='__main__':main()
