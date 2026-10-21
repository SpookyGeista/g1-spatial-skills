#!/usr/bin/env python3
"""Read-only, reproducible three-run platform evidence/source comparison."""
from pathlib import Path
import argparse,ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
RUNS=('platform_002_followthrough','platform_003_yplus_yawplus','platform_004_yminus_yawminus')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build():
    records=[];snapshots=[]
    for name in RUNS:
        run=ROOT/'runs'/name;m=json.loads((run/'manifest.json').read_text())
        evaluation='evaluation_v2_native_release.json' if name==RUNS[0] else 'evaluation.json'
        supplement='evaluation_v3_timing_recording.json' if name==RUNS[0] else 'evaluation_timing_recording.json'
        e=json.loads((run/evaluation).read_text());t=json.loads((run/supplement).read_text())
        events={v['event']:v for v in m['events']}
        records.append(dict(run=str(run.relative_to(ROOT)),accepted=t['accepted'],variation=m['start_variation'],
            skill_seconds=events['native_planner_rejoined']['sim_time']-events['skill_started']['sim_time'],
            whole_recorded_run_seconds=t['recording']['first_last_sim_time'][1],measured=e['measured'],
            timing=t['timing'],uncut_recording_pass=t['uncut_recording_pass'],
            reference_digest=m['reference']['ordered_channel_digest'],
            sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in (run/'manifest.json',run/evaluation,run/supplement)}))
        snapshots.append(m['source_sha256'])
    allkeys=sorted(set.union(*(set(s) for s in snapshots)));identical={};different={}
    for key in allkeys:
        values=[s.get(key) for s in snapshots]
        if len(set(values))==1:identical[key]=values[0]
        else:different[key]=values
    definitions=('contact_module','runtime_module','sim_main','StairTrial','install_reused_run')
    execution_ast={}
    for name in RUNS:
        source=ROOT/'runs'/name/'source_snapshot/tools/stair_trial.py';tree=ast.parse(source.read_text())
        execution_ast[name]={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
            for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in definitions}
    active_equal=all(execution_ast[RUNS[0]]==execution_ast[n] for n in RUNS[1:])
    # Optional crawl-only files are enumerated by the broad snapshot glob;
    # they are not imported by this trial's custom motor-loop hook.
    unused=('punktnull/skills/ankle_braking.py','punktnull/skills/target_limits.py')
    allowed_differences=set(unused)|{'tools/stair_trial.py'}
    sources_consistent=active_equal and not(set(different)-allowed_differences)
    return dict(schema=1,kind='Three actual isolated platform repeats, one baseline plus two start variations',
        sample_size=3,accepted_runs=sum(bool(r['accepted']) for r in records),records=records,
        same_reference=len(set(r['reference_digest'] for r in records))==1,
        declared_source_comparison_pass=sources_consistent,
        identical_snapshotted_sources=identical,different_snapshotted_sources=different,
        active_stair_execution_ast_sha256=execution_ast,
        difference_explanation={'tools/stair_trial.py':'Only post-run evaluate changed: strict owned phase plus bounded explicit paused native release. All five execution definitions byte-normalized-AST identical.',
            **{x:'Optional crawl-only control module; no import/hook from executed platform custom sim_main/guard/runtime/reference; not applied.' for x in unused}},
        supplementary_current_dependency_hashes={str(p.relative_to(ROOT)):sha(p) for p in (
            ROOT/'punktnull/terrain/kinematics.py',ROOT/'tools/native_reference_evidence.py',ROOT/'tools/platform_timing_audit.py',Path(__file__))},
        provenance_limit='Comparison covers retained per-run source snapshots. Kinematics helper was not included in those manifests; current hash is supplementary and not a retrospective execution snapshot.',
        failures_preserved=['runs/stairs_001_two_riser','runs/stairs_002_swing_clearance','runs/stairs_003_swing_clearance','runs/stairs_004_second_swing','runs/platform_001_stepup'],
        limitations=['One configured120mm riser only; no arbitrary staircase.',
            'No sensor-derived admission or full building-mission integration.',
            'Actual200Hz sampled limits, ankle-filtered contact forces and local G1 simulation; no physical robot or external validation.',
            'Component test automatic supervisor ownership release; no live human intervention.'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=build()
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(accepted_runs=result['accepted_runs'],sample_size=result['sample_size'],same_reference=result['same_reference'],source_comparison=result['declared_source_comparison_pass']),indent=2))
if __name__=='__main__':main()
