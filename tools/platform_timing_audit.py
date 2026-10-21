#!/usr/bin/env python3
"""Separate actual timing/uncut evidence supplement; never alters runtime."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def timing_metrics(wall,sim,begin,end):
    wall=np.asarray(wall,float);sim=np.asarray(sim,float)
    valid=bool(len(wall)>1 and len(wall)==len(sim) and np.isfinite(wall).all() and np.isfinite(sim).all()
        and np.all(np.diff(wall)>0) and np.all(np.diff(sim)>0) and wall[0]<=begin<end<=wall[-1])
    if not valid:return dict(accepted=False,reason='Missing/nonfinite/nonmonotonic full actual clock interval')
    ratio=float((np.interp(end,wall,sim)-np.interp(begin,wall,sim))/(end-begin))
    windows=[];start=begin
    while start+5<=end:
        stop=start+5;value=float((np.interp(stop,wall,sim)-np.interp(start,wall,sim))/5)
        windows.append(dict(wall_begin=start,wall_end=stop,ratio=value,passed=.95<=value<=1.05));start=stop
    return dict(accepted=.95<=ratio<=1.05 and all(x['passed'] for x in windows),
        ratio=ratio,wall_duration_seconds=end-begin,full_5s_windows=windows,
        window_basis='Consecutive complete5s wall-clock windows anchored at actual skill_started; terminal remainder included in overall ratio',
        limits=[.95,1.05])

def audit(run,evaluation):
    run=Path(run).resolve();evaluation=Path(evaluation).resolve()
    if not evaluation.is_relative_to(run):raise ValueError('Evaluation must belong to this run')
    base=json.loads(evaluation.read_text());m=json.loads((run/'manifest.json').read_text())
    events={e['event']:e for e in m['events']};begin=events.get('skill_started');end=events.get('native_planner_rejoined')
    z=np.load(run/'sim/trace.npz',allow_pickle=False);a=z['data'];cols=list(z['columns'])
    timing=timing_metrics(a[:,cols.index('wall_monotonic')],a[:,cols.index('sim_time')],begin['monotonic'],end['monotonic']) if begin and end else dict(accepted=False,reason='Skill never completed')
    recording=json.loads((run/'sim/recording.json').read_text());video=run/'sim/uncut_runtime.mp4'
    rows=[json.loads(x) for x in (run/'sim/video_timestamps.jsonl').read_text().splitlines()]
    times=np.array([x['sim_time'] for x in rows]);walls=np.array([x['wall_monotonic'] for x in rows])
    import cv2
    cap=cv2.VideoCapture(str(video));decoded=0
    stream=dict(fps=cap.get(cv2.CAP_PROP_FPS),width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        valid,frame=cap.read()
        if not valid:break
        decoded+=1
    cap.release()
    recording_ok=bool(recording.get('uncut') is True and recording.get('rendered_from_running_physx') is True
        and recording.get('error') is None and decoded==recording['frames']==len(rows)
        and len(times)>1 and np.isfinite(times).all() and np.isfinite(walls).all() and np.all(np.diff(walls)>0)
        and np.all(np.abs(np.diff(times)-1/recording['fps'])<1e-6)
        and begin and end and times[0]<=begin['sim_time'] and times[-1]>=end['sim_time']-1/recording['fps']-.01)
    paths=[evaluation,run/'manifest.json',run/'sim/trace.npz',run/'sim/recording.json',run/'sim/video_timestamps.jsonl',video,Path(__file__)]
    return dict(schema=1,kind='Post-run platform component acceptance + actual timing and uncut supplement',
        accepted=bool(base.get('accepted') and timing['accepted'] and recording_ok),
        underlying_evaluation_pass=bool(base.get('accepted')),timing=timing,uncut_recording_pass=recording_ok,
        recording=dict(actual_decoded_frames=decoded,timestamp_rows=len(rows),stream=stream,
            first_last_sim_time=times[[0,-1]].tolist()),
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitation='Isolated configured single120mm platform; no building integration, general stairs, live operator or hardware evidence')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Output must be new; preserve prior evaluations')
    report=audit(a.run,a.evaluation)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(accepted=report['accepted'],timing=report['timing'],uncut=report['uncut_recording_pass']),indent=2))
    return 0 if report['accepted'] else 2
if __name__=='__main__':raise SystemExit(main())
