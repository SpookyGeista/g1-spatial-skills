#!/usr/bin/env python3
"""CPU-only diagnosis of retained measured crawl trials; never changes gates."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from punktnull.skills.legacy import OLD,LEGACY,WORKSPACE,ground_module,load
from punktnull.skills.session import REGISTRY


def analyze(path):
    path=Path(path).resolve();ref=ground_module().Reference(REGISTRY['crawl'].source)
    names=load('tracking_audit_bridge',LEGACY/'bridge.py').BODY_JOINT_NAMES
    trace=path/'sim/trace.npz';meta_path=path/'sim/metadata.json'
    meta=json.loads(meta_path.read_text());z=np.load(trace,allow_pickle=False);data=z['data'];cols=list(z['columns'])
    get=lambda key:data[:,cols.index(key)]
    frame=get('guard_reference_frame').astype(int)
    chosen=(frame>=0)&(get('guard_ground_permission')>.5);frame=frame[chosen]
    q=data[chosen][:,[cols.index('q_'+str(i)) for i in range(29)]]
    tau=data[chosen][:,[cols.index('tau_'+str(i)) for i in range(29)]]
    error=q-ref.joints[frame];rms=np.sqrt(np.mean(error*error,axis=1));bad=rms>.35
    limits=np.asarray(meta['effort_limits_nm'])
    report={'run':str(path),'samples':int(chosen.sum()),'above_035_samples':int(bad.sum()),
        'joint_rms_p95_rad':float(np.percentile(rms,95)),'joint_rms_max_rad':float(rms.max()),
        'phase_basis':'saved per-physics-step guard_reference_frame and padded ground permission; historical keyboard and stream inference differ',
        'phases':[],'largest_bad_sample_joint_mse':[], 'wrist_limits':[]}
    timeline=json.loads((ref.source/'build_metadata.json').read_text())['timeline']
    for phase in timeline:
        mask=(frame>=phase['start_frame'])&(frame<phase['end_frame_exclusive'])
        report['phases'].append({'id':phase['id'],'samples':int(mask.sum()),
            'joint_rms_p95_rad':float(np.percentile(rms[mask],95)) if mask.any() else None,
            'above_035_samples':int((bad&mask).sum())})
    if bad.any():
        mean=np.mean(error[bad]**2,axis=0)
        report['largest_bad_sample_joint_mse']=[{'joint':names[i],'sdk_index':int(i),'mean_squared_error_rad2':float(mean[i])} for i in np.argsort(mean)[-6:][::-1]]
        for i in [20,27,21,28]:
            report['wrist_limits'].append({'joint':names[i],'sdk_index':i,'unchanged_effort_limit_nm':float(limits[i]),
              'above_035_actual_angle_range_rad':[float(q[bad,i].min()),float(q[bad,i].max())],
              'above_035_reference_angle_range_rad':[float(ref.joints[frame[bad],i].min()),float(ref.joints[frame[bad],i].max())],
              'above_035_torque_at_99pct_limit_fraction':float(np.mean(abs(tau[bad,i])>=.99*limits[i]))})
    peaks=np.argsort(rms)[-3:][::-1]
    report['peak_examples']=[{'frame':int(frame[i]),'joint_rms_rad':float(rms[i]),
        'left_right_wrist_pitch_reference_rad':ref.joints[frame[i],[20,27]].tolist(),
        'left_right_wrist_pitch_actual_rad':q[i,[20,27]].tolist(),
        'left_right_wrist_pitch_torque_nm':tau[i,[20,27]].tolist()} for i in peaks]
    lower=[]
    for qi,fi in zip(q[::4],frame[::4]):
        candidates=ref.joints[max(0,fi-20):min(len(ref.joints),fi+21)]
        lower.append(float(np.sqrt(np.mean((candidates-qi)**2,axis=1)).min()))
    report['diagnostic_only_optimistic_joint_fit']={'sampling':'every fourth selected physics sample',
        'search':'±20 reference frames; unconstrained per-sample minimum JOINT error only, not valid execution alignment',
        'p95_joint_rms_rad':float(np.percentile(lower,95)),
        'changes_acceptance':False}
    paths=[trace,meta_path,ref.source/'joint_pos.csv',ref.source/'build_metadata.json']
    logpath=path/'controller.log';submitted=path/'submitted_reference.jsonl'
    if submitted.exists() and logpath.exists():
        sends=[json.loads(line) for line in submitted.read_text().splitlines()];log=logpath.read_text()
        values=np.array([tuple(map(int,m)) for m in re.findall(r'incoming_frame_start: (\d+).*?global_playback_frame: (\d+)',log)])
        def counts(a):
            v,n=np.unique(a,return_counts=True);return {str(int(x)):int(y) for x,y in zip(v,n)}
        report['native_merger']={'submitted_packets':len(sends),'logged_packet_merges':len(values),
            'incoming_minus_native_frame_lag_min_median_p95_max':np.quantile(values[:,0]-values[:,1],[0,.5,.95,1]).tolist(),
            'initial_or_later_catchup_messages':len(re.findall(r'Catch-up: Reset to frame',log)),
            'waiting_following_motion_messages':log.count('waiting following motion'),
            'received_packet_wire_index_differences':counts(np.diff(values[:,0])),
            'native_playback_frame_differences_between_received_packets':counts(np.diff(values[:,1])),
            'interpretation':'Packet merges are not every native control tick; a 2-frame difference between received packets does not prove skipped execution. Overlapping 51-frame windows cover missing packets.'}
        paths.extend([logpath,submitted])
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',default=[OLD/'data/isaac_sonic_bones_crawl_2026-09-16/trials/crawl_only_gui_01',WORKSPACE/'runs/skills/003_stream_flat',WORKSPACE/'runs/skills/005_aligned_clock'])
    parser.add_argument('--output',type=Path,default=WORKSPACE/'reports/skill_tracking_005')
    args=parser.parse_args();jp=args.output.with_suffix('.json');mp=args.output.with_suffix('.md')
    if jp.exists() or mp.exists():parser.error('Choose a new report path; prior evidence stays unchanged')
    reports=[analyze(p) for p in args.runs]
    result={'scope':'Read-only actual-trace analysis; no model inference, simulator, gate or source changes',
        'runs':reports,'findings':[
            '005 completes the physical cycle and native planner rejoin but retains tracking failure.',
            '005 early alignment phases and central crawl pass .35 individually; crawl-stop and get-up exceed it.',
            'Large measured wrist-angle deviations coincide with unchanged 5 Nm applied-torque saturation; historical keyboard baseline already exhibits the same get-up limitation.',
            'Even an optimistic wider joint-only phase fit remains above .35 for 003 and005; phase-window widening cannot explain away the failure.',
            'The logs show only initial stream initialization catch-up and no waiting-for-following-motion. They do not establish every executed reference frame.',
            'Contact-loading cause is plausible but not force-verified: current trials did not record filtered wrist/hand contact force. Do not call it proven floor-force measurement.'],
        'next_bounded_check':['Add existing native --target-motion-logfile to record actual current joint/quaternion targets without controller edits.',
            'Record actual wrist/forearm contact loads in the next permitted simulation to distinguish load-limited wrist deviation from policy target deviation.',
            'Evaluate an existing elbow-supported reference or native elbow-crawl candidate on flat ground before admitting a tunnel. Preserve5Nm limits and.35gate; do not retry until random pass.']}
    jp.parent.mkdir(parents=True,exist_ok=True);jp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Crawl-Tracking: tatsächliche Läufe, unveränderte Grenze','',
        'Die ausgeführte Bewegung und der Rückwechsel zum Planner funktionieren; die neue Kriechfähigkeit besteht die Tracking-Abnahme weiterhin nicht. Diese Auswertung verändert weder Kriterien noch Laufresultate.','',
        '| Lauf | Ground-Samples | RMS >0,35 rad | P95 Gelenk-RMS | Optimistische ±20-Frame-Untergrenze¹ |','|---|---:|---:|---:|---:|']
    for r in reports:lines.append(f"| {Path(r['run']).name} | {r['samples']} | {r['above_035_samples']} | {r['joint_rms_p95_rad']:.6f} | {r['diagnostic_only_optimistic_joint_fit']['p95_joint_rms_rad']:.6f} |")
    lines+=['','¹Nur Diagnose: pro Sample beliebige minimale Gelenkabweichung im Suchfenster, keine zulässige neue Phasenbewertung. Historische Keyboard-Phase und neue inferierte Phase sind unterschiedlich. Der alte offizielle P95-Wert bleibt0,349766rad; hier wird für alle Läufe die gespeicherte gepolsterte Ground-Maske verwendet.','',
       'Im Lauf005 ist die Hinlegephase nach dem zusätzlichen gemessenen Startabgleich besser. Die Grenzverletzungen liegen jetzt in Kriech-Stopp und Aufstehen. Bei96,1% seiner230über der Grenze liegenden Samples erreicht das rechte Wrist-Pitch-Gelenk mindestens99% seines unveränderten5Nm-Limits. Bei einem Peak liegt das tatsächliche rechte Handgelenk bei−1,201rad, die Referenz bei+0,012rad, während+5Nm anliegen. Das ist eine erhebliche gemessene Gelenkabweichung und nicht allein ein Phasenfehler.','',
       'Auch der alte knappe Keyboard-Pass zeigt Handgelenk-Sättigung und einen Aufsteh-P95 von etwa0,407rad;005 liegt dort bei0,409rad. Das Grundproblem war bereits im vorhandenen Kandidaten enthalten. Die Zuordnung zu Hand-/Bodenkontakten ist plausibel, aber ohne Kontaktkraftaufzeichnung nicht bewiesen.','',
       'Native Merge-Logs zeigen in005 etwa sechs Frames Abstand zwischen eingehender Referenz und aktuellem Playback, nur die erwartete Initialisierung und kein Warten auf Folgedaten. Einzelne ausgelassene Pakete werden durch überlappende51-Frame-Fenster abgedeckt. Die Empfangslogs sind keine lückenlose Liste aller tatsächlich ausgeführten Referenzframes.','',
       'Nächster gezielter Test: den bereits vorhandenen nativen `--target-motion-logfile` aktivieren und Hand-/Unterarmkontaktkräfte messen. Danach einen bereits vorhandenen ellbogengestützten Kandidaten prüfen, falls die Lastverteilung ursächlich ist. Keine Erhöhung der Drehmomentgrenzen und keine Lockerung der0,35rad-Grenze.','',
       'Die JSON-Datei neben diesem Bericht enthält Phasen, Gelenkbeiträge, Sättigungsanteile, Beispiele und Quellhashes. Reproduktion: `python -B tools/analyze_skill_tracking.py --output reports/NEUER_NAME`.']
    # Human-readable spacing is intentional; do not inherit compact tool payload text.
    text='\n'.join(lines)+'\n'
    for a,b in [('bleibt0','bleibt 0'),('0,349766rad','0,349766 rad'),('etwa0','etwa 0'),('rad;005','rad; 005'),('bei0','bei 0'),('Bei96','Bei 96'),('seiner230über','seiner 230 über'),('mindestens99','mindestens 99'),('unveränderten5Nm','unveränderten 5 Nm'),('bei−','bei −'),('rad, die',' rad, die'),('bei+','bei +'),('während+5Nm','während +5 Nm'),('überlappende51','überlappende 51'),('der0,35rad','der 0,35 rad'),('Lauf005','Lauf 005'),('in005','in 005'),('005liegt','005 liegt')]:text=text.replace(a,b)
    mp.write_text(text);print(json.dumps({'json':str(jp),'markdown':str(mp),'runtime_changed':False}));return 0

if __name__=='__main__':raise SystemExit(main())
