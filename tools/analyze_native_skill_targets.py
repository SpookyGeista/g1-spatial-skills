#!/usr/bin/env python3
"""Read-only CPU diagnosis; native reference != policy PD target; no gate changes."""
from pathlib import Path
import argparse, ast, csv, hashlib, json, operator, re, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from punktnull.skills.legacy import OLD, WORKSPACE, LEGACY, load, ground_module
from punktnull.skills.session import REGISTRY
ARCHIVE=OLD/'data/teleop_archives/g1_pose_2026-09-09/gear_sonic_deploy'
NATIVE=ARCHIVE/'src/g1/g1_deploy_onnx_ref'
SOURCE=NATIVE/'src/g1_deploy_onnx_ref.cpp'
HEADER=NATIVE/'include/policy_parameters.hpp'
LOGGER=NATIVE/'src/state_logger.cpp'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scalar(expr, env):
    """Only numeric literals/names and arithmetic in audited C++ constants."""
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
    def visit(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
        if isinstance(n,ast.Name): return env[n.id]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub): return -visit(n.operand)
        if isinstance(n,ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](visit(n.left),visit(n.right))
        raise ValueError('Unsupported header expression '+expr)
    return visit(ast.parse(expr.strip(),mode='eval').body)
def constants():
    text=re.sub(r'//[^\n]*','',HEADER.read_text());env={}
    for name,expr in re.findall(r'const double (\w+)\s*=\s*([^;]+);',text):env[name]=scalar(expr,env)
    result={}
    for name in ['isaaclab_to_mujoco','mujoco_to_isaaclab','default_angles','g1_action_scale','kps','kds']:
        body=re.search(r'\b'+name+r'\s*=\s*\{(.*?)\}',text,re.S).group(1)
        result[name]=np.array([scalar(v,env) for v in body.split(',') if v.strip()])
        assert result[name].shape==(29,)
    assert np.array_equal(result['mujoco_to_isaaclab'][result['isaaclab_to_mujoco'].astype(int)],np.arange(29))
    return result

def table(path):
    with Path(path).open() as f:
        r=csv.DictReader(f);cols=r.fieldnames; rows=list(r)
    return cols,rows

def numbers(path):
    cols,rows=table(path);return cols,np.array([[float(r[k]) for k in cols] for r in rows])

def stat(values):
    v=np.asarray(values);return {'n':int(v.size),'min_median_p95_max':np.quantile(v,[0,.5,.95,1]).tolist()} if v.size else {'n':0}

def analyze(run):
    run=Path(run).resolve(); manifest=json.loads((run/'manifest.json').read_text());cfg=constants()
    assert Path(manifest['controller_command'][0])==ARCHIVE/'target/release/g1_deploy_onnx_ref'
    ref=ground_module().Reference(REGISTRY['crawl'].source);names=load('native_audit_bridge',LEGACY/'bridge.py').BODY_JOINT_NAMES
    qcols,qraw=numbers(run/'controller/q.csv');q=qraw[:,5:];t=qraw[:,3]/1000;n=len(q)
    others={key:numbers(run/f'controller/{key}.csv')[1] for key in ['action','dq','motor_torque','encoder_mode','motion_playing']}
    _,motion=table(run/'controller/motion_name.csv')
    target=np.loadtxt(run/'native_target_motion.csv',delimiter=',',usecols=range(36),ndmin=2);targetq=target[:,7:]
    assert n==len(target)==len(motion)
    assert np.array_equal(qraw[:,0],np.arange(n))
    for a in others.values(): assert a.shape[0]==n and np.array_equal(a[:,:5],qraw[:,:5])
    assert all(int(row['index'])==i and abs(float(row['time_monotonic_ms'])-qraw[i,3])<1e-6 for i,row in enumerate(motion))
    log=(run/'controller.log').read_text();badlog=[x for x in log.splitlines() if re.search(r'Error:|\[StateLogger ERROR\]|Failed to log token',x)]
    assert not badlog, badlog[:5]
    stream=np.array([x['motion_name']=='streamed' for x in motion]); assert np.all(others['encoder_mode'][stream,5]==0)
    # target row n is the native motion reference; action row n+1 stores policy output computed in tick n.
    perm=cfg['isaaclab_to_mujoco'].astype(int)
    priorpolicy=others['action'][:,5:][:,perm]*cfg['g1_action_scale']+cfg['default_angles']
    policy=np.full_like(priorpolicy,np.nan);policy[:-1]=priorpolicy[1:]
    z=np.load(run/'sim/trace.npz');a=z['data'];cols=list(z['columns']);get=lambda k:a[:,cols.index(k)]
    qt=a[:,[cols.index('q_'+str(i)) for i in range(29)]];tt=get('wall_monotonic')
    ti=[]; state_residual=[]
    for stamp,qi in zip(t,q):
        lo=min(len(tt)-1,max(0,np.searchsorted(tt,stamp-.15)));hi=max(lo+1,min(len(tt),np.searchsorted(tt,stamp+.01)))
        errors=np.max(abs(qt[lo:hi]-qi),axis=1);best=np.flatnonzero(errors<=errors.min()+1e-10)
        # If identical poses repeat, use the closest wall-time sample, never fit against a target motion.
        k=int(best[np.argmin(abs(tt[lo+best]-stamp))])+lo;ti.append(k);state_residual.append(errors[k-lo])
    ti=np.array(ti);state_residual=np.array(state_residual);exact=state_residual<1e-7
    gf=get('guard_reference_frame')[ti].astype(int);ground=(get('guard_ground_permission')[ti]>.5)&(gf>=0)
    native_frame=np.full(n,-1,dtype=int);native_residual=np.full(n,np.nan);native_ambiguous=np.zeros(n,dtype=bool)
    for row in np.flatnonzero(stream):
        dif=np.max(abs(ref.joints-targetq[row]),axis=1);k=int(dif.argmin());native_frame[row]=k;native_residual[row]=dif[k];native_ambiguous[row]=np.sum(dif<1e-5)>1
    matched=stream&(native_residual<1e-5)
    # Sources are checked, never inferred from measured-state closeness. Ambiguous endpoint poses are disclosed.
    selected=ground&stream&exact
    error=q-targetq;rms=np.sqrt(np.mean(error**2,axis=1));bad=selected&(rms>.35)
    submitted=[json.loads(x) for x in (run/'submitted_reference.jsonl').read_text().splitlines()]
    st=np.array([x['monotonic'] for x in submitted]);sf=np.array([x['source_frame'] for x in submitted]);si=np.searchsorted(st,t,side='right')-1
    ok=matched&(si>=0)&~native_ambiguous
    lag=sf[np.maximum(si,0)]-native_frame
    report={'run':str(run),'scope':'Diagnostic only; unchanged official evaluator and gates',
      'alignment':{'target_rows':len(target),'state_rows':n,'all_state_metadata_indices_contiguous':True,'equal_counts_post_state_and_target':True,
        'ordinal_basis':'Audited control loop has one state write before one target write; no target write without state, equal counts and contiguous post-state prove no unmatched iteration under this source. No standalone timestamp in target file.',
        'native_binary_source_provenance_limit':'Launched archived binary and adjacent source hashed; no reproducible build receipt links binary to source. CSV transform cross-checked against actual trace.',
        'source_stage_errors':badlog,'terminal_lowstate_lost_messages':log.count('Lost LowState data connection from robot!'),
        'state_to_trace_exact_fraction':float(exact.mean()),'native_rows_after_last_trace_wall':int((t>tt[-1]).sum()),'shutdown_note':'Native logger continues on stale last LowState until its 0.5 s watchdog; no new physical state is inferred.','state_to_trace_max_abs_joint_residual_rad':stat(state_residual),'state_age_native_log_minus_trace_wall_s':stat((t-tt[ti])[exact]),
        'q_order':'absolute SDK29 (CSV serialization reverses internal IsaacLab/default offset)',
        'action_order':'raw IsaacLab29; scale/remap/default required; row n+1 is output computed in native tick n',
        'motor_torque_order':'SDK29 tau_est from simulator last_torque (previous physics step applied command), not a force/contact sensor',
        'action_delivery_limit':'Policy target reconstructed from output; low-command DDS receive/write latency is not logged per command. Not an exact force decomposition.'},
      'native_reference':{'stream_rows':int(stream.sum()),'source_matches_at_1e_5_rad':int(matched.sum()),'source_unique_matches':int((matched&~native_ambiguous).sum()),
        'max_abs_reference_match_residual_rad':stat(native_residual[stream]),'content_match_source_frame_range':[int(native_frame[matched].min()),int(native_frame[matched].max())],
        'matched_frame_step_differences':{str(int(v)):int(k) for v,k in zip(*np.unique(np.diff(native_frame[matched]),return_counts=True))},
        'post_send_cursor_minus_content_frame':stat(lag[ok]),'cursor_limit':'submitted status is post-increment (typically one beyond packet start); monotonic sample before target write; no synchronized packet receipt timestamps',
        'native_frame_minus_guard_inferred_frame':stat((native_frame-gf)[selected&matched])},
      'diagnostic_ground_rows':int(selected.sum()),'native_current_reference_joint_rms_p95_rad':float(np.percentile(rms[selected],95)),
      'above_035_native_rows':int(bad.sum()),'wrist_bad_rows':[], 'phases':[]}
    limits=np.asarray(json.loads((run/'sim/metadata.json').read_text())['effort_limits_nm']);tau=others['motor_torque'][:,5:]
    trace_tau=a[:,[cols.index('tau_'+str(i)) for i in range(29)]]
    active=t<=tt[-1]
    report['alignment']['active_state_age_s']=stat((t-tt[ti])[active])
    report['alignment']['lowstate_torque_vs_previous_trace_applied_max_abs_nm']=stat(np.max(abs(tau[active]-trace_tau[np.maximum(ti[active]-1,0)]),axis=1))
    report['native_reference']['all_source_frames_selected_in_order_without_skip']=bool(np.array_equal(np.unique(native_frame[matched]),np.arange(len(ref.joints))) and np.all(np.isin(np.diff(native_frame[matched]),[0,1])))
    for j in [20,27,21,28]:
        b=bad&np.isfinite(policy[:,j]);sign=np.sign(targetq[:,j]-q[:,j]);push=sign*(policy[:,j]-q[:,j])
        report['wrist_bad_rows'].append({'joint':names[j],'sdk_index':j,'limit_nm':float(limits[j]),'rows':int(b.sum()),
          'reference_rad':stat(targetq[b,j]),'measured_rad':stat(q[b,j]),'current_policy_PD_target_rad':stat(policy[b,j]),
          'policy_minus_reference_rad':stat(policy[b,j]-targetq[b,j]),'abs_measured_minus_reference_rad':stat(abs(error[b,j])),
          'abs_measured_minus_policy_target_rad':stat(abs(q[b,j]-policy[b,j])),
          'policy_target_pushes_toward_reference_fraction':float(np.mean(push[b]>0)),
          'native_previous_applied_torque_at_99pct_limit_fraction':float(np.mean(abs(tau[b,j])>=.99*limits[j])),
          'native_previous_applied_torque_toward_reference_fraction':float(np.mean(sign[b]*tau[b,j]>0))})
    timeline=json.loads((ref.source/'build_metadata.json').read_text())['timeline']
    for phase in timeline:
        m=selected&matched&(native_frame>=phase['start_frame'])&(native_frame<phase['end_frame_exclusive'])
        report['phases'].append({'id':phase['id'],'native_rows':int(m.sum()),'p95_joint_rms_rad':float(np.percentile(rms[m],95)) if m.any() else None,'above_035_rows':int((m&bad).sum())})
    peak=np.flatnonzero(bad)[np.argsort(rms[bad])[-3:][::-1]] if bad.any() else []
    report['peak_examples']=[{'native_row':int(i),'native_monotonic_s':float(t[i]),'trace_row':int(ti[i]),'trace_sim_s':float(get('sim_time')[ti[i]]),'source_frame_from_content':int(native_frame[i]),'source_max_abs_residual':float(native_residual[i]),'joint_rms_rad':float(rms[i]),'left_right_wrist_pitch':{'reference':targetq[i,[20,27]].tolist(),'measured':q[i,[20,27]].tolist(),'current_policy_PD_target':policy[i,[20,27]].tolist(),'native_previous_applied_torque_nm':tau[i,[20,27]].tolist()}} for i in peak]
    paths=[run/f for f in ['manifest.json','native_target_motion.csv','controller/q.csv','controller/action.csv','controller/dq.csv','controller/motor_torque.csv','controller/motion_name.csv','controller/encoder_mode.csv','controller/motion_playing.csv','submitted_reference.jsonl','sim/trace.npz','sim/metadata.json','controller.log']]
    paths += [ref.source/'joint_pos.csv',ref.source/'build_metadata.json',SOURCE,HEADER,LOGGER,Path(manifest['controller_command'][0])]
    report['source_sha256']={str(p):sha(p) for p in paths}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,default=[WORKSPACE/'runs/skills/006_passage_diagnostic',WORKSPACE/'runs/skills/007_passage_10fps']);p.add_argument('--output',type=Path,default=WORKSPACE/'reports/native_skill_targets_006_007');args=p.parse_args()
    jp=args.output.with_suffix('.json');mp=args.output.with_suffix('.md')
    if jp.exists() or mp.exists():p.error('Use a new output path; retained reports will not be overwritten')
    reports=[analyze(x) for x in args.runs]
    out={'scope':'CPU-only source and retained native-log analysis; no simulator, controller, runtime, model or gate changes','source_anchor':{
      'Control_state_read_and_early_returns':str(SOURCE)+':3840','Control_reference_snapshot':str(SOURCE)+':3914','policy_output':str(SOURCE)+':3118','native_reference_write':str(SOURCE)+':4005',
      'logger_wall_and_monotonic_clock':str(LOGGER)+':106','CSV_absolute_SDK_remap':str(LOGGER)+':294','SDK_to_policy_permutation':str(HEADER)+':100'},
      'limits':['Target lines have no timestamp; ordinal alignment depends on audited source and complete single-session logs. Adjacent archived source has no build receipt.',
      'Content matching identifies native reference joints, not commanded root motion. No actual base pose is inferred from target XYZ.',
      'Current-frame native target and policy motor target differ by design: native encoder also uses future reference observations.',
      'Motor torque is the simulator commanded effort; contact force, external load and exact per-command DDS delivery are not recorded. Saturation correlation does not identify contact as cause.',
      '50 Hz native diagnostic rows do not replace unchanged 200 Hz guard/evaluator or relax acceptance.'], 'runs':reports}
    jp.parent.mkdir(parents=True,exist_ok=True);jp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    lines=['# Native Referenz, Policy-Sollwert und gemessene Handgelenke','',
    'CPU-Diagnose der gespeicherten Versuche 006 und 007. Die ursprünglichen Abnahmewerte und Fehler bleiben unverändert. `native_target_motion.csv` enthält die vom nativen Controller gelesene Bewegungsreferenz, **nicht** den Motor-Sollwinkel der Policy.','',
    '## Zuordnung','',
    'Die passende archivierte C++-Quelle schreibt pro erfolgreichem Kontrollzyklus zuerst den Zustand, dann den Bewegungsnamen/Encoder, berechnet die Policy und schreibt genau eine Target-Zeile. Alle CSV-Indizes und Zeitstempel sowie die Target-Zeilenzahlen sind hier vollständig gleich. Zwischenfehler sind nicht protokolliert; der spätere LowState-Verlust erfolgt vor einem neuen State-Write beim Herunterfahren. Damit ist die zeilenweise Zuordnung unter dieser Implementierung begründet, nicht allein aus gleicher Zeilenzahl geraten. Ein reproduzierbarer Build-Nachweis zwischen Binärdatei und danebenliegender Quelle fehlt; beide sind gehasht.','',
    '`q.csv` rechnet die interne IsaacLab-Reihenfolge zurück in absolute SDK-Winkel. `action.csv` enthält rohe Policy-Ausgaben in IsaacLab-Reihenfolge, vom vorigen Tick. Nach Umordnung, Aktionsskalierung und Default-Offset liefert Zeile n+1 den in Tick n berechneten Motor-Positionssollwert. Die letzte Ausgabe kann daher nicht rekonstruiert werden.','',
    '| Lauf | Native Zeilen | Zustand identisch mit tatsächlichem Physiksample | Stream-Zeilen / Referenzmatches | Native Ground-P95¹ |','|---|---:|---:|---:|---:|']
    for r in reports:
        al=r['alignment'];nr=r['native_reference'];lines.append(f"| {Path(r['run']).name} | {al['state_rows']} | {100*al['state_to_trace_exact_fraction']:.2f}% | {nr['stream_rows']} / {nr['source_matches_at_1e_5_rad']} | {r['native_current_reference_joint_rms_p95_rad']:.6f} rad |")
    lines+=['','¹Diagnose an tatsächlich nativen 50-Hz-Zeilen und aktuellem Target; kein Ersatz für die ursprüngliche Physik-Abnahme. Quellframes werden ausschließlich am Referenzinhalt zugeordnet, niemals durch Optimierung gegen die gemessene Roboterhaltung. Wiederholte/gleiche Endpunkt-Posen sind entsprechend mehrdeutig.','',
    '## Handgelenke bei nativen Zeilen über 0,35 rad Gelenk-RMS','',
    '| Lauf / Gelenk | Policy drückt Richtung Referenz | Vorheriges Effort ≥99% Limit | Median Soll − Referenz | Median Ist − Referenz (absolut) |','|---|---:|---:|---:|---:|']
    for r in reports:
        for w in r['wrist_bad_rows'][:2]:lines.append(f"| {Path(r['run']).name[:3]} / {w['joint']} | {100*w['policy_target_pushes_toward_reference_fraction']:.1f}% | {100*w['native_previous_applied_torque_at_99pct_limit_fraction']:.1f}% | {w['policy_minus_reference_rad']['min_median_p95_max'][1]:+.3f} rad | {w['abs_measured_minus_reference_rad']['min_median_p95_max'][1]:.3f} rad |")
    lines+=['','Die JSON-Datei enthält alle Phasen, konkrete Peak-Zeilen, rekonstruierten Policy-Sollwinkel, Zustandsalter, Referenz-Abstände und SHA-256 aller Eingaben. Die simulierten `tau_est`-Werte stammen aus dem zuletzt angewandten Motor-Effort; sie sind **keine gemessenen Kontaktkräfte**. Eine festgestellte Sättigung mit großer Istabweichung belegt deshalb keine Ursache durch Bodenlast. Auch DDS-Zustellung und Physik-Anwendung jedes einzelnen Policy-Kommandos sind nicht separat zeitgestempelt. Die Policy verarbeitet außerdem zukünftige Referenzbeobachtungen; ihr motorischer Sollwert muss nicht mit dem aktuellen Referenzwinkel übereinstimmen.','',
    '## Befund und Grenze der Aussage','',
    'In beiden Läufen wählt der native Controller sämtliche 1.024 vorhandenen Referenzframes in Reihenfolge aus: 1.023 Fortschritte um genau einen Frame, sonst nur Wiederholungen. Alle nativen Stream-Zeilen stimmen bis auf höchstens 0,000005 rad mit der gelieferten Quelle überein. Das ist ein Nachweis der tatsächlich ausgewählten Referenzfolge, kein Beleg erfolgreichen körperlichen Trackings. Der gespeicherte Guard-Frame weicht zeitweise davon ab; die große Handgelenkabweichung bleibt auch gegenüber dem echten nativen Target bestehen.','',
    'Bei Lauf 007, nativem Frame 76, liegt der linke Wrist-Pitch-Istwert bei −1,150 rad, die native Referenz bei +0,095 rad und der rekonstruierte Policy-Sollwert bei +0,278 rad; das vorherige angewandte Effort beträgt +5 Nm. Rechts stehen +1,214 rad Ist gegen +0,142 rad Referenz und +0,208 rad Policy-Soll bei −5 Nm. Die erheblichen Abweichungen in dieser Zeile entstehen somit nicht dadurch, dass der motorische Sollwert dieselben falschen Extremwinkel fordert. In Lauf 006 zeigt der Aufsteh-Peak ebenfalls rechts −1,148 rad Ist gegen +0,061 rad Referenz und +0,377 rad Policy-Soll bei +5 Nm.','',
    'Die Befunde stützen eine tatsächliche Reaktions-/Trackinggrenze mit Effort-Sättigung, nicht einen falsch übertragenen Referenzwinkel. Sie trennen jedoch äußere Belastung, Kontaktgeometrie, Dynamik und Policy-Verhalten nicht kausal. Die Policy-Sollwerte weichen selbst von der aktuellen Referenz ab und verarbeiten Lookahead; ein idealer Regler ist damit nicht nachgewiesen. Nächster sinnvoller Diagnosekanal sind tatsächliche Hand-/Unterarmkontaktkräfte zusammen mit angewandtem Motorbefehl und Empfangszeit, ohne Limits zu erhöhen.','',
    '006 bleibt wegen Timing durchgefallen; 007 bleibt wegen Tracking durchgefallen. Diese Auswertung ändert keine Zulassung. Jeweils 25 späte native Logzeilen wiederholen nach Physikende den letzten LowState bis zum 0,5-s-Watchdog. Sie sind keine weiteren Simulationsschritte und fallen nicht in die untersuchte Ground-Phase.','',
    'Reproduktion: `python -B tools/analyze_native_skill_targets.py --output reports/NEUER_NAME`. Keine Änderungen an Bewegungen, Reglern, Limits, Simulator oder Abnahme.']
    mp.write_text('\n'.join(lines)+'\n');print(json.dumps({'json':str(jp),'markdown':str(mp),'runs':[{'run':r['run'],'alignment':r['alignment'],'native_reference':r['native_reference'],'p95':r['native_current_reference_joint_rms_p95_rad'],'wrist':r['wrist_bad_rows'][:2],'peaks':r['peak_examples']} for r in reports]},indent=2))
if __name__=='__main__':main()
