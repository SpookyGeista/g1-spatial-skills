#!/usr/bin/env python3
"""Bounded actual-simulation negative tests using the existing operator channel.

No simulator, controller, pose publisher or actuator is launched. Test cases
remain separate: unavailable_pose must reach its own unavailable-source abort;
explicit_abort tests the operator abort during a measured active scan. Each
first sends stale takeover/release commands and verifies that they are consumed
without acceptance. A negative-test PASS always leaves the mission FAILED.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from punktnull.retained import EXPLORATION,load,prepare
prepare()
retained=load('_punktnull_fault_retained_replay',EXPLORATION/'replay_handover.py')
from tools.replay_mission_review import native_rows

KINDS=('unavailable_pose','explicit_abort')


def read(path):return json.loads(Path(path).read_text())


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(m):
    if (m.get('test')!='explore' or m.get('sim_only') is not True or m.get('real_actuators_enabled') is not False
        or m.get('live_human_tested') is not False or m.get('injection_kind') is not None):
        raise ValueError('Only a dedicated simulation exploration run without other injection is eligible')
    command=m.get('controller_command')
    if not isinstance(command,list) or len(command)<2 or command[1]!='lo':
        raise ValueError('Actual native launch must explicitly address loopback lo')


def listener_rows(contents,port=5556):
    result=[]
    for text in contents.splitlines()[1:]:
        fields=text.split()
        if len(fields)<10:raise ValueError('Incomplete proc TCP record')
        if fields[3]=='0A' and int(fields[1].rsplit(':',1)[1],16)==port:
            result.append({'local_address_hex':fields[1],'state_hex':fields[3],'inode':fields[9]})
    return result


def pose_listener_probe():
    sources={};listeners=[]
    for name in ['tcp','tcp6']:
        p=Path('/proc/net')/name
        if not p.exists():
            if name=='tcp':raise RuntimeError('No readable process TCP inventory')
            continue
        raw=p.read_text();sources[name]=hashlib.sha256(raw.encode()).hexdigest()
        listeners.extend(dict(table=name,**r) for r in listener_rows(raw))
    return {'monotonic':time.monotonic(),'method':'read-only /proc/net/tcp and tcp6 LISTEN port5556 inventory',
            'listeners':listeners,'listener_absent':not listeners,'inventory_sha256':sources,
            'limit':'Local network-namespace snapshot; historical absence is not retrospectively reprobed'}


def fault_physics(data,names,metadata,request,abort,kind):
    c={str(n):i for i,n in enumerate(names)}
    required=['sim_time','wall_monotonic','physics_step_index','support','tilt_deg']+[f'root_{i}' for i in range(13)]+[f'{k}_{i}' for k in ('q','dq','tau') for i in range(29)]
    core=data[:,[c[n] for n in required]]
    if len(data)<200 or not np.isfinite(core).all():raise ValueError('Missing or nonfinite full-rate physics state')
    sim=data[:,c['sim_time']];wall=data[:,c['wall_monotonic']];support=data[:,c['support']]
    root=data[:,[c[f'root_{i}'] for i in range(13)]];dq=data[:,[c[f'dq_{i}'] for i in range(29)]]
    speed=np.linalg.norm(root[:,7:10],axis=1)
    stable=(support==0)&(root[:,2]>=.70)&(root[:,2]<=.90)&(data[:,c['tilt_deg']]<=12)&(speed<=.13)&(np.max(abs(dq),axis=1)<=2.)
    q=root[:,3:7];yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
    def interval(a,b):
        indexes=np.flatnonzero((sim>=a-1e-6)&(sim<=b+1e-6))
        if len(indexes)<2:raise ValueError('Missing physical observation interval')
        selected=root[indexes]
        return {'samples':len(indexes),'duration_sim_s':float(sim[indexes[-1]]-sim[indexes[0]]),
            'stable_fraction':float(stable[indexes].mean()),'all_stable':bool(stable[indexes].all()),
            'max_linear_speed_m_s':float(speed[indexes].max()),'max_joint_speed_rad_s':float(abs(dq[indexes]).max()),
            'max_tilt_deg':float(data[indexes,c['tilt_deg']].max()),
            'yaw_change_rad':float(yaw[indexes[-1]]-yaw[indexes[0]]),
            'max_angular_speed_rad_s':float(np.linalg.norm(selected[:,10:13],axis=1).max()),
            'max_displacement_m':float(np.linalg.norm(selected[:,:3]-selected[0,:3],axis=1).max())}
    before=interval(request['sim_time']-.5,request['sim_time'])
    finish=interval(abort['sim_time']-.6,abort['sim_time'])
    released=np.flatnonzero(support==0);active=wall>=request['monotonic']
    step=float(np.linalg.norm(np.diff(root[:,:3],axis=0),axis=1).max())
    stop_delay=float(wall[-1]-abort['monotonic'])
    checks={'finite_contiguous_200hz_clock':bool(np.all(np.diff(wall)>0) and np.all(np.diff(data[:,c['physics_step_index']])==1)
            and np.max(abs(np.diff(sim)-float(metadata['physics_dt_s'])))<1e-6),
        'support_removed_never_reattached':bool(len(released) and np.all(support[released[0]:]==0) and np.all(support[active]==0)),
        'no_large_recorded_base_jump':step<.10,
        'active_scan_rotation_observed_before_injection':abs(before['yaw_change_rad'])>.01,
        'physics_stopped_within_025s_of_abort':-.03<=stop_delay<=.25 and metadata.get('end_reason')=='user_stop',
        'no_simulator_fault':metadata.get('error') in (None,'') and not any(e.get('name') in
            ('fall','nonfinite_state','motor_command_watchdog','supervisor_watchdog') for e in metadata.get('events',[]))}
    if kind=='unavailable_pose':checks['settled_before_unavailable_source_abort']=finish['samples']>=118 and finish['all_stable']
    return {'samples':len(data),'before_injection':before,'pre_abort_window':finish,
        'last_physics_minus_abort_wall_s':stop_delay,'max_base_step_m':step},checks


def audit(run,folder):
    run,folder=Path(run),Path(folder);m=read(run/'manifest.json');validate_manifest(m)
    plan=read(folder/'plan.json');injection=read(folder/'injection.json');kind=plan['fault']
    if kind not in KINDS:raise ValueError('Unknown fault plan')
    if m.get('outcome')=='RUNNING' or 'cleanup_errors' not in m:raise ValueError('Finalized mission required')
    events=m['events'];aborts=[e for e in events if e.get('event')=='transition' and e.get('state')=='ABORTED']
    if len(aborts)!=1:raise ValueError('Exactly one actual mission abort required')
    aborted=aborts[0];request=injection['fresh_command']
    unavailable=kind=='unavailable_pose'
    expected='POSE unavailable' if unavailable else 'Operator abort'
    accepted=[e for e in events if e.get('event')=='operator_command' and e.get('command') in ('takeover','release')]
    stale_interval=injection['stale_command_window']
    stale_accepted=[e for e in accepted if stale_interval['sent_monotonic']<=e['monotonic']<=stale_interval['confirmed_consumed_monotonic']]
    takeovers=[e for e in accepted if e.get('command')=='takeover']
    if unavailable:
        if len(takeovers)!=1:raise ValueError('One fresh takeover acceptance required')
        request=takeovers[0]
    native=[r for r in native_rows(run) if m['mission_started_monotonic']<=r[1]<=aborted['monotonic']]
    holding=[r for r in native if request['monotonic']<=r[1]<=aborted['monotonic']]
    meta=read(run/'sim/metadata.json')
    with np.load(run/'sim/trace.npz',allow_pickle=False) as z:physics,checks=fault_physics(z['data'],z['columns'],meta,request,aborted,kind)
    forbidden=[e for e in events if e.get('state') in ('TAKEOVER_PENDING','OPERATOR_CONTROL','RETURN_CHECK') or e.get('event')=='autonomy_resumed']
    sources_ok=bool(injection['pose_listener_before_stale']['listener_absent'] and injection['pose_listener_before_fresh']['listener_absent'])
    checks.update({'mission_remains_failed':m.get('outcome')=='FAILED' and expected in m.get('failure',''),
        'actual_abort_matches_injected_case':expected in aborted.get('detail','') and aborted['monotonic']>=request['monotonic'],
        'plan_written_before_injection':plan['declared_monotonic']<stale_interval['sent_monotonic']<request['monotonic'],
        'stale_takeover_release_consumed_without_acceptance':stale_interval['all_files_consumed'] is True and not stale_accepted
            and {c['command'] for c in stale_interval['commands']}=={'takeover','release'}
            and all(stale_interval['sent_monotonic']-c['monotonic']>=10 for c in stale_interval['commands']),
        'no_pose_listener_at_injection_snapshots':sources_ok,
        'native_never_transferred_ownership':len(native)>=20 and all(r[2:]==('planner_motion',0,1) for r in native),
        'no_operator_or_resume_phase':not forbidden,
        'cleanup_complete':m.get('cleanup_errors')==[] and read(folder/'stop_observation.json')['simulator_pid_absent'] is True,
        'no_live_or_success_claim':m.get('live_human_tested') is False and not any(e.get('event')=='mission_result' for e in events)})
    if unavailable:
        assistance=[e for e in events if e.get('state')=='ASSISTANCE_REQUESTED' and request['monotonic']<=e['monotonic']<aborted['monotonic']]
        checks['source_failure_after_planner_hold']=bool(assistance and len(holding)>=20 and all(r[2:]==('planner_motion',0,1) for r in holding))
    paths=[run/'manifest.json',run/'sim/metadata.json',run/'sim/trace.npz',folder/'plan.json',folder/'injection.json',folder/'stop_observation.json',
           *[run/'controller'/(f+'.csv') for f in ('motion_name','encoder_mode','motion_playing')]]
    return {'negative_test_pass':bool(all(checks.values())),'fault':kind,'mission_outcome':m['outcome'],'checks':checks,
        'physics':physics,'native_mission_samples':len(native),'native_hold_samples':len(holding),
        'actual_abort_event':aborted,'live_human_acceptance':False,'mission_completed':False,
        'evidence_kind':'Actual simulation negative case; never a successful mission or human recovery',
        'limitations':['TCP absence is a recorded network-namespace snapshot, not proof of all possible publishers.',
          'Continuous recorded physics and process teardown are bounded observations, not production-level safety.',
          'Unavailable-source abort and explicit operator abort are separate tests; the first does not prove the second.',
          'Stability gates constrain height/tilt/linear/joint speed; angular residual is reported separately.'],
        'source_sha256':{str(p.relative_to(run)):digest(p) for p in paths},'audit_source_sha256':digest(__file__)}


class FaultHarness:
    def __init__(self,run,kind):
        self.run=Path(run).resolve();self.kind=kind
        if not self.run.is_relative_to(ROOT/'runs') or self.run==ROOT/'runs':raise ValueError('Dedicated workspace runs/ directory required')
        m=read(self.run/'manifest.json');validate_manifest(m)
        if m['outcome']!='RUNNING':raise ValueError('Mission must already be running')
        self.folder=self.run/('fault_test_'+kind);self.folder.mkdir()
        self.plan={'fault':kind,'declared_monotonic':time.monotonic(),'expected_mission_outcome':'FAILED',
            'expected':'Missing POSE -> existing stable planner hold -> automatic unavailable-source abort' if kind=='unavailable_pose' else 'Explicit operator abort during active scan -> simulation teardown',
            'precondition':'simulation only, loopback native, active measured scan, no5556listener, no other injection',
            'stale_commands':'send takeover and release10s old; must be consumed without acceptance',
            'maximum_abort_to_last_physics_s':.25,'source_sha256':digest(__file__),'live_human':False}
        retained.runtime.atomic_json(self.folder/'plan.json',self.plan)
        (self.folder/'harness.py').write_bytes(Path(__file__).read_bytes())
        self.injected=False

    def record(self,event,**data):
        row={'event':event,'monotonic':time.monotonic(),**data}
        with (self.folder/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print('MISSION_FAULT '+event,flush=True)

    def snapshot(self):
        m=read(self.run/'manifest.json');validate_manifest(m)
        status=read(self.run/'status.json');state=read(self.run/'sim_state.json');now=time.monotonic()
        if not 0<=now-status['monotonic']<=.35:raise RuntimeError('Mission status stale')
        error=retained.mission_predicates.state_error(state,now)
        if error:raise RuntimeError(error)
        return m,status,state

    def wait_scan(self):
        deadline=time.monotonic()+150
        while time.monotonic()<deadline:
            m=read(self.run/'manifest.json');validate_manifest(m)
            status=read(self.run/'status.json')
            if m['outcome']!='RUNNING':raise RuntimeError('Mission ended before injection')
            starts=[e for e in m['events'] if e.get('event')=='scan_started']
            if status['state']=='SCANNING' and starts and time.monotonic()-starts[-1]['monotonic']>=1:
                return self.snapshot()
            time.sleep(.05)
        raise TimeoutError('No active scan before bounded timeout')

    def execute(self):
        result={'negative_test_pass':False,'fault':self.kind,'live_human_acceptance':False,'mission_completed':False}
        try:
            _,status,state=self.wait_scan();before=pose_listener_probe()
            if not before['listener_absent']:raise RuntimeError('POSE listener present; refusing unavailable-source test')
            stale_sent=time.monotonic();payloads=[];paths=[]
            for key in ['takeover','release']:
                payload={'command':key,'monotonic':stale_sent-10.01};payloads.append(payload)
                path=self.run/'commands'/(str(time.monotonic_ns())+'-fault-'+uuid.uuid4().hex+'.json');paths.append(path)
                retained.runtime.atomic_json(path,payload)
            deadline=time.monotonic()+1.5
            while time.monotonic()<deadline:
                m,status,state=self.snapshot()
                if m['outcome']!='RUNNING':raise RuntimeError('Stale commands unexpectedly ended mission')
                if not any(p.exists() for p in paths) and status['monotonic']>stale_sent+.25:break
                time.sleep(.025)
            else:raise TimeoutError('Stale commands were not consumed')
            consumed=time.monotonic()
            if any(e.get('event')=='operator_command' and e.get('command') in ('takeover','release') and e['monotonic']>=stale_sent for e in m['events']):
                raise RuntimeError('Stale takeover/release was accepted')
            if status['state']!='SCANNING':raise RuntimeError('Active scan ended before fresh fault injection')
            second=pose_listener_probe()
            if not second['listener_absent']:raise RuntimeError('POSE listener appeared; abort test preparation')
            command='takeover' if self.kind=='unavailable_pose' else 'abort'
            fresh={'command':command,'monotonic':time.monotonic(),'sim_time':state['sim_time'],'state':status['state']}
            injection={'pose_listener_before_stale':before,'pose_listener_before_fresh':second,
                'stale_command_window':{'sent_monotonic':stale_sent,'confirmed_consumed_monotonic':consumed,'all_files_consumed':True,'commands':payloads},
                'fresh_command':fresh,'observed_status':status,'measured_state_before':state}
            retained.runtime.atomic_json(self.folder/'injection.json',injection)
            retained.runtime.post_command(self.run,command);self.injected=True;self.record('fresh_fault_command',**fresh)
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                m=read(self.run/'manifest.json')
                if ('cleanup_errors' in m and ('native_map_saved' in m or 'map_save_error' in m)
                    and (self.run/'sim/trace.npz').is_file() and (self.run/'sim/metadata.json').is_file()):break
                time.sleep(.10)
            else:raise TimeoutError('Expected fault did not finalize evidence in45s')
            meta=read(self.run/'sim/metadata.json');pid=int(meta['pid'])
            stop={'monotonic':time.monotonic(),'simulator_pid':pid,'simulator_pid_absent':not Path('/proc',str(pid)).exists(),
                  'trace_sha256':digest(self.run/'sim/trace.npz'),'end_reason':meta.get('end_reason')}
            retained.runtime.atomic_json(self.folder/'stop_observation.json',stop)
            result=audit(self.run,self.folder)
        except Exception as exc:
            result['error']=f'{type(exc).__name__}: {exc}';self.record('fault_test_failed',reason=result['error'])
            if self.injected and read(self.run/'manifest.json').get('outcome')=='RUNNING':
                retained.runtime.post_command(self.run,'abort');self.record('fallback_abort')
        retained.runtime.atomic_json(self.folder/'summary.json',result)
        print(json.dumps(result,indent=2),flush=True)
        return 0 if result['negative_test_pass'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true');mode.add_argument('--evaluate',action='store_true')
    parser.add_argument('--fault',choices=KINDS,required=True);parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--output-name',default='summary_reaudit.json');args=parser.parse_args()
    if args.check:
        print(json.dumps({'fault':args.fault,'opens_sockets':False,'launches_processes':False,'sends_commands':False,
            'pose_port_inventory':pose_listener_probe(),'operator_channel_source':str(EXPLORATION/'replay_handover.py')},indent=2));return 0
    if args.run_dir is None:parser.error('--run-dir required')
    if args.execute:return FaultHarness(args.run_dir,args.fault).execute()
    folder=args.run_dir/('fault_test_'+args.fault)
    if Path(args.output_name).name!=args.output_name or not args.output_name.endswith('.json'):parser.error('Simple JSON output filename required')
    target=folder/args.output_name
    if target.exists():parser.error('Refusing to overwrite evidence')
    try:result=audit(args.run_dir,folder)
    except Exception as exc:result={'negative_test_pass':False,'fault':args.fault,'error':f'{type(exc).__name__}: {exc}','live_human_acceptance':False,'mission_completed':False}
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2));return 0 if result['negative_test_pass'] else 2


if __name__=='__main__':raise SystemExit(main())
