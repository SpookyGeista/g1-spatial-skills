#!/usr/bin/env python3
"""Bounded simulation-only native reference trial, preserving all failed runs.

--check is CPU/file-only. --execute launches existing Isaac/SONIC, exclusively
on loopback, in a NEW output directory. No real robot launcher is called.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from punktnull.skills.legacy import OLD,WORKSPACE,LEGACY,load,planner_module
from punktnull.skills.reference import ReferenceClip,ReferencePoseStream,attach_reference_validation
from punktnull.skills.limits import acceptance_contract
from punktnull.skills.session import REGISTRY,SkillSession,ClearanceEvidence,measured_error


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n');tmp.replace(path)


class Trial:
    def __init__(self,args):
        self.args=args;self.clip=ReferenceClip.load(args.reference or REGISTRY['crawl'].source)
        self.fixture=None;self.fixture_margin=.08
        if args.terrain=='passage':
            from punktnull.terrain.service_passage import FIXTURE_CONTRACT,collision_boxes
            report=json.loads(args.fixture_preflight.read_text())
            if args.reference is not None and (report.get('reference_source')!=str(self.clip.source) or report.get('reference_source_digest')!=self.clip.digest):
                raise ValueError('Generated crawl passage preflight must bind the selected reference source and digest')
            if report['fixture']!=FIXTURE_CONTRACT or report['collision_boxes']!=collision_boxes():
                raise ValueError('Preflight geometry differs from actual configured passage')
            for source,digest in report['sources_sha256'].items():
                if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Preflight source changed: '+source)
            datasets=report['datasets']
            if set(datasets)!={'reference','measured'} or any(not v['all_sampled_aabbs_clear'] for v in datasets.values()):
                raise ValueError('Missing/failed reference and measured body envelope preflight')
            self.fixture_margin=min(min(v['roof_vertical_margin_m'],v['minimum_side_margin_m']) for v in datasets.values())
            if self.fixture_margin<.08:raise ValueError('Geometric proposal has insufficient margin')
            self.fixture=FIXTURE_CONTRACT
        self.output=args.output.resolve()
        if not self.output.is_relative_to(WORKSPACE/'runs') or self.output.exists():
            raise ValueError('Use NEW output directory below workspace/runs')
        self.output.mkdir(parents=True)
        (self.output/'commands').mkdir()
        self.session=SkillSession(self.clip,experimental=True,allow_generated=args.reference is not None);self.stream=ReferencePoseStream(self.clip)
        self.state=None;self.started=time.monotonic();self.phase='STARTUP';self.released=False
        self.sim=self.controller=self.client=None;self.token=None;self.next_packet=0.
        self.manifest={'schema':1,'sim_only':True,'real_actuators_enabled':False,
            'task':'Experimental existing crawl reference over native protocol1; standing recovery and planner rejoin',
            'outcome':'RUNNING','live_human_tested':False,'localization':'ideal measured PhysX pose for skill evaluation',
            'clearance_source':'disclosed configured '+args.terrain+' fixture, not sensor-derived navigation',
            'terrain':args.terrain,'fixture_contract':self.fixture,
            'autonomous_skill_admission':False,
            'experimental_scope':'Isolated prospective actual-limits v2 diagnostic; no automatic building-mission admission',
            'generated_crawl_derivative_requested':args.reference is not None,
            'generated_reference':self.session.generated_reference,
            'experimental_ankle_roll_target_margin_rad':getattr(args,'ankle_roll_target_margin_rad',0.),
            'experimental_limit_ankle_pitch':getattr(args,'limit_ankle_pitch',False),
            'experimental_ankle_braking':getattr(args,'experimental_ankle_braking',False),
            'reference':self.clip.provenance(),'events':[],'native_reference_stream_validated':False,
            'source_sha256':{str(p.relative_to(WORKSPACE)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((WORKSPACE/'punktnull/skills').glob('*.py'))},
            'acceptance':{**acceptance_contract(),'ground_displacement_min_m':.5,'stable_recovery_seconds':.7,
                          'reference_tracking_acceptance':'diagnostic_only',
                          'completion_requires_native_planner_rejoin':True}}
        sources=[*sorted((WORKSPACE/'punktnull/skills').glob('*.py')),WORKSPACE/'tools/skill_trial.py',WORKSPACE/'tools/skill_sim.py']
        if self.fixture:
            sources += [WORKSPACE/'punktnull/terrain/service_passage.py',WORKSPACE/'punktnull/terrain/kinematics.py']
            atomic(self.output/'fixture_preflight.json',report)
        self.manifest['source_sha256']={str(p.relative_to(WORKSPACE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        for source in sources:
            dest=self.output/'source_snapshot'/source.relative_to(WORKSPACE)
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        atomic(self.output/'manifest.json',self.manifest)

    def event(self,name,**data):
        item={'event':name,'monotonic':time.monotonic(),'sim_time':self.state.get('sim_time') if self.state else None,**data}
        self.manifest['events'].append(item)
        with (self.output/'events.jsonl').open('a') as f:f.write(json.dumps(item,allow_nan=False)+'\n')
        atomic(self.output/'manifest.json',self.manifest)
        print('SKILL_'+name.upper()+' '+json.dumps(data),flush=True)

    def tick(self):
        now=time.monotonic()
        atomic(self.output/'status.json',{'monotonic':now,'state':self.phase,
            'owner':'reference skill' if self.token else 'native planner / supervisor',
            'detail':f'Existing SONIC + {self.clip.seconds:.2f}s reference; experimental {self.args.terrain} execution',
            'evidence_mode':'actual Isaac; prerecorded reference, no live human'})
        for label,child in [('Isaac',self.sim),('SONIC',self.controller)]:
            if child and child.process.poll() is not None:raise RuntimeError(label+' exited; retained logs explain failure')
        if self.sim and any(x in self.sim.text for x in ('ISAAC_FALL','ISAAC_ERROR','ISAAC_NONFINITE','ISAAC_MOTOR_COMMAND_WATCHDOG','ISAAC_SUPERVISOR_WATCHDOG')):
            raise RuntimeError('Simulator guard failed; inspect sim.log')
        if self.controller and ('Lost LowState' in self.controller.text or 'Safety check failed' in self.controller.text):
            raise RuntimeError('Native state/safety failure')
        try:self.state=json.loads((self.output/'sim_state.json').read_text())
        except (OSError,ValueError):pass
        if self.released:
            error=measured_error(self.state,now)
            if error:raise RuntimeError(error)
        if self.client and (self.client.status().get('fault') or not self.client.status()['worker_alive']):
            raise RuntimeError('Native transport failure: '+str(self.client.status()))
        if now-self.started>self.args.max_seconds:raise TimeoutError('Bounded trial deadline')
        if (self.output/'abort.request').exists():raise RuntimeError('Explicit abort.request')
        for command_path in sorted((self.output/'commands').glob('*.json')):
            command=json.loads(command_path.read_text());command_path.unlink()
            if command.get('command')=='abort':raise RuntimeError('Explicit operator abort command')
            raise RuntimeError('Only ABORT is supported during an isolated skill diagnostic')
        time.sleep(.005)

    def wait(self,predicate,seconds):
        deadline=time.monotonic()+seconds
        while not predicate():
            self.tick()
            if time.monotonic()>deadline:raise TimeoutError('Condition not observed before timeout')

    def marker(self,child,text,seconds=15):self.wait(lambda:text in child.text,seconds)

    def native(self,name):
        fb=self.state.get('native_feedback') if self.state else None
        return bool(fb and 0<=time.monotonic()-fb['monotonic']<=.15 and fb['motion']==name
                    and fb['playing'] and fb['encoder_mode']==0)

    def clearance(self):
        return ClearanceEvidence(time.monotonic(),'disclosed_simulation_fixture',True,True,self.fixture_margin,
                                 self.fixture['id'] if self.fixture else 'retained_floor_no_course')

    def authorize(self):
        atomic(self.output/'skill_stream.json',self.session.authorization(time.monotonic(),self.stream))

    def send(self,advance):
        now=time.monotonic()
        if now<self.next_packet:return
        packet=self.stream.packet(now,advance=advance)
        self.client.offer_pose(packet)
        self.next_packet=(self.next_packet+.02) if self.next_packet else (now+.02)
        with (self.output/'submitted_reference.jsonl').open('a') as f:
            f.write(json.dumps(self.stream.status(),allow_nan=False)+'\n')

    def run(self):
        sys.path[:0]=[str(OLD/'data/sonic_handover_diagnostics_2026-09-10'),str(OLD/'data/bones_seed_video_modules_2026-09-10'),str(LEGACY)]
        from run_replay_trial import ProcessLog
        from test_motion_tracking import reject_concurrent_processes,verify_existing_trt_caches
        from preflight import check_publishers
        reject_concurrent_processes();check_publishers()
        self.manifest['retained_cache_check']=verify_existing_trt_caches()
        launcher=load('skill_native_launcher',OLD/'data/bones_seed_walk_cover_2026-09-15/meta_teleop/launcher.py')
        env=launcher.runtime_environment();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',ROS_LOCALHOST_ONLY='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        simenv=os.environ.copy();simenv.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',ROS_LOCALHOST_ONLY='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
            CYCLONEDDS_URI='file://'+str(OLD/'data/sonic_handover_diagnostics_2026-09-10/dds_loopback.xml'))
        simenv['PUNKTNULL_ANKLE_ROLL_MARGIN_RAD']=str(getattr(self.args,'ankle_roll_target_margin_rad',0.))
        simenv['PUNKTNULL_LIMIT_ANKLE_PITCH']='1' if getattr(self.args,'limit_ankle_pitch',False) else '0'
        simenv['PUNKTNULL_ANKLE_BRAKING']='1' if getattr(self.args,'experimental_ankle_braking',False) else '0'
        if simenv.get('PUNKTNULL_SKILL_SCENE_MODULE'):
            raise ValueError('Select the explicit --terrain contract; environment scene overrides are forbidden')
        if self.fixture:simenv['PUNKTNULL_SKILL_SCENE_MODULE']='punktnull.terrain.service_passage'
        command=['/home/loya2/miniconda3/envs/unitree_sim_env/bin/python','-B',str(WORKSPACE/'tools/skill_sim.py'),
            '--device','cpu','--no-hall','--output',str(self.output/'sim'),
            '--ground-reference',str(self.clip.source),'--controller-logs',str(self.output/'controller')]
        if not self.fixture:command.append('--no-course')
        if not self.args.gui:command.append('--headless')
        if self.args.record:command+=['--record','--enable_cameras']
        self.manifest['simulator_command']=command
        self.sim=ProcessLog(command,self.output/'sim.log',simenv,OLD,terminal=False)
        self.marker(self.sim,'ISAAC_READY',110)
        self.client=planner_module().PlannerClient(simulation_only=True,max_speed=.30).start()
        attach_reference_validation(self.client)
        command=launcher.controller_command(OLD/'data/bones_seed_walk_cover_2026-09-15/pack/ready',17656)
        command[command.index('--input-type')+1]='zmq_manager'
        command+=['--planner-file','planner/target_vel/V2/planner_sonic.onnx','--planner-precision','32',
                  '--enable-csv-logs','--logs-dir',str(self.output/'controller'),
                  '--target-motion-logfile',str(self.output/'native_target_motion.csv')]
        self.manifest['controller_command']=command
        self.controller=ProcessLog(command,self.output/'controller.log',env,launcher.DEPLOY,terminal=True)
        self.marker(self.controller,'Init Done',90)
        self.sim.send(b'lower\n');self.marker(self.sim,'ISAAC_LOWERED',8)
        self.wait(lambda:self.state is not None,5)
        until=time.monotonic()+1;self.wait(lambda:time.monotonic()>=until,3)
        self.client.set_anchor(math.radians(self.state['yaw_deg']));self.client.start_control()
        self.marker(self.controller,'transitioning to CONTROL',12)
        self.wait(lambda:self.native('planner_motion'),12);self.client.acknowledge_started()
        self.sim.send(b'release\n');self.marker(self.sim,'ISAAC_RELEASED',5)
        self.wait(lambda:self.state and not self.state['support'],3);self.released=True
        self.phase='STABLE_ENTRY';self.client.hold(math.radians(self.state['yaw_deg']))
        self.wait(lambda:self.state and self.state.get('stable_full_rate_sim_seconds',0)>=.70,15)
        self.event('startup_support_removed',base=self.state['base'][:7])
        self.token=self.session.acquire('autonomy',self.state,self.clearance(),time.monotonic())
        self.authorize();self.client.request_pose();self.phase='PRIMING';self.event('skill_started',reference=self.clip.provenance())
        while self.session.phase!='RETURN_READY':
            self.tick();now=time.monotonic()
            last=self.state.get('ground_guard_last') or [-1,0,0,0,0]
            action=self.session.update(self.token,self.state,self.clearance(),now,
                native_stream_active=self.native('streamed'),submitted_frame=self.stream.source_frame,
                inferred_frame=last[0],tracking_error=last[2:],finished_sending=self.stream.finished_sending)
            self.phase=self.session.phase;self.authorize()
            if action['action']=='PAUSE_SIM_AND_STOP_NATIVE':raise RuntimeError(action['reason'])
            self.send(action['action']=='ADVANCE_REFERENCE')
        self.event('measured_skill_return_ready',ground_displacement_m=self.session.ground_displacement,
                   base=self.state['base'][:7],session_events=self.session.events)
        # Explicit harness ownership release; no claim of a human release.
        self.session.release(self.token,fresh_map_safe=True,explicit_release=True)
        self.token=None;self.authorize();self.phase='RETURN_CHECK'
        # Publish one final fresh endpoint before changing native owner.
        self.send(False)
        self.client.resume_planner(math.radians(self.state['yaw_deg']))
        self.wait(lambda:self.native('planner_motion'),12);self.client.acknowledge_started()
        self.client.hold(math.radians(self.state['yaw_deg']))
        until=time.monotonic()+1.;self.wait(lambda:time.monotonic()>=until,4)
        self.wait(lambda:self.state.get('stable_full_rate_sim_seconds',0)>=.7,10)
        self.event('native_planner_rejoined',base=self.state['base'][:7])
        self.manifest['outcome']='STREAMED_CRAWL_RETURN_EXECUTED_PENDING_TRACE_EVALUATION'
        self.manifest['native_reference_stream_validated']=True

    def close(self):
        # Stop physics first. Native STOP is not a stable posture hold.
        self.phase='STOPPING'
        if self.sim:
            try:self.sim.send(b'stop\n');self.sim.process.wait(timeout=20)
            except Exception as exc:self.manifest.setdefault('cleanup_errors',[]).append('sim stop: '+str(exc))
        if self.client:
            self.client.stop_control('isolated simulation ending');time.sleep(.05);self.client.close()
        for name,child in [('sim',self.sim),('controller',self.controller)]:
            if child:
                try:child.close()
                except Exception as exc:self.manifest.setdefault('cleanup_errors',[]).append(name+': '+str(exc))
        self.manifest['session_phase']=self.session.phase;self.manifest['stream']=self.stream.status()
        self.manifest['session_events']=self.session.events
        atomic(self.output/'manifest.json',self.manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--gui',action='store_true');p.add_argument('--record',action='store_true')
    p.add_argument('--max-seconds',type=float,default=200)
    p.add_argument('--terrain',choices=('flat','passage'),default='flat')
    p.add_argument('--fixture-preflight',type=Path)
    p.add_argument('--reference',type=Path,help='Explicit verified workspace crawl derivative for isolated trial; default retained clip unchanged')
    p.add_argument('--ankle-roll-target-margin-rad',type=float,default=0.,help='Experimental native PD ankle-roll target margin;0=off, maximum.08rad')
    p.add_argument('--limit-ankle-pitch',action='store_true',help='Also constrain native ankle-pitch desired positions using the explicit margin')
    p.add_argument('--experimental-ankle-braking',action='store_true',help='Explicit simulation-only end-stop assistance; requires pitch and roll target limiter; actual limits unchanged')
    args=p.parse_args()
    if not 60<=args.max_seconds<=360:p.error('max-seconds must be 60..360')
    if not 0<=args.ankle_roll_target_margin_rad<=.08:p.error('Ankle target margin must be0..08rad')
    if args.limit_ankle_pitch and args.ankle_roll_target_margin_rad==0:p.error('--limit-ankle-pitch requires positive margin')
    if args.experimental_ankle_braking and not(args.limit_ankle_pitch and args.ankle_roll_target_margin_rad>0):p.error('Experimental ankle braking requires positive margin and --limit-ankle-pitch')
    if args.terrain=='passage' and args.fixture_preflight is None:p.error('Passage diagnostic needs --fixture-preflight')
    clip=ReferenceClip.load(args.reference or REGISTRY['crawl'].source)
    session=SkillSession(clip,experimental=True,allow_generated=args.reference is not None)
    if args.check:
        print(json.dumps({'reference':clip.provenance(),'controller_started':False,
                          'trial_scope':args.terrain+' experimental component test; no autonomous admission',
                          'acceptance_contract':acceptance_contract(),'generated_crawl_derivative_requested':args.reference is not None},indent=2));return 0
    if args.output is None:p.error('--execute needs --output NEW_WORKSPACE_RUN')
    lockpath=WORKSPACE/'.skill_trial.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        trial=Trial(args)
        def interrupted(number,_):raise KeyboardInterrupt('Signal '+str(number))
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:trial.run()
        except (Exception,KeyboardInterrupt) as exc:
            trial.manifest['outcome']='FAILED';trial.manifest['error']=type(exc).__name__+': '+str(exc)
            trial.event('trial_failed',reason=trial.manifest['error'])
        finally:trial.close()
        print(json.dumps({'output':str(trial.output),'outcome':trial.manifest['outcome'],'error':trial.manifest.get('error')},indent=2))
        return 2 if trial.manifest['outcome']=='FAILED' else 0

if __name__=='__main__':raise SystemExit(main())
