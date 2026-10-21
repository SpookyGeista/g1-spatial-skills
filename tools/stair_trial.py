#!/usr/bin/env python3
"""Bounded actual Isaac/SONIC two-riser component trial, simulation only.

No robot controller is added. The retained startup/stream/rejoin orchestration
is reused with explicit source-checked fixture/runtime substitutions.
"""
from pathlib import Path
import argparse,fcntl,hashlib,inspect,json,math,os,signal,sys,time,types,textwrap
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from punktnull.skills.legacy import OLD,WORKSPACE,LEGACY,load,planner_module
from punktnull.skills.reference import ReferenceClip,ReferencePoseStream,attach_reference_validation
from punktnull.skills.stairs import StairSession,generate,stable_landing,TOP
from punktnull.skills.session import ClearanceEvidence,measured_error
from tools.skill_trial import Trial as BaseTrial,atomic
CONTACTS=None


def contact_module(scene_module=None):
    """Retained read-only ContactSensor, filtered to ankle vs real stair boxes."""
    old=load('stair_contact_observer',LEGACY/'wall_contacts.py')
    old.DEFAULT_BODIES=('left_ankle_roll_link','right_ankle_roll_link');old.OPTIONAL_BODIES=()
    if scene_module is None:
        import punktnull.terrain.stairs as scene_module
    old.explicit_filters=lambda scene:tuple(b['prim'] for b in scene_module.collision_boxes())
    base=old.WallContactRecorder
    class StairContactRecorder(base):
        def __init__(self,*a,**k):
            global CONTACTS
            super().__init__(*a,**k);CONTACTS=self
        def save(self,output):
            report=super().save(output)
            report.update(scope='Actual filtered ankle normal-force vectors against configured raised-surface collider prims; no motor control',
                full_trial_provenance='../manifest.json',interpretation='Both ankles sustained upward landing normal force >10N is the measured support condition; no hardware claim.')
            Path(output,'wall_contacts.json').write_text(json.dumps(report,indent=2)+'\n');return report
    old.WallContactRecorder=StairContactRecorder
    return old


def runtime_module(*,support_height=TOP,landing_filter=1,landing_bounds=None):
    from punktnull.skills import runtime,guard
    class StairRuntimeView(runtime.RuntimeView):
        def __init__(self,*a,**k):
            super().__init__(*a,**k);self.landing_since=None;self.landing_samples=0;self.first_support_since=None
            self.foot_fk=None
            if landing_bounds is not None:
                from punktnull.terrain.kinematics import CollisionFK
                from punktnull.skills.limits import load_joint_limits
                from punktnull.skills.stairs import URDF
                self.foot_fk=CollisionFK(URDF,load_joint_limits().names)
                self.foot_fk.collisions=[g for g in self.foot_fk.collisions if g['link'].endswith('_ankle_roll_link')]
        def update(self,base,torso,q,dq,bodies,names,support,status,sim_time,step,tilt):
            now=time.monotonic();contact={}
            if CONTACTS and CONTACTS.times:
                stamp=CONTACTS.times[-1];forces=CONTACTS.forces[-1]
                # Sensor is post-step, pose is pre-step in retained loop. Maximum
                # allowed single-step offset is explicitly checked, not hidden.
                contact=dict(fresh=0<=now-stamp[0]<=.15 and abs(sim_time-stamp[1])<=.0101,
                    sensor_sim_time=stamp[1],pose_sim_time=float(sim_time),sensor_step=stamp[2],
                    upper_upward_normal_force_n=forces[:,landing_filter,2].astype(float).tolist(),
                    first_upward_normal_force_n=forces[:,0,2].astype(float).tolist(),
                    both_on_landing=bool(np.all(forces[:,landing_filter,2]>10.)),
                    basis='Real filtered PhysX ankle/landing normal forces, post-step')
            state=dict(support=bool(support),command_mode=status['mode'],base=base.tolist(),dq=dq.tolist(),
                       tilt_deg=float(tilt),stair_contacts=contact)
            first=(not support and contact.get('fresh',False) and contact.get('first_upward_normal_force_n',[0,0])[1]>10 and tilt<=12)
            if first:
                if self.first_support_since is None:self.first_support_since=sim_time
            else:self.first_support_since=None
            feet_inside=True;foot_bounds=None
            if self.foot_fk is not None:
                from scipy.spatial.transform import Rotation
                poses={}
                for side in ('left','right'):
                    name=side+'_ankle_roll_link';body=bodies[names.index(name)]
                    transform=np.eye(4)[None].copy();transform[0,:3,:3]=Rotation.from_quat(body[3:7][[1,2,3,0]]).as_matrix();transform[0,:3,3]=body[:3];poses[name]=transform
                b=self.foot_fk.bounds(poses)[0];lo=np.asarray(landing_bounds['aabb_min']);hi=np.asarray(landing_bounds['aabb_max'])
                feet_inside=bool(np.all(b[:,0,:2]>=lo[:2]+.02) and np.all(b[:,1,:2]<=hi[:2]-.02))
                foot_bounds=dict(minimum=b[:,0].min(axis=0).tolist(),maximum=b[:,1].max(axis=0).tolist(),required_horizontal_margin_m=.02)
            landing=stable_landing(state,support_height=support_height) and feet_inside
            if landing:
                if self.landing_since is None:self.landing_since=sim_time;self.landing_samples=0
                self.landing_samples+=1
            else:self.landing_since=None;self.landing_samples=0
            # Retain first-person/camera/runtime behavior; replace telemetry only
            # to add actual elevated support-aware stability.
            super().update(base,torso,q,dq,bodies,names,support,status,sim_time,step,tilt)
            if self.current and self.current.get('physics_step_index')==int(step):
                self.current.update(stair_contacts=contact,feet_inside_landing=feet_inside,foot_collision_bounds_world=foot_bounds,
                    first_riser_contact_seconds=0. if self.first_support_since is None else float(sim_time-self.first_support_since),
                    landing_stable_full_rate_seconds=0. if self.landing_since is None else float(sim_time-self.landing_since),
                    landing_stable_physics_samples=self.landing_samples,
                    applied_efforts=np.asarray(getattr(self,'current_applied_efforts',[])).tolist())
                runtime.atomic_json(self.run/'sim_state.json',self.current)
    module=types.ModuleType('stair_runtime_view');module.RuntimeView=StairRuntimeView;module.post_command=runtime.post_command
    return module


def sim_main(*,scene_module=None,support_height=TOP,landing_filter=1,landing_bounds=None):
    sys.path.insert(1,str(LEGACY));sys.path.insert(2,str(OLD))
    from punktnull.skills import guard
    if scene_module is None:
        import punktnull.terrain.stairs as scene_module
    sys.modules['ground_guard']=guard;sys.modules['runtime_view']=runtime_module(support_height=support_height,landing_filter=landing_filter,landing_bounds=landing_bounds)
    sys.modules['scene']=scene_module;sys.modules['wall_contacts']=contact_module(scene_module)
    path=LEGACY/'run_isaac.py';original=path.read_text();source=original
    replacements=[
        ('ground_guard.assess(now, base[:3], base[3:7], q, dq, support)',
         'ground_guard.assess(time.monotonic(), base[:3], base[3:7], q, dq, support, efforts=torque, state_monotonic=now, command_status=status)'),
        ('runtime_view.update(base, torso_state, q, dq, state, list(robot.body_names), support,',
         'runtime_view.current_applied_efforts = torque.copy(); runtime_view.update(base, torso_state, q, dq, state, list(robot.body_names), support,'),
        ("parser.add_argument('--render-every', type=int, default=10,","parser.add_argument('--render-every', type=int, default=20,"),
        ('if args.record and (not args.output or args.render_every != 10):','if args.record and (not args.output or args.render_every not in (10,20)):'),
        ("parser.error('Recording requires output and render-every 10 (20fps)')","parser.error('Recording requires output and render-every10 or20')")]
    for before,after in replacements:
        if source.count(before)!=1:raise RuntimeError('Retained staircase runtime hook changed: '+before)
        source=source.replace(before,after)
    before="            if not support and bridge and status['mode'] != 'active':"
    after="""            if not support and support_was_active and ground_guard:
                release_error=ground_guard.joint_limits.check(q,dq,None if torque is None else np.clip(torque,-limits,limits))
                if release_error:
                    meta['end_reason']='release_refused'
                    event('release_refused',reason=release_error)
                    break
"""+before
    if source.count(before)!=1:raise RuntimeError('Retained support release gate changed')
    source=source.replace(before,after)
    runner=types.ModuleType('stair_motor_loop');runner.__file__=str(path)
    exec(compile(source,str(path),'exec'),runner.__dict__);runner.HERE=WORKSPACE
    print('STAIR_RUNTIME_REUSE',hashlib.sha256(original.encode()).hexdigest(),flush=True);runner.main()


class StairTrial(BaseTrial):
    def __init__(self,args):
        from punktnull.terrain.stairs import FIXTURE_CONTRACT
        self.args=args;self.clip=ReferenceClip.load(args.reference)
        if json.loads((self.clip.source/'generated_reference.json').read_text())['skill']!='stairs':raise ValueError('Stair reference required')
        self.output=args.output.resolve()
        if self.output.exists() or not self.output.is_relative_to(WORKSPACE/'runs'):raise ValueError('NEW workspace/runs directory required')
        self.output.mkdir(parents=True);(self.output/'commands').mkdir()
        self.fixture=FIXTURE_CONTRACT;self.fixture_margin=None
        self.session=StairSession(self.clip);self.stream=ReferencePoseStream(self.clip)
        self.state=None;self.started=time.monotonic();self.phase='STARTUP';self.released=False
        self.sim=self.controller=self.client=None;self.token=None;self.next_packet=0.
        sources=[Path(__file__),WORKSPACE/'tools/skill_trial.py',*sorted((WORKSPACE/'punktnull/skills').glob('*.py')),WORKSPACE/'punktnull/terrain/stairs.py']
        self.manifest=dict(schema=1,sim_only=True,real_actuators_enabled=False,task='Actual bounded two-riser staircase; generated BONES/IK reference over existing native SONIC',
            outcome='RUNNING',live_human_tested=False,terrain='stairs',fixture_contract=self.fixture,
            localization='Measured simulator pose, isolated component test; no navigation integration or sensor terrain admission',
            autonomous_skill_admission=False,reference=self.clip.provenance(),events=[],native_reference_stream_validated=False,
            acceptance=dict(actual_joint_limits_every_physics_step=True,stable_landing_seconds=.7,landing_contact='Both ankles upward normal force on actual landing >10N',
                minimum_displacement_m=.50,planner_rejoin=True,tracking_error='diagnostic; actual limits remain mandatory'),
            source_sha256={str(p.relative_to(WORKSPACE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        for p in sources:
            dest=self.output/'source_snapshot'/p.relative_to(WORKSPACE);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        atomic(self.output/'manifest.json',self.manifest)
    def clearance(self):
        return None # Ground-corridor margin does not apply to a stair contact task.
    def authorize(self):
        auth=self.session.authorization(time.monotonic(),self.stream)
        auth['clearance_basis']='Declared static stair fixture; no8cm ground-corridor margin or sensor clearance claim'
        atomic(self.output/'skill_stream.json',auth)


def install_reused_run(cls=StairTrial,*,sim_tool='stair_trial.py',scene_module='punktnull.terrain.stairs',
                       detail='Two real risers120/144mm | native SONIC + declared BONES/IK reference',
                       outcome='STAIR_ASCENT_AND_PLANNER_RETURN_EXECUTED_PENDING_TRACE_EVALUATION'):
    source=textwrap.dedent(inspect.getsource(BaseTrial.run))
    replacements=[
        ("'punktnull.terrain.service_passage'",repr(scene_module)),
        ("str(WORKSPACE/'tools/skill_sim.py'),",f"str(WORKSPACE/'tools/{sim_tool}'),'--sim','--wall-contacts',"),
        ("self.wait(lambda:self.state.get('stable_full_rate_sim_seconds',0)>=.7,10)",
         "self.wait(lambda:self.state.get('landing_stable_full_rate_seconds',0)>=.7,10)"),
        ("'STREAMED_CRAWL_RETURN_EXECUTED_PENDING_TRACE_EVALUATION'",repr(outcome))]
    for before,after in replacements:
        if source.count(before)!=1:raise RuntimeError('Retained supervisor stair substitution changed: '+before)
        source=source.replace(before,after)
    namespace=dict(globals());exec(compile(source,str(Path(__file__))+'[retained-supervisor]','exec'),namespace)
    cls.run=namespace['run']
    tick=textwrap.dedent(inspect.getsource(BaseTrial.tick))
    before="f'Existing SONIC + {self.clip.seconds:.2f}s reference; experimental {self.args.terrain} execution'"
    after=repr(detail)
    if tick.count(before)!=1:raise RuntimeError('Retained skill status wording changed')
    tick=tick.replace(before,after)
    exec(compile(tick,str(Path(__file__))+'[retained-supervisor-status]','exec'),namespace)
    cls.tick=namespace['tick']


def evaluate(run,*,terrain_module=None,support_height=TOP,
             outcome='STAIR_ASCENT_AND_PLANNER_RETURN_EXECUTED_PENDING_TRACE_EVALUATION',
             first_prim='/World/ServiceStairs/FirstRiser',landing_prim='/World/ServiceStairs/Landing',
             minimum_rise=.20,minimum_displacement=.50):
    """Post-run raw state/contact/native evidence. No navigation input."""
    import csv
    from punktnull.skills.limits import load_joint_limits
    if terrain_module is None:
        import punktnull.terrain.stairs as terrain_module
    FIXTURE_CONTRACT=terrain_module.FIXTURE_CONTRACT;collision_boxes=terrain_module.collision_boxes
    run=Path(run);manifest=json.loads((run/'manifest.json').read_text());checks={}
    report=dict(schema=1,accepted=False,kind='Actual isolated PhysX/native two-riser ascent',checks=checks,
        live_human_tested=False,physical_robot_tested=False,
        limitations=['Prototype static120/144mm rises only; not arbitrary stairs or climbing.',
          'No sensor-derived terrain selection or integrated autonomous building mission.',
          'Normal forces cover both ankles vs step/landing, not whole-body contact or friction.',
          'Reference root positions are diagnostics; nativePOSE packets contain no root translation.'])
    paths=[run/'manifest.json',run/'sim/trace.npz',run/'sim/metadata.json',run/'sim/wall_contacts.npz',run/'submitted_reference.jsonl']
    missing=[str(p) for p in paths if not p.exists()]
    if missing:report['missing_evidence']=missing;return report
    with np.load(paths[1],allow_pickle=False) as z:a=z['data'];cols=list(z['columns'])
    get=lambda p,n:a[:,[cols.index(p+str(i)) for i in range(n)]]
    col=lambda key:a[:,cols.index(key)]
    meta=json.loads(paths[2].read_text());root=get('root_',13);support=col('support');released=support==0
    limits=load_joint_limits();errors=[limits.check(q,dq,tau) for q,dq,tau in zip(get('q_',29)[released],get('dq_',29)[released],get('tau_',29)[released])]
    checks['runtime_completed']=manifest['outcome']==outcome
    checks['actual_all_joint_limits']=bool(errors) and not any(errors)
    checks['finite_contiguous_physics']=bool(np.isfinite(a).all() and np.all(np.diff(col('physics_step_index'))==1) and np.all(abs(np.diff(col('sim_time'))-.005)<1e-7))
    checks['single_startup_support_release']=bool(np.count_nonzero(np.diff(support)==-1)==1 and not np.any(np.diff(support)>0))
    checks['no_root_reset_or_jump']=bool(released.any() and np.max(np.linalg.norm(np.diff(root[released,:3],axis=0),axis=1))<.03)
    checks['correct_collision_fixture']=meta.get('scene',{}).get('fixture_contract')==FIXTURE_CONTRACT and meta.get('scene',{}).get('collision_boxes')==collision_boxes()
    checks['no_runtime_guard_or_cleanup_failure']=not meta.get('error') and meta.get('end_reason') not in ('fall','command_watchdog','supervisor_watchdog') and not manifest.get('cleanup_errors')
    events={e['event']:e for e in manifest['events']};begin=events.get('skill_started');ready=events.get('measured_skill_return_ready');rejoin=events.get('native_planner_rejoined')
    checks['ordered_stream_landing_planner_events']=bool(begin and ready and rejoin and begin['monotonic']<ready['monotonic']<rejoin['monotonic'])
    with np.load(paths[3],allow_pickle=False) as z:
        times=z['times'];forces=z['normal_forces_world_n'];body=list(z['body_names']);prims=list(z['obstacle_prim_paths'])
    expected=['left_ankle_roll_link','right_ankle_roll_link'];landing=prims.index(landing_prim)
    checks['correct_force_sensor_scope']=body==expected
    first=prims.index(first_prim)
    firstmask=(forces[:,1,first,2]>10)&(times[:,3]==0)
    firstchunks=np.flatnonzero(np.diff(np.r_[False,firstmask,False])).reshape(-1,2)
    firstlong=max([(b-a-1)*.005 for a,b in firstchunks],default=0.)
    checks['actual_first_riser_support_010s']=bool(firstlong>=.10)
    forceok=np.all(forces[:, :, landing,2]>10,axis=1)&(times[:,3]==0)
    # Match each pre-step root sample to the force sample on that same physics
    # index; the two clocks differ by one known5ms step in the retained loop.
    index={int(x):i for i,x in enumerate(times[:,2])};mask=np.zeros(len(a),bool)
    for i,step in enumerate(col('physics_step_index')):
        j=index.get(int(step));mask[i]=j is not None and forceok[j]
    dq=get('dq_',29);stable=mask&released&(root[:,2]-support_height>=.70)&(root[:,2]-support_height<=.90)&(col('tilt_deg')<=12)&(np.linalg.norm(root[:,7:10],axis=1)<=.10)&(np.sqrt(np.mean(dq*dq,axis=1))<=.20)
    chunks=np.flatnonzero(np.diff(np.r_[False,stable,False])).reshape(-1,2)
    longest=max([(b-a-1)*.005 for a,b in chunks],default=0.)
    checks['measured_stable_two_foot_landing_070s']=bool(longest>=.70)
    checks['measured_rise_and_progress']=bool(ready and root[released,2].max()-root[released,2][0]>minimum_rise and ready.get('ground_displacement_m',0)>=minimum_displacement)
    tables={}
    for key in ('motion_name','motion_playing','encoder_mode'):
        path=run/'controller'/(key+'.csv');paths.append(path)
        tables[key]={int(r['index']):r for r in csv.DictReader(path.open())} if path.exists() else {}
    joined=sorted(set.intersection(*(set(t) for t in tables.values())));native=[]
    for i in joined:
        t=float(tables['motion_name'][i]['time_monotonic_ms'])/1000
        native.append((t,tables['motion_name'][i]['motion_name'],int(float(tables['encoder_mode'][i]['encoder_mode_0'])),float(tables['motion_playing'][i]['playing_0'])))
    from tools.native_reference_evidence import native_reference_phase
    stream=[r for r in native if r[1]=='streamed']
    report['native_reference_phase']=native_reference_phase(native,begin['monotonic'] if begin else None,ready['monotonic'] if ready else None)
    checks['actual_native_pose_v1_stream']=report['native_reference_phase']['accepted']
    checks['no_native_smpl_source']=not any(r[2]==2 for r in native)
    checks['actual_native_planner_rejoin']=bool(ready and rejoin and any(r[0]>ready['monotonic'] and r[1]=='planner_motion' and r[2]==0 and r[3]>.5 for r in native))
    submitted=[json.loads(x) for x in paths[4].read_text().splitlines()];frames=np.array([x['source_frame'] for x in submitted]);wire=np.array([x['wire_index'] for x in submitted])
    checks['no_reference_skips_or_reset']=bool(len(frames) and frames[0] in (0,1) and frames[-1]==manifest['reference']['frames']-1 and np.all((np.diff(frames)>=0)&(np.diff(frames)<=1)) and np.all(np.diff(wire)==1))
    report['measured']=dict(physics_samples=len(a),unsupported_samples=int(released.sum()),actual_limit_errors=sorted(set(e for e in errors if e)),
        root_start_end_m=root[released,:3][[0,-1]].tolist() if released.any() else None,
        longest_stable_landing_s=longest,longest_first_riser_support_s=firstlong,native_streamed_samples=len(stream),
        maximum_upward_landing_force_n=forces[:,:,landing,2].max(axis=0).tolist())
    report['sources_sha256']={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    report['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['native_phase_evaluator_sha256']=hashlib.sha256((ROOT/'tools/native_reference_evidence.py').read_bytes()).hexdigest()
    report['accepted']=bool(checks and all(checks.values()));return report


def main():
    if '--sim' in sys.argv:
        sys.argv.remove('--sim');return sim_main()
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--generate',type=Path);mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true')
    p.add_argument('--reference',type=Path,default=ROOT/'assets/stairs_s01_reference_003')
    p.add_argument('--output',type=Path);p.add_argument('--gui',action='store_true');p.add_argument('--record',action='store_true')
    p.add_argument('--max-seconds',type=float,default=200)
    a=p.parse_args();a.terrain='stairs'
    if a.generate:print(json.dumps(generate(a.generate).provenance(),indent=2));return 0
    clip=ReferenceClip.load(a.reference);install_reused_run()
    if a.check:print(json.dumps(dict(reference=clip.provenance(),simulator_started=False,physical_validation=False),indent=2));return 0
    if a.output is None:p.error('--execute requires NEW --output')
    if not 60<=a.max_seconds<=360:p.error('Bounded max-seconds60..360 required')
    os.environ.setdefault('PUNKTNULL_SKILL_CAMERA','side');os.environ.setdefault('PUNKTNULL_SKILL_CAMERA_EYE','0.8,-5.0,1.3')
    os.environ.setdefault('PUNKTNULL_SKILL_CAMERA_TARGET','0.8,0,0.65');os.environ.setdefault('PUNKTNULL_SKILL_CAMERA_WIDTH_M','4.5')
    with (ROOT/'.skill_trial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);trial=StairTrial(a)
        def interrupted(n,_):raise KeyboardInterrupt('Signal '+str(n))
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:trial.run()
        except (Exception,KeyboardInterrupt) as exc:
            trial.manifest['outcome']='FAILED';trial.manifest['error']=type(exc).__name__+': '+str(exc)
            trial.event('trial_failed',reason=trial.manifest['error'])
        finally:trial.close()
        try:atomic(trial.output/'evaluation.json',evaluate(trial.output))
        except Exception as exc:atomic(trial.output/'evaluation.json',dict(accepted=False,evaluation_error=repr(exc)))
        print(json.dumps(dict(output=str(trial.output),outcome=trial.manifest['outcome'],error=trial.manifest.get('error')),indent=2))
        return 2 if trial.manifest['outcome']=='FAILED' else 0
if __name__=='__main__':raise SystemExit(main())
