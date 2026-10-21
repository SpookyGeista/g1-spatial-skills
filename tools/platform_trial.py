#!/usr/bin/env python3
"""Actual isolated120mm service-platform step-up; not general staircase ability."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from punktnull.skills.reference import ReferenceClip
from punktnull.skills.platform import PlatformSession,generate,TOP
from punktnull.terrain import platform as terrain
from tools.stair_trial import StairTrial,install_reused_run,sim_main,evaluate,atomic
OUTCOME='PLATFORM_120MM_ASCENT_AND_PLANNER_RETURN_EXECUTED_PENDING_TRACE_EVALUATION'

class PlatformTrial(StairTrial):
    def __init__(self,args):
        super().__init__(args)
        self.fixture=terrain.FIXTURE_CONTRACT;self.session=PlatformSession(self.clip)
        self.manifest.update(task='One physical120mm service-platform step-up, stable two-foot top support and native planner return',
            terrain='platform120',fixture_contract=self.fixture,autonomous_skill_admission=False,
            scope='Isolated bounded service-platform climb; not general stairs or live-human validation')
        self.manifest['acceptance'].update(minimum_displacement_m=.30,minimum_root_rise_m=.08,
            fixed_height_m=.120,entire_foot_horizontal_margin_m=.02,contract_id='platform120_actual_v2')
        self.manifest['start_variation']=dict(y_m=args.start_y,yaw_deg=args.start_yaw)
        for p in (Path(__file__),ROOT/'punktnull/terrain/platform.py'):
            key=str(p.relative_to(ROOT));self.manifest['source_sha256'][key]=hashlib.sha256(p.read_bytes()).hexdigest()
            dest=self.output/'source_snapshot'/key;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        atomic(self.output/'manifest.json',self.manifest)

def install():
    install_reused_run(PlatformTrial,sim_tool='platform_trial.py',scene_module='punktnull.terrain.platform',
        detail='One real120mm platform rise | native SONIC + declared BONES/IK reference',outcome=OUTCOME)
    # Keep the actual retained bounded spawn-variation arguments. They affect
    # only the new run's startup, never an executing robot or scene.
    import inspect,textwrap
    from tools.skill_trial import Trial as Original
    source=textwrap.dedent(inspect.getsource(Original.run))
    # Reuse the already substituted function source from its code semantics
    # by performing the same exact retained substitutions plus startup args.
    substitutions=[
        ("'punktnull.terrain.service_passage'", "'punktnull.terrain.platform'"),
        ("str(WORKSPACE/'tools/skill_sim.py'),", "str(WORKSPACE/'tools/platform_trial.py'),'--sim','--wall-contacts',"),
        ("        '--ground-reference',str(self.clip.source),'--controller-logs',str(self.output/'controller')]", 
         "        '--ground-reference',str(self.clip.source),'--controller-logs',str(self.output/'controller'),'--start-y',str(self.args.start_y),'--start-yaw',str(self.args.start_yaw)]"),
        ("self.wait(lambda:self.state.get('stable_full_rate_sim_seconds',0)>=.7,10)","self.wait(lambda:self.state.get('landing_stable_full_rate_seconds',0)>=.7,10)"),
        ("'STREAMED_CRAWL_RETURN_EXECUTED_PENDING_TRACE_EVALUATION'",repr(OUTCOME))]
    for before,after in substitutions:
        if source.count(before)!=1:raise RuntimeError('Retained platform startup hook changed: '+before)
        source=source.replace(before,after)
    import tools.stair_trial as parent
    namespace=dict(parent.__dict__);exec(compile(source,str(Path(__file__))+'[retained-supervisor]','exec'),namespace)
    PlatformTrial.run=namespace['run']

def evaluate_platform(run):
    report=evaluate(run,terrain_module=terrain,support_height=TOP,outcome=OUTCOME,
        first_prim=terrain.BASE+'/Landing',landing_prim=terrain.BASE+'/Landing',minimum_rise=.08,minimum_displacement=.30)
    report['kind']='Actual isolated PhysX/native120mm service-platform step-up'
    report['limitations'][0]='One120mm vertical riser with2230mm long landing; not a staircase or generic climbing.'
    report['contract_id']='platform120_actual_v2'
    report['platform_evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Independent CPU reconstruction of actual logged ankles, not reference
    # poses. A positive contact force alone can be only an overhanging toe.
    if not report.get('missing_evidence'):
        import numpy as np
        from punktnull.skills.limits import load_joint_limits
        from punktnull.skills.stairs import URDF
        from punktnull.terrain.kinematics import CollisionFK
        run=Path(run);z=np.load(run/'sim/trace.npz',allow_pickle=False);a=z['data'];cols=list(z['columns'])
        get=lambda p,n:a[:,[cols.index(p+str(i)) for i in range(n)]]
        col=lambda key:a[:,cols.index(key)]
        fk=CollisionFK(URDF,load_joint_limits().names);fk.collisions=[g for g in fk.collisions if g['link'].endswith('_ankle_roll_link')]
        root=get('root_',13);bounds=fk.bounds(fk.links(root[:,:3],root[:,3:7],get('q_',29)))
        box=terrain.collision_boxes()[0];lo=np.array(box['aabb_min']);hi=np.array(box['aabb_max'])
        inside=np.all(bounds[:,:,0,:2]>=lo[:2]+.02,axis=(1,2)) & np.all(bounds[:,:,1,:2]<=hi[:2]-.02,axis=(1,2))
        contact=np.load(run/'sim/wall_contacts.npz',allow_pickle=False);times=contact['times'];force=contact['normal_forces_world_n']
        lookup={int(t):i for i,t in enumerate(times[:,2])};normal=np.zeros(len(a),bool)
        for i,step in enumerate(col('physics_step_index')):
            j=lookup.get(int(step));normal[i]=j is not None and np.all(force[j,:,0,2]>10)
        dq=get('dq_',29);stable=inside & normal & (col('support')==0) & (col('tilt_deg')<=12) & (np.linalg.norm(root[:,7:10],axis=1)<=.10) & (np.sqrt(np.mean(dq*dq,axis=1))<=.20)
        chunks=np.flatnonzero(np.diff(np.r_[False,stable,False])).reshape(-1,2)
        longest=max([(b-a-1)*.005 for a,b in chunks],default=0.)
        manifest=json.loads((run/'manifest.json').read_text());ready=next((e for e in manifest['events'] if e['event']=='measured_skill_return_ready'),None)
        at_release=int(np.argmin(abs(col('sim_time')-ready['sim_time']))) if ready else None
        report['checks']['entire_both_feet_supported_with_20mm_edge_margin_070s']=bool(longest>=.70)
        report['checks']['entire_foot_margin_at_explicit_release']=bool(at_release is not None and inside[at_release])
        report.setdefault('measured',{}).update(longest_stable_full_foot_platform_support_s=float(longest),
            foot_xy_margin_at_release_m=float(min((bounds[at_release,:,0,:2]-lo[:2]).min(),(hi[:2]-bounds[at_release,:,1,:2]).min())) if at_release is not None else None)
        report['accepted']=bool(report['checks'] and all(report['checks'].values()))
    return report

def main():
    if '--sim' in sys.argv:
        sys.argv.remove('--sim');return sim_main(scene_module=terrain,support_height=TOP,landing_filter=0,landing_bounds=terrain.collision_boxes()[0])
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--generate',type=Path);m.add_argument('--check',action='store_true');m.add_argument('--execute',action='store_true')
    p.add_argument('--reference',type=Path,default=ROOT/'assets/platform_p12_reference_002');p.add_argument('--output',type=Path)
    p.add_argument('--gui',action='store_true');p.add_argument('--record',action='store_true');p.add_argument('--max-seconds',type=float,default=200)
    p.add_argument('--start-y',type=float,default=0.);p.add_argument('--start-yaw',type=float,default=0.)
    a=p.parse_args();a.terrain='platform120'
    if a.generate:print(json.dumps(generate(a.generate).provenance(),indent=2));return 0
    if abs(a.start_y)>.02 or abs(a.start_yaw)>1:p.error('Platform variation bounded to2cm and1deg')
    clip=ReferenceClip.load(a.reference);install()
    if a.check:print(json.dumps(dict(reference=clip.provenance(),fixture=terrain.FIXTURE_CONTRACT,physics_executed=False),indent=2));return 0
    if a.output is None:p.error('NEW --output required')
    if not 60<=a.max_seconds<=360:p.error('Bounded max-seconds60..360 required')
    os.environ.setdefault('PUNKTNULL_SKILL_CAMERA','side');os.environ.setdefault('PUNKTNULL_SKILL_CAMERA_EYE','0.8,-5.0,1.3')
    os.environ.setdefault('PUNKTNULL_SKILL_CAMERA_TARGET','0.8,0,0.65');os.environ.setdefault('PUNKTNULL_SKILL_CAMERA_WIDTH_M','4.5')
    with (ROOT/'.skill_trial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);trial=PlatformTrial(a)
        def interrupted(n,_):raise KeyboardInterrupt('Signal '+str(n))
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:trial.run()
        except (Exception,KeyboardInterrupt) as exc:
            trial.manifest['outcome']='FAILED';trial.manifest['error']=type(exc).__name__+': '+str(exc);trial.event('trial_failed',reason=trial.manifest['error'])
        finally:trial.close()
        try:component=evaluate_platform(trial.output)
        except Exception as exc:component=dict(accepted=False,evaluation_error=repr(exc))
        atomic(trial.output/'evaluation.json',component)
        # Runs are already closed. This is independent post-run evidence,
        # not a change to physics, controller, recorded frames or old reports.
        complete=None
        if a.record:
            try:
                from tools.platform_timing_audit import audit
                complete=audit(trial.output,trial.output/'evaluation.json')
            except Exception as exc:
                complete=dict(accepted=False,evaluation_error=repr(exc),kind='Post-run timing/recording evidence could not be completed')
            atomic(trial.output/'evaluation_timing_recording.json',complete)
        print(json.dumps(dict(output=str(trial.output),outcome=trial.manifest['outcome'],error=trial.manifest.get('error'),
            component_trace_accepted=bool(component.get('accepted')),
            full_component_accepted=None if complete is None else bool(complete.get('accepted')),
            evidence_scope='Full timing/uncut supplement' if a.record else 'No recording requested; full component acceptance not assessed'),indent=2))
        return 2 if trial.manifest['outcome']=='FAILED' or not component.get('accepted') or (complete is not None and not complete.get('accepted')) else 0
if __name__=='__main__':raise SystemExit(main())
