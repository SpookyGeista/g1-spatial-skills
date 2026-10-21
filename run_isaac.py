#!/usr/bin/env python3
"""Reuse unchanged retained motor control with disclosed sensor/render hooks."""
import difflib
import hashlib
import importlib
import os
import sys
import types
from punktnull.retained import ROOT,LEGACY,prepare
prepare()
if os.environ.get('PUNKTNULL_MISSION_SCENE'):
    if os.environ['PUNKTNULL_MISSION_SCENE']!='punktnull.terrain.inspection_passage':
        raise ValueError('Unsupported mission scene override')
    sys.modules['scene']=importlib.import_module(os.environ['PUNKTNULL_MISSION_SCENE'])
from punktnull import isaac_runtime
sys.modules['runtime_view']=isaac_runtime
from punktnull.skills.target_limit_hook import configure as configure_target_limits,extend as extend_target_limits
native_target_limiter,target_limiter_output=configure_target_limits(os.environ,sys.argv)
path=LEGACY/'run_isaac.py';original=path.read_text();source=original
changes=[
    ("parser.add_argument('--render-every', type=int, default=10,", "parser.add_argument('--render-every', type=int, default=20,"),
    ('if args.record and (not args.output or args.render_every != 10):',
     'if args.record and (not args.output or args.render_every not in (10,20)):'),
    ("parser.error('Recording requires output and render-every 10 (20fps)')",
     "parser.error('Recording requires output and render-every 10 (20fps) or20 (10fps)')"),
    ('sim_cfg = sim_utils.SimulationCfg(\n',
     'sim_cfg = sim_utils.SimulationCfg(\n            enable_scene_query_support=True,\n'),
    ('            sim.step(render=False)\n',
     '            if runtime_view is not None:\n                runtime_view.before_physics(torso_state, elapsed_physics, engine_index-engine_index_epoch, status)\n            sim.step(render=False)\n'),
    ('                runtime_view.update(base, torso_state, q, dq, state, list(robot.body_names), support,\n',
     '                runtime_view.current_applied_efforts = torque.copy()\n                runtime_view.update(base, torso_state, q, dq, state, list(robot.body_names), support,\n')]
if '--ground-reference' in sys.argv:
    changes.append(('ground_guard.assess(now, base[:3], base[3:7], q, dq, support)',
                    'ground_guard.assess(time.monotonic(), base[:3], base[3:7], q, dq, support, efforts=torque, state_monotonic=now, command_status=status)'))
    release_needle="            if not support and bridge and status['mode'] != 'active':"
    changes.append((release_needle,"""            if not support and support_was_active and ground_guard:
                release_error=ground_guard.joint_limits.check(q,dq,None if torque is None else np.clip(torque,-limits,limits))
                if release_error:
                    meta['end_reason']='release_refused'
                    event('release_refused',reason=release_error)
                    break
"""+release_needle))
for before,after in changes:
    if source.count(before)!=1:raise RuntimeError('Retained motor loop changed; review sensor hook: '+before)
    source=source.replace(before,after)
if native_target_limiter is not None:source=extend_target_limits(source)
if '--ground-reference' in sys.argv:
    from punktnull.skills import guard
    sys.modules['ground_guard']=guard
module=types.ModuleType('_retained_lio_motor_loop');module.__file__=str(path)
exec(compile(source,str(path), 'exec'),module.__dict__)
module.HERE=ROOT
module.native_target_limiter=native_target_limiter
if __name__=='__main__':
    if 'PUNKTNULL_START_YAW_OFFSET' in os.environ:
        if '--start-yaw' in sys.argv:raise RuntimeError('Duplicate start variation')
        sys.argv+=['--start-yaw',os.environ['PUNKTNULL_START_YAW_OFFSET']]
    print('RETAINED_MOTOR_LOOP_SHA256',hashlib.sha256(original.encode()).hexdigest(),flush=True)
    print('SENSOR_HOOK_PATCH\n'+''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True))),flush=True)
    try:module.main()
    finally:
        if native_target_limiter is not None and not native_target_limiter.saved:native_target_limiter.save(target_limiter_output)
