#!/usr/bin/env python3
"""Run retained Isaac motor loop with additive skill guard; simulator only."""
from pathlib import Path
import importlib
import os
import sys
import types
import hashlib
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from punktnull.skills.legacy import OLD,WORKSPACE,LEGACY,load

def main():
    # Preserve actual bridge, controller geometry and motor integration unchanged.
    sys.path.insert(1,str(LEGACY));sys.path.insert(2,str(OLD))
    from punktnull.skills import guard,runtime
    from punktnull.skills.target_limit_hook import configure,extend
    native_target_limiter,target_limiter_output=configure(os.environ,sys.argv)
    sys.modules['ground_guard']=guard
    sys.modules['runtime_view']=runtime
    scene_module=os.environ.get('PUNKTNULL_SKILL_SCENE_MODULE')
    if scene_module:
        if not scene_module.startswith('punktnull.'):raise ValueError('Scene hook must belong to this additive package')
        sys.modules['scene']=importlib.import_module(scene_module)
    path=LEGACY/'run_isaac.py';original=path.read_text()
    before='ground_guard.assess(now, base[:3], base[3:7], q, dq, support)'
    after='ground_guard.assess(time.monotonic(), base[:3], base[3:7], q, dq, support, efforts=torque, state_monotonic=now, command_status=status)'
    if original.count(before)!=1:raise RuntimeError('Retained guard call changed; review timestamp hook')
    source=original.replace(before,after)
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
    before='runtime_view.update(base, torso_state, q, dq, state, list(robot.body_names), support,'
    after='runtime_view.current_applied_efforts = torque.copy(); runtime_view.update(base, torso_state, q, dq, state, list(robot.body_names), support,'
    if source.count(before)!=1:raise RuntimeError('Retained applied effort telemetry hook changed')
    source=source.replace(before,after)
    # Match the LiDAR mission's disclosed 10fps presentation profile. The
    # reference controller stays50Hz and the real physics stays200Hz.
    for before,after in [
        ("parser.add_argument('--render-every', type=int, default=10,", "parser.add_argument('--render-every', type=int, default=20,"),
        ('if args.record and (not args.output or args.render_every != 10):',
         'if args.record and (not args.output or args.render_every not in (10,20)):'),
        ("parser.error('Recording requires output and render-every 10 (20fps)')",
         "parser.error('Recording requires output and render-every10 or20')")]:
        if source.count(before)!=1:raise RuntimeError('Retained render hook changed')
        source=source.replace(before,after)
    runner=types.ModuleType('skill_motor_loop');runner.__file__=str(path)
    if native_target_limiter is not None:source=extend(source)
    exec(compile(source,str(path),'exec'),runner.__dict__)
    runner.native_target_limiter=native_target_limiter
    print('SKILL_GUARD_CLOCK_HOOK',hashlib.sha256(original.encode()).hexdigest(),
          'Fresh receipt clock after render; age limits unchanged; default presentation10fps / physics200Hz',flush=True)
    runner.HERE=WORKSPACE
    try:runner.main()
    finally:
        if native_target_limiter is not None and not native_target_limiter.saved:native_target_limiter.save(target_limiter_output)

if __name__=='__main__':main()
