"""Explicit opt-in simulator hook for a bounded native ankle target filter."""
import argparse
from pathlib import Path


def configure(environ,argv):
    margin=float(environ.get('PUNKTNULL_ANKLE_ROLL_MARGIN_RAD','0'))
    braking=environ.get('PUNKTNULL_ANKLE_BRAKING','0')
    include_pitch=environ.get('PUNKTNULL_LIMIT_ANKLE_PITCH','0')
    if braking not in ('0','1') or include_pitch not in ('0','1'):
        raise ValueError('Explicit ankle pitch/braking options must be0or1')
    if braking=='1' and (margin<=0 or include_pitch!='1'):
        raise ValueError('Experimental ankle braking requires explicit positive target margin and pitch limiting')
    if margin==0:return None,None
    if not 0<margin<=.08:raise ValueError('Experimental ankle target margin must be >0 and <=.08rad')
    if '--ground-reference' not in argv:raise ValueError('Ankle target filter requires explicit reference-skill simulator')
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args(argv[1:])
    from .target_limits import AnkleTargetLimiter
    return AnkleTargetLimiter(enabled=True,margin_rad=margin,include_pitch=include_pitch=='1',experimental_braking=braking=='1'),args.output


def extend(source):
    before='            torque = np.clip(np.asarray(torque), -limits, limits)'
    after="""            if native_target_limiter is not None:
                torque = native_target_limiter.apply(
                    bridge.receive_snapshot() if bridge else None, torque, q, dq,
                    wall=time.monotonic(), sim=elapsed_physics, support=support, status=status)
"""+before
    if source.count(before)!=1:raise RuntimeError('Retained effort-clipping hook changed; review native target limiting')
    source=source.replace(before,after)
    before='        for sig, handler in previous_signals.items():'
    after="""        if native_target_limiter is not None:
            native_target_limiter.save(args.output)
"""+before
    if source.count(before)!=1:raise RuntimeError('Retained pre-Kit-exit evidence hook changed')
    return source.replace(before,after)
