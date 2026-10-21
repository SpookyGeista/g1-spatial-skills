"""Post-run evidence check of native streamed ownership, not runtime control."""
import hashlib
from punktnull.skills.legacy import OLD

NATIVE = OLD/'data/teleop_archives/g1_pose_2026-09-09/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref'

def native_reference_phase(rows, begin, ready):
    """Rows=(monotonic,name,encoder,play); bounded paused release is explicit.

    Native ZMQManager handles planner safety reset by clearing play first,
    initializes planner with measured pose, then sets play true. A logged
    `streamed,0,false` AFTER explicit release is therefore a stop transition,
    not active POSE execution. It is never allowed inside the owned skill.
    """
    result=dict(accepted=False,contract='native_reference_owned_phase_v2',
        transition_max_seconds=.10,transition_max_paused_stream_rows=5,
        maximum_csv_gap_seconds=.0401,entry_ack_max_seconds=.12)
    if begin is None or ready is None or not rows or ready<=begin:
        result['reason']='Missing ordered measured start/release';return result
    stream=[r for r in rows if r[1]=='streamed']
    active=[r for r in stream if begin<=r[0]<ready]
    paused=[r for r in stream if r[3]<=.5]
    full=[r for r in rows if active and active[0][0]<=r[0]<ready]
    steady=bool(active and len(active)>50 and all(r[1]=='streamed' and r[2]==0 and r[3]>.5 for r in full))
    fresh=bool(full and 0<=active[0][0]-begin<=.12 and 0<=ready-full[-1][0]<=.0401
        and all(0<b[0]-a[0]<=.0401 for a,b in zip(full,full[1:])))
    bounded=bool(len(paused)<=5 and all(ready<=r[0]<=ready+.10 and r[2]==0 for r in paused)
        and all(begin<=r[0]<=ready+.10 and r[2]==0 for r in stream))
    result.update(accepted=steady and fresh and bounded,active_phase=steady,csv_freshness=fresh,
        bounded_explicit_release=bounded,active_stream_rows=len(active),paused_release_rows=paused,
        active_duration_seconds=(full[-1][0]-full[0][0]) if full else None,
        first_ack_delay_seconds=active[0][0]-begin if active else None,
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            NATIVE/'include/input_interface/zmq_manager.hpp',
            NATIVE/'src/g1_deploy_onnx_ref.cpp',NATIVE/'src/state_logger.cpp')})
    return result
