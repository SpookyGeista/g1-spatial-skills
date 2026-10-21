"""Validate existing references and serialize native joint-based POSE v1.

No learned model or robot controller is implemented here. Protocol v1 selects
native encoder mode 0. Teleoperation's v3 selects SMPL mode 2 and MUST NOT be
filled with invented zero SMPL data to play a joint-space reference.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .legacy import OLD, WORKSPACE, ground_module, load

FPS = 50
HEADER_SIZE = 1280
WIDTHS = {'joint_pos': 29, 'joint_vel': 29, 'body_pos': 42,
          'body_quat': 56, 'body_lin_vel': 42, 'body_ang_vel': 42}


def write_skill_reference(path, root_xyz, root_wxyz, q_sdk, sourcehashes, modifications, *, skill):
    """Export declared local50Hz targets with exact retained FK/permutation.

    This writes no measured execution claim and never alters the input arrays.
    Effort/contact bounds still require the actual200Hz simulator experiment.
    """
    from .limits import load_joint_limits
    path=Path(path).resolve()
    if path.exists() or not any(path.is_relative_to(WORKSPACE/p) for p in ('assets','evidence')):
        raise ValueError('Use a NEW local assets/evidence reference directory')
    if skill not in ('crawl','stairs') or not modifications:raise ValueError('Explicit skill/modifications required')
    converter_path=OLD/'tools/convert_bones_to_sonic_reference.py'
    mjcf=OLD/'gear_sonic_deploy/g1/g1_29dof.xml'
    converter=load('generic_skill_converter',converter_path)
    fk=converter.G1Kinematics(mjcf);limits=load_joint_limits()
    xyz=np.asarray(root_xyz,float);quat=np.asarray(root_wxyz,float);q=np.asarray(q_sdk,float)
    if xyz.shape!=(len(q),3) or quat.shape!=(len(q),4) or q.shape!=(len(q),29) or len(q)<100:
        raise ValueError('Invalid50Hz reference shape')
    if fk.joint_names!=list(limits.names):raise ValueError('Native FK joint order changed')
    if not np.isfinite(np.c_[xyz,quat,q]).all() or abs(np.linalg.norm(quat,axis=1)-1).max()>1e-5:
        raise ValueError('Nonfinite/nonunit reference')
    qil=np.empty_like(q);qil[:,converter.MJ_TO_IL]=q
    bp,bq=fk.forward(xyz,quat,q)
    dq,bv,bw=converter.calculate_velocities(qil,bp,bq,FPS)
    if any(limits.check(qi,vi,require_effort=False) for qi,vi in zip(q,dq[:,converter.MJ_TO_IL])):
        raise ValueError('Reference violates actual position/speed limits')
    channels=dict(joint_pos=qil,joint_vel=dq,body_pos=bp,body_quat=bq,body_lin_vel=bv,body_ang_vel=bw)
    hashes=dict(sourcehashes)
    for p in (converter_path,mjcf,Path(__file__)):
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Source hash mismatch '+name)
    path.mkdir(parents=True)
    for key,width in WIDTHS.items():
        np.savetxt(path/(key+'.csv'),channels[key].reshape(len(q),width),delimiter=',',
            header=','.join(f'{key}_{i}' for i in range(width)),comments='',fmt='%.9f')
    (path/'metadata.txt').write_text(f'Total timesteps: {len(q)}\nFPS: 50\nJoint order: IsaacLab29\nGenerated targets; physical execution unvalidated.\n')
    manifest=dict(schema=1,kind='generated_g1_skill_reference',skill=skill,fps=FPS,joint_order='IsaacLab29',
        sources_sha256=hashes,modifications=modifications,
        channels_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())},
        physical_execution=False)
    (path/'generated_reference.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return ReferenceClip.load(path)


@dataclass(frozen=True)
class ReferenceClip:
    source: Path
    data: dict
    hashes: dict
    digest: str

    @classmethod
    def load(cls, source):
        source = Path(source).resolve()
        generated = None
        if source.is_relative_to(OLD):
            # Reuse the actual existing loader and exact quaternion/joint conventions.
            route = load('crawl_route_data', OLD/'data/isaac_sonic_bones_crawl_2026-09-16/route_data.py')
            original = route._read_reference(str(source.relative_to(OLD)))
            data = {k: v.copy().reshape(len(v), -1) for k, v in original.items()}
        elif any(source.is_relative_to(WSPACE) for WSPACE in (WORKSPACE/'assets', WORKSPACE/'evidence')):
            generated = json.loads((source/'generated_reference.json').read_text())
            if (generated.get('schema') != 1 or generated.get('kind') != 'generated_g1_skill_reference'
                    or generated.get('skill') not in ('crawl','stairs') or generated.get('fps') != FPS
                    or generated.get('joint_order') != 'IsaacLab29' or not generated.get('modifications')):
                raise ValueError('Unsupported generated reference provenance')
            # A local manifest is not execution evidence. Bind every source and
            # generated channel to immutable bytes before streaming any pose.
            sources = generated.get('sources_sha256', {})
            if not sources or not any(Path(p).resolve().is_relative_to(OLD) for p in sources):
                raise ValueError('Generated reference needs retained source provenance')
            for name,digest in sources.items():
                path=Path(name).resolve()
                if not (path.is_relative_to(OLD) or path.is_relative_to(WORKSPACE)):
                    raise ValueError('Generated provenance source outside audited repositories')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                    raise ValueError('Generated reference source changed: '+name)
            data = {}
        else:
            raise ValueError('Only retained or declared local generated references are accepted')
        for key, width in WIDTHS.items():
            if key not in data:
                data[key] = np.loadtxt(source/(key+'.csv'), delimiter=',', skiprows=1, ndmin=2)
            a = data[key]
            if a.ndim != 2 or a.shape[1] != width or not np.isfinite(a).all():
                raise ValueError('Invalid channel '+key)
        frames = len(data['joint_pos'])
        if not 100 <= frames <= 30000 or any(len(a) != frames for a in data.values()):
            raise ValueError('Invalid or inconsistent frame counts')
        metadata = (source/'metadata.txt').read_text()
        count = re.search(r'Total timesteps:\s*(\d+)', metadata)
        if not count or int(count.group(1)) != frames:
            raise ValueError('Metadata frame count mismatch')
        q = data['body_quat'].reshape(frames, 14, 4)
        if abs(np.linalg.norm(q, axis=-1)-1).max() > 1e-5:
            raise ValueError('Invalid quaternion norms')
        for a in data.values(): a.setflags(write=False)
        names = [k+'.csv' for k in WIDTHS] + ['metadata.txt']
        hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
        if generated is not None:
            if generated.get('channels_sha256') != hashes:
                raise ValueError('Generated reference channel hash mismatch')
            hashes['generated_reference.json']=hashlib.sha256((source/'generated_reference.json').read_bytes()).hexdigest()
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        return cls(source, data, hashes, digest)

    @property
    def frames(self): return len(self.data['joint_pos'])

    @property
    def seconds(self): return (self.frames-1)/FPS

    @property
    def root(self): return self.data['body_pos'][:, :3]

    @property
    def quaternion(self): return self.data['body_quat'][:, :4]

    def provenance(self):
        return {'source': str(self.source), 'source_sha256': self.hashes,
                'ordered_channel_digest': self.digest, 'frames': self.frames,
                'fps': FPS, 'fps_basis': 'retained SONIC control_dt 0.02',
                'joint_order': 'IsaacLab 29; retained converter permutation',
                'backend': 'native SONIC POSE protocol 1, encoder 0',
                'physical_validation': False}


def pack_reference_pose(fields):
    # Reuse the retained serializer; change only its explicitly versioned header.
    gateway = load('pose_gateway', OLD/'data/null_vc_demo_2026-09-16/pose_gateway.py')
    packet = gateway.pack_pose(fields)
    header = json.loads(packet[4:4+HEADER_SIZE].split(b'\0', 1)[0])
    header['v'] = 1
    encoded = json.dumps(header, separators=(',', ':')).encode()
    return b'pose'+encoded.ljust(HEADER_SIZE, b'\0')+packet[4+HEADER_SIZE:]


def validate_reference_pose(packet, *, now, last_frame=None, max_age=.20):
    if not isinstance(packet, bytes) or len(packet) < 4+HEADER_SIZE:
        raise ValueError('Invalid POSE packet')
    raw = packet[4:4+HEADER_SIZE]
    if packet[:4] != b'pose' or raw.partition(b'\0')[2].strip(b'\0'):
        raise ValueError('Invalid topic/header padding')
    try: header = json.loads(raw.split(b'\0', 1)[0])
    except (ValueError, UnicodeDecodeError) as exc: raise ValueError('Invalid POSE header') from exc
    if header.get('v') != 1:
        raise ValueError('Reference playback requires joint-based native protocol 1')
    header['v'] = 3
    normalized = b'pose'+json.dumps(header, separators=(',', ':')).encode().ljust(HEADER_SIZE,b'\0')+packet[4+HEADER_SIZE:]
    gateway = load('pose_gateway', OLD/'data/null_vc_demo_2026-09-16/pose_gateway.py')
    fields = gateway.unpack_pose(normalized)
    required = {'joint_pos','joint_vel','body_quat_w','frame_index','timestamp_monotonic','catch_up'}
    if set(fields) != required or any(not np.isfinite(a).all() for a in fields.values()):
        raise ValueError('Invalid reference POSE fields')
    frames = fields['frame_index']; n = len(frames)
    if frames.ndim != 1 or frames.dtype.kind not in 'iu' or not 1 <= n <= 64 or frames[0] < 0 or np.any(np.diff(frames) != 1):
        raise ValueError('Reference frames must be contiguous and nonnegative')
    newest = int(frames[-1])
    if last_frame is not None and newest <= last_frame:
        raise ValueError('Repeated/regressing reference packet')
    for key, shape in [('joint_pos',(n,29)),('joint_vel',(n,29)),('body_quat_w',(n,4))]:
        if fields[key].shape != shape or fields[key].dtype.kind != 'f': raise ValueError('Invalid '+key)
    if abs(np.linalg.norm(fields['body_quat_w'],axis=1)-1).max() > .02:
        raise ValueError('Invalid quaternion')
    stamp = fields['timestamp_monotonic']
    if stamp.shape != (1,) or stamp.dtype.kind != 'f': raise ValueError('Invalid producer timestamp')
    stamp = float(stamp[0])
    if not np.isfinite(now) or stamp <= 0 or not -.02 <= now-stamp <= max_age:
        raise ValueError('Stale/future reference packet')
    if fields['catch_up'].shape != (1,) or bool(fields['catch_up'][0]):
        raise ValueError('Reference catch-up skips are forbidden')
    return fields, newest, stamp


class ReferencePoseStream:
    """One owner, 50 Hz cursor; future reference horizon, no execution claims.

    Prime/final hold repeats the existing endpoint pose with advancing wire IDs.
    Each motion call advances exactly one source frame. Delayed callers fail;
    they never skip source frames to catch up. Completion requires measured
    postconditions in SkillSession, not this sent cursor reaching its end.
    """
    def __init__(self, clip, *, horizon=51):
        if not 46 <= horizon <= 64: raise ValueError('Native encoder needs 10 frames with stride5 lookahead')
        self.clip, self.horizon = clip, horizon
        self.wire_index = 0; self.source_frame = 0; self.last_time = None
        self.sent_packets = 0; self.started = False; self.finished_sending = False

    def packet(self, now, *, advance=False):
        if not np.isfinite(now) or now <= 0: raise ValueError('Invalid time')
        if self.last_time is not None and not 0 < now-self.last_time <= .10:
            raise TimeoutError('Reference stream late, duplicate time or regressing time; pause simulation')
        if advance: self.started = True
        start = self.source_frame
        ids = np.arange(self.wire_index,self.wire_index+self.horizon,dtype='<i8')
        indices = np.minimum(np.arange(start,start+self.horizon),self.clip.frames-1) if advance else np.full(self.horizon,start)
        fields = {'joint_pos': np.asarray(self.clip.data['joint_pos'][indices],dtype='<f4'),
                  'joint_vel': np.asarray(self.clip.data['joint_vel'][indices] if advance else np.zeros((self.horizon,29)),dtype='<f4'),
                  'body_quat_w': np.asarray(self.clip.quaternion[indices],dtype='<f4'),
                  'frame_index': ids,'timestamp_monotonic': np.array([now],dtype='<f8'),
                  'catch_up': np.array([False],dtype='?')}
        packet = pack_reference_pose(fields)
        validate_reference_pose(packet,now=now)
        self.last_time=now; self.wire_index+=1; self.sent_packets+=1
        if advance:
            self.source_frame=min(self.source_frame+1,self.clip.frames-1)
            self.finished_sending=start == self.clip.frames-1
        return packet

    def status(self):
        return {'source_frame':self.source_frame,'wire_index':self.wire_index,
                'sent_packets':self.sent_packets,'monotonic':self.last_time,
                'finished_sending':self.finished_sending,'cursor_basis':'submitted reference, NOT native executed frame',
                'source_digest':self.clip.digest}


def attach_reference_validation(client):
    """Reuse exclusive retained PlannerClient with strict v1 validation only.

    Install before request_pose. Native startup, ownership transitions, stale
    POSE stop and DDS source remain the retained implementation.
    """
    with client._lock:
        if client._mode not in ('DISARMED','PLANNER'):
            raise RuntimeError('Cannot change packet contract during active POSE')
        client._validate_pose = validate_reference_pose
