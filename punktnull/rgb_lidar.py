"""Display-only RGB/LiDAR fusion. No controller or estimator inputs.

Simulation poses compensate sensor motion against a static scene. Moving
objects remain subject to scan skew; camera depth rejects inconsistent returns.
"""
from collections import deque
from pathlib import Path
import io
import json
import threading
import time

import numpy as np

from .sensors.mid360 import rotation_wxyz

MAX_POINTS = 20000
MAX_PACKET_BYTES = 2 * 1024 * 1024


def rigid(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(t[:3, :3]) < .999):
        raise ValueError('Expected a rigid transform')
    return t


def scan_to_world(scan, poses):
    """Use the pose at each actual 5ms ray batch, never the end-of-scan pose."""
    points = np.asarray(scan['xyz_sensor_m'], dtype=float)
    stamps = int(scan['scanstart_ns']) + np.asarray(scan['pointdt_ns'], dtype=np.int64)
    if points.shape != (len(stamps), 3) or not np.isfinite(points).all():
        raise ValueError('Invalid scan points')
    world = np.empty_like(points)
    for stamp in np.unique(stamps):
        t = rigid(poses[int(stamp)])  # missing pose is an error, not an approximation
        mask = stamps == stamp
        world[mask] = points[mask] @ t[:3, :3].T + t[:3, 3]
    return world


def colorize(points_world, rgb, depth, camera_to_world, intrinsics, *, tolerance_m=.04):
    """Return only camera-visible, depth-consistent LiDAR samples in optical XYZ."""
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.dtype != np.uint8 or rgb.shape != (*depth.shape, 3) or depth.ndim != 2:
        raise ValueError('Expected matching uint8 RGB and metric depth')
    t = rigid(camera_to_world)
    points = (np.asarray(points_world, dtype=float) - t[:3, 3]) @ t[:3, :3]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected Nx3 points')
    k = {name: float(intrinsics[name]) for name in ('fx', 'fy', 'cx', 'cy')}
    if not np.isfinite(list(k.values())).all() or min(k['fx'], k['fy']) <= 0:
        raise ValueError('Invalid intrinsics')
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > .1)
    points = points[valid]
    uv = points[:, :2] / points[:, 2:3] * [k['fx'], k['fy']] + [k['cx'], k['cy']]
    h, w = depth.shape
    visible = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    points, uv = points[visible], uv[visible].astype(int)
    measured = depth[uv[:, 1], uv[:, 0]]
    # Collision geometry and rendered surfaces can differ slightly. An invalid
    # or occluding camera depth never contributes a fabricated colour.
    consistent = np.isfinite(measured) & (measured > .1) & (abs(measured - points[:, 2]) <= tolerance_m)
    points, uv = points[consistent], uv[consistent]
    if len(points) > MAX_POINTS:
        indices = np.linspace(0, len(points)-1, MAX_POINTS, dtype=int)
        points, uv = points[indices], uv[indices]
    return points.astype('<f4'), rgb[uv[:, 1], uv[:, 0]].copy()


def write_packet(path, xyz, colors, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.next.npz')
    np.savez(temporary, xyz=np.asarray(xyz, dtype='<f4'), rgb=np.asarray(colors, dtype=np.uint8),
             metadata=json.dumps(metadata, allow_nan=False))
    temporary.replace(path)


def read_packet(path, *, max_age_s=2., now=None, replay=False):
    """Atomic local-file transport, bounded and pickle-free. Stale is explicit."""
    with Path(path).open('rb') as handle:
        payload = handle.read(MAX_PACKET_BYTES+1)
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError('Display packet too large')
    # Our packets are uncompressed; reject zip bombs before NumPy materializes.
    import zipfile
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > MAX_PACKET_BYTES:
            raise ValueError('Expanded display packet too large')
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        xyz, rgb = data['xyz'].copy(), data['rgb'].copy()
        metadata = json.loads(str(data['metadata']))
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) > MAX_POINTS
            or rgb.shape != xyz.shape or xyz.dtype != np.dtype('<f4')
            or rgb.dtype != np.uint8 or not np.isfinite(xyz).all()):
        raise ValueError('Malformed display cloud')
    if metadata.get('schema') != 'punktnull.rgb_lidar.v1' or not metadata.get('simulation_only'):
        raise ValueError('Unsupported display source')
    age = (time.monotonic() if now is None else now) - float(metadata['capture_monotonic'])
    if not np.isfinite(age) or (not replay and (age < 0 or age > max_age_s)):
        raise ValueError('Display cloud stale')
    return xyz, rgb, metadata


class DisplayFusion:
    """Bounded latest-frame worker; disk/fusion never run on the physics thread."""
    def __init__(self, directory):
        self.path = Path(directory)/'latest.npz'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.poses = {}
        self.pose_stamps = deque()
        self.scan = None
        self.pending = None
        self.error = None
        self.closed = False
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._work, name='rgb-lidar-display', daemon=True)
        self.thread.start()

    def record_pose(self, stamp_ns, torso, T_torso_lidar):
        t = np.eye(4)
        t[:3, :3] = rotation_wxyz(torso[3:7])
        t[:3, 3] = torso[:3]
        self.poses[int(stamp_ns)] = t @ T_torso_lidar
        self.pose_stamps.append(int(stamp_ns))
        while len(self.pose_stamps) > 100:
            self.poses.pop(self.pose_stamps.popleft(), None)

    def offer_scan(self, scan):
        if not scan['complete']:
            return
        stamps = int(scan['scanstart_ns']) + np.asarray(scan['pointdt_ns'], dtype=np.int64)
        poses = {int(s): self.poses[int(s)] for s in np.unique(stamps)}
        with self.condition:
            self.scan = (scan, poses)

    def offer_frame(self, metadata, rgb, depth):
        with self.condition:
            if self.scan is None or self.closed:
                return
            scan, poses = self.scan
            skew = float(metadata['sim_time']) - int(scan['scan_end_ns']) / 1e9
            if not -.006 <= skew <= .12:
                return
            self.pending = (scan, poses, dict(metadata), rgb.copy(), depth.copy())
            self.condition.notify()

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.closed:
                    return
                scan, poses, meta, rgb, depth = self.pending
                self.pending = None
            try:
                world = scan_to_world(scan, poses)
                xyz, colors = colorize(world, rgb, depth, meta['camera_to_world'], meta['intrinsics'])
                record = dict(schema='punktnull.rgb_lidar.v1', simulation_only=True,
                    frame_id=int(meta['frame_id']), sim_time=float(meta['sim_time']),
                    capture_monotonic=float(meta['monotonic']), scanstart_ns=int(scan['scanstart_ns']),
                    scan_end_ns=int(scan['scan_end_ns']), input_points=len(world), colored_points=len(xyz),
                    coordinate_frame='camera optical: x right, y down, z forward; metres',
                    camera_source=meta.get('source'),
                    pose_source='simulator per-ray-batch poses, display only; static-scene motion compensation',
                    lidar_model='MID360-like PhysX proxy; not calibrated hardware',
                    uncolored_policy='omit outside image, invalid depth and occluded returns')
                write_packet(self.path, xyz, colors, record)
                self.error = None
            except Exception as exc:
                self.error = str(exc)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join(timeout=3)
