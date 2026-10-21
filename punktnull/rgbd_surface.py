"""Metric observed RGB-D surfaces, without filling unseen space or depth edges."""
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from .sensors.mid360 import rotation_wxyz

MAX_BYTES = 2 * 1024 * 1024


def write_rgbd(path, rgb, depth, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(schema='punktnull.rgbd.v1', simulation_only=True,
                  frame_id=int(metadata['frame_id']), capture_monotonic=float(metadata['monotonic']),
                  sim_time=float(metadata['sim_time']), intrinsics=metadata['intrinsics'],
                  source='simulated torso RGB-D; not real D435i calibration')
    temporary = path.with_suffix('.next.npz')
    np.savez(temporary, rgb=np.asarray(rgb, np.uint8), depth=np.asarray(depth, '<f4'),
             metadata=json.dumps(record, allow_nan=False))
    temporary.replace(path)


def read_rgbd(path, *, max_age_s=2., replay=False):
    with Path(path).open('rb') as handle:
        payload = handle.read(MAX_BYTES+1)
    if len(payload) > MAX_BYTES:
        raise ValueError('RGB-D packet too large')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(e.file_size for e in archive.infolist()) > MAX_BYTES:
            raise ValueError('Expanded RGB-D packet too large')
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        rgb, depth = data['rgb'].copy(), data['depth'].copy()
        meta = json.loads(str(data['metadata']))
    if (depth.ndim != 2 or min(depth.shape) < 2 or depth.size > 640*360
            or depth.dtype != np.dtype('<f4') or rgb.dtype != np.uint8
            or rgb.shape != (*depth.shape, 3)):
        raise ValueError('Malformed RGB-D image')
    if meta.get('schema') != 'punktnull.rgbd.v1' or meta.get('simulation_only') is not True:
        raise ValueError('Unsupported RGB-D source')
    k = {key: float(meta['intrinsics'][key]) for key in ('fx', 'fy', 'cx', 'cy')}
    if not np.isfinite(list(k.values())).all() or min(k['fx'], k['fy']) <= 0:
        raise ValueError('Invalid intrinsics')
    age = time.monotonic()-float(meta['capture_monotonic'])
    if not np.isfinite(age) or (not replay and not 0 <= age <= max_age_s):
        raise ValueError('RGB-D source stale')
    return rgb, depth, meta


def surface_mesh(rgb, depth, intrinsics, *, stride=2, near=.15, far=8.):
    """Image-grid mesh in metres, +X right, +Y up, -Z forward.

    Invalid depths and discontinuities remain holes. This is a live observed
    front surface, not a watertight model or a fabricated 360-degree scene.
    """
    d = np.asarray(depth)[::stride, ::stride]
    v, u = np.mgrid[:depth.shape[0]:stride, :depth.shape[1]:stride]
    k = intrinsics
    valid = np.isfinite(d) & (d >= near) & (d <= far)
    z = np.where(valid, d, 0.)
    xyz = np.stack(((u-k['cx'])*z/k['fx'], -(v-k['cy'])*z/k['fy'], -z), axis=-1)
    colors = np.asarray(rgb)[::stride, ::stride].astype(np.float32)/255.
    vertices = np.concatenate((xyz, colors), axis=-1).reshape(-1, 6).astype(np.float32)
    ids = np.arange(d.size, dtype=np.uint32).reshape(d.shape)
    a, b, c, e = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    triangles = np.concatenate((np.stack((a, c, b), -1).reshape(-1, 3),
                                np.stack((b, c, e), -1).reshape(-1, 3)))
    depths = z.ravel()[triangles]
    continuous = (depths.max(axis=1)-depths.min(axis=1)) <= .04+.02*depths.min(axis=1)
    keep = valid.ravel()[triangles].all(axis=1) & continuous
    return vertices, np.ascontiguousarray(triangles[keep].ravel())


def pose_matrix(position, quaternion_wxyz):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation_wxyz(quaternion_wxyz)
    matrix[:3, 3] = position
    return matrix


def projection(angles, near=.05, far=30.):
    """OpenGL projection from the runtime's asymmetric eye FOV, radians."""
    left, right, up, down = np.tan(angles)
    p = np.zeros((4, 4), np.float32)
    p[0, 0], p[1, 1] = 2/(right-left), 2/(up-down)
    p[0, 2], p[1, 2] = (right+left)/(right-left), (up+down)/(up-down)
    p[2, 2], p[2, 3], p[3, 2] = -(far+near)/(far-near), -2*far*near/(far-near), -1
    return p
