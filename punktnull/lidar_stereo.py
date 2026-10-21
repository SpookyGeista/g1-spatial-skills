"""Metric off-axis point rendering through a fixed stereo window.

The installed Televiz 1.3 exposes stereo quads, not projection layers. Each
eye therefore sees a separate off-axis projection onto the SAME stage plane.
This is a spatial viewing window, not a panoramic immersive reconstruction.
"""
import numpy as np

from .sensors.mid360 import rotation_wxyz


class StereoWindow:
    def __init__(self, width=640, height=480, ipd_m=.064):
        if not .05 <= ipd_m <= .08:
            raise ValueError('IPD must be 50..80 mm')
        self.width, self.height, self.ipd = width, height, ipd_m
        self.size = np.array([1.6, 1.2])
        self.distance = 1.
        self.anchor_position = None
        self.anchor_rotation = None

    def anchor(self, position, quaternion_wxyz):
        self.anchor_position = np.asarray(position, float).copy()
        self.anchor_rotation = rotation_wxyz(quaternion_wxyz)

    def placement(self):
        return self.anchor_position + self.anchor_rotation @ [0., 0., -self.distance]

    def render(self, xyz_optical, colors, head_position, head_quaternion):
        if self.anchor_position is None:
            self.anchor(head_position, head_quaternion)
        # Cloud remains in initial head axes. Re-anchoring requires an explicit
        # new session; no automatic recentering while the operator moves.
        xyz = np.asarray(xyz_optical, float) * [1., -1., -1.]
        colors = np.asarray(colors, np.uint8)
        r = rotation_wxyz(head_quaternion)
        eyes = [np.asarray(head_position) + r @ [side*self.ipd/2, 0., 0.] for side in (-1, 1)]
        return tuple(self._eye(xyz, colors, (eye-self.anchor_position) @ self.anchor_rotation) for eye in eyes)

    def _eye(self, points, colors, eye):
        pixels = np.zeros((self.height, self.width, 4), np.uint8)
        pixels[..., :3] = [8, 13, 20]
        pixels[..., 3] = 255
        # An eye at/behind the window plane cannot view this portal correctly.
        if eye[2] <= -self.distance+.05 or not np.isfinite(eye).all():
            return pixels
        relative = points-eye
        forward = -relative[:, 2]
        valid = np.isfinite(points).all(axis=1) & (forward > .08)
        p, c, z = relative[valid], colors[valid], forward[valid]
        scale = (self.distance+eye[2])/z
        plane = eye[:2] + p[:, :2]*scale[:, None]
        uv = (plane/self.size + .5) * [self.width, self.height]
        uv[:, 1] = self.height-uv[:, 1]
        inside = (uv[:, 0] >= 1) & (uv[:, 0] < self.width-1) & (uv[:, 1] >= 1) & (uv[:, 1] < self.height-1)
        uv, c, z = uv[inside].astype(int), c[inside], z[inside]
        # A real z-buffer for 3x3 point splats: minimum depth wins, including
        # adjacent splats, instead of relying on repeated-index assignment.
        if len(z):
            offsets = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)])
            xy = (uv[:, None, :]+offsets).reshape(-1, 2)
            depths = np.repeat(z, 9)
            rgb = np.repeat(c, 9, axis=0)
            indices = xy[:, 1]*self.width+xy[:, 0]
            order = np.lexsort((depths, indices))
            sorted_indices = indices[order]
            first = np.r_[True, sorted_indices[1:] != sorted_indices[:-1]]
            winners = order[first]
            pixels.reshape(-1, 4)[indices[winners], :3] = rgb[winners]
        return pixels
