from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

from punktnull.rgbd_surface import surface_mesh, projection, pose_matrix, read_rgbd, write_rgbd


class SurfaceTests(unittest.TestCase):
    def test_metric_unprojection_and_depth_edges(self):
        rgb = np.full((4, 4, 3), 100, np.uint8)
        depth = np.ones((4, 4), np.float32)*2
        k = dict(fx=2., fy=2., cx=1., cy=1.)
        vertices, indices = surface_mesh(rgb, depth, k, stride=1)
        np.testing.assert_array_equal(vertices[5, :3], [0, 0, -2])
        self.assertEqual(len(indices)//3, 18)
        depth[:, 2:] = 4
        vertices, indices = surface_mesh(rgb, depth, k, stride=1)
        self.assertEqual(len(indices)//3, 12)
        self.assertTrue(np.all(np.ptp(vertices[indices.reshape(-1, 3), 2], axis=1) == 0))
        depth[:] = np.nan
        vertices, indices = surface_mesh(rgb, depth, k)
        self.assertEqual(len(indices), 0)
        self.assertTrue(np.isfinite(vertices).all())

    def test_stereo_metric_disparity_and_head_rotation(self):
        p = projection(np.radians([-45, 45, 45, -45]))
        def project(position, point):
            clip = p @ np.linalg.inv(pose_matrix(position, [1, 0, 0, 0])) @ point
            return clip[:2]/clip[3]
        near = project([-.032, 0, 0], [0, 0, -1, 1])-project([.032, 0, 0], [0, 0, -1, 1])
        far = project([-.032, 0, 0], [0, 0, -2, 1])-project([.032, 0, 0], [0, 0, -2, 1])
        np.testing.assert_allclose(near, 2*far)
        self.assertAlmostEqual(near[0], .064, places=6)
        rotated = pose_matrix([0, 0, 0], [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
        np.testing.assert_allclose((np.linalg.inv(rotated) @ [-2, 0, 0, 1])[:3], [0, 0, -2], atol=1e-6)

    def test_transport_freshness_and_explicit_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'latest.npz'
            meta = dict(frame_id=7, monotonic=time.monotonic()-10, sim_time=1., intrinsics=dict(fx=1, fy=1, cx=1, cy=1))
            write_rgbd(path, np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), meta)
            with self.assertRaises(ValueError):
                read_rgbd(path)
            self.assertEqual(read_rgbd(path, replay=True)[2]['frame_id'], 7)
            meta['monotonic'] = time.monotonic()
            write_rgbd(path, np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), meta)
            self.assertEqual(read_rgbd(path)[0].shape, (4, 4, 3))
