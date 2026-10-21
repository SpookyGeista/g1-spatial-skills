import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from punktnull.rgb_lidar import DisplayFusion, colorize, read_packet, scan_to_world, write_packet
from punktnull.lidar_stereo import StereoWindow
from punktnull.wholebody_lidar_xr import client_with_display


class ColorTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((8, 8, 3), np.uint8)
        self.rgb[4, 4] = [255, 50, 10]
        self.depth = np.full((8, 8), 2., np.float32)
        self.k = dict(fx=4., fy=4., cx=4., cy=4.)

    def test_color_visibility_occlusion_and_invalid_depth(self):
        # Same pixel but behind the surface must not borrow its foreground RGB.
        points = [[0, 0, 2], [0, 0, 3], [0, 0, -1], [9, 0, 2], [float('nan'), 0, 2]]
        xyz, colors = colorize(points, self.rgb, self.depth, np.eye(4), self.k)
        np.testing.assert_array_equal(xyz, [[0, 0, 2]])
        np.testing.assert_array_equal(colors, [[255, 50, 10]])
        self.depth[4, 4] = np.nan
        self.assertEqual(len(colorize(points, self.rgb, self.depth, np.eye(4), self.k)[0]), 0)

    def test_translated_rotated_camera(self):
        t = np.array([[0, 0, 1, 10], [1, 0, 0, 20], [0, 1, 0, 30], [0, 0, 0, 1]], float)
        xyz, _ = colorize([[12, 20, 30]], self.rgb, self.depth, t, self.k)
        np.testing.assert_allclose(xyz, [[0, 0, 2]])

    def test_scan_motion_uses_each_acquisition_pose(self):
        moved = np.eye(4); moved[0, 3] = 1.
        scan = dict(scanstart_ns=0, xyz_sensor_m=[[3, 0, 0], [2, 0, 0]], pointdt_ns=[0, 5_000_000])
        np.testing.assert_allclose(scan_to_world(scan, {0: np.eye(4), 5_000_000: moved}), [[3, 0, 0], [3, 0, 0]])
        with self.assertRaises(KeyError):
            scan_to_world(scan, {0: np.eye(4)})

    def test_reflection_not_accepted_as_rigid_rotation(self):
        t = np.eye(4); t[0, 0] = -1
        with self.assertRaises(ValueError):
            colorize([[0, 0, 2]], self.rgb, self.depth, t, self.k)

    def test_worker_pairs_sim_time_and_preserves_source_capture_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = DisplayFusion(tmp)
            try:
                worker.record_pose(0, [0, 0, 0, 1, 0, 0, 0], np.eye(4))
                worker.offer_scan(dict(complete=True, scanstart_ns=0, scan_end_ns=100_000_000,
                    xyz_sensor_m=np.array([[0, 0, 2]]), pointdt_ns=np.array([0])))
                now = time.monotonic()
                meta = dict(sim_time=.5, monotonic=now, camera_to_world=np.eye(4), intrinsics=self.k, frame_id=1)
                worker.offer_frame(meta, self.rgb, self.depth)
                self.assertIsNone(worker.pending)
                meta['sim_time'] = .1
                worker.offer_frame(meta, self.rgb, self.depth)
                deadline = time.monotonic()+2
                while not worker.path.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                xyz, colors, record = read_packet(worker.path)
                self.assertEqual(record['capture_monotonic'], now)
                self.assertEqual(record['colored_points'], 1)
                np.testing.assert_array_equal(colors, [[255, 50, 10]])
            finally:
                worker.close()
            self.assertFalse(worker.thread.is_alive())

    def test_transport_stale_future_and_shape_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'latest.npz'
            meta = dict(schema='punktnull.rgb_lidar.v1', simulation_only=True, capture_monotonic=10.)
            write_packet(path, [[0, 0, 2]], [[255, 0, 0]], meta)
            self.assertEqual(len(read_packet(path, now=11.)[0]), 1)
            for now in (9., 13.):
                with self.assertRaises(ValueError):
                    read_packet(path, now=now)
            write_packet(path, [[0, 0, 2]], [], meta)
            with self.assertRaises(ValueError):
                read_packet(path, now=11.)

    def test_shutdown_before_first_scan_keeps_status_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'sim'/'rgb_lidar'
            worker = DisplayFusion(directory)
            worker.close()
            self.assertTrue(directory.is_dir())
            self.assertFalse(worker.path.exists())
            self.assertFalse(worker.thread.is_alive())


class StereoTests(unittest.TestCase):
    def setUp(self):
        self.window = StereoWindow()

    def centroid(self, pixels, channel):
        y, x = np.nonzero(pixels[..., channel] == 255)
        self.assertGreater(len(x), 0)
        return float(x.mean())

    def test_eyes_have_correct_depth_disparity(self):
        left, right = self.window.render([[0, 0, 2]], [[255, 0, 0]], [0, 0, 0], [1, 0, 0, 0])
        # Behind the window: the left texture lies left of the right texture.
        self.assertLess(self.centroid(left, 0), self.centroid(right, 0))

    def test_point_on_window_has_no_texture_disparity(self):
        left, right = self.window.render([[0, 0, 1]], [[255, 0, 0]], [0, 0, 0], [1, 0, 0, 0])
        np.testing.assert_array_equal(left, right)

    def test_head_translation_changes_offaxis_projection(self):
        left, _ = self.window.render([[0, 0, 2]], [[255, 0, 0]], [0, 0, 0], [1, 0, 0, 0])
        moved, _ = self.window.render([[0, 0, 2]], [[255, 0, 0]], [.2, 0, 0], [1, 0, 0, 0])
        self.assertGreater(self.centroid(moved, 0), self.centroid(left, 0))

    def test_near_splat_wins_independent_of_input_order(self):
        eye = np.array([0., 0., 0.])
        a = self.window._eye(np.array([[0, 0, -2], [0, 0, -1]]), np.array([[0, 255, 0], [255, 0, 0]]), eye)
        b = self.window._eye(np.array([[0, 0, -1], [0, 0, -2]]), np.array([[255, 0, 0], [0, 255, 0]]), eye)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[240, 320, :3], [255, 0, 0])

    def test_invalid_ipd(self):
        with self.assertRaises(ValueError):
            StereoWindow(ipd_m=.2)


class TrackingTests(unittest.TestCase):
    def test_original_client_unchanged_and_old_snapshot_not_renewed(self):
        class Original:
            def __init__(self):
                pass
            def get_pose_by_name(self, name):
                return self._get_tracker_data()[name]
        adapted = client_with_display(Original, '/nonexistent')()
        adapted._display_latest = {'head': 'actual-device-snapshot'}
        adapted._display_stamp = time.monotonic()
        self.assertEqual(adapted.get_pose_by_name('head'), 'actual-device-snapshot')
        adapted._display_stamp -= 1.
        self.assertIsNone(adapted._get_tracker_data())
        self.assertFalse(hasattr(Original, 'start_streaming'))
        adapted.close()


if __name__ == '__main__':
    unittest.main()
