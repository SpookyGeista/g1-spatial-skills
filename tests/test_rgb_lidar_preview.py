import base64
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from punktnull.rgb_lidar import write_packet
from tools.view_rgb_lidar import preview_frame


class PreviewTests(unittest.TestCase):
    def test_missing_corrupt_stale_and_explicit_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertIsNone(preview_frame(run)['image'])
            packet = run/'sim/rgb_lidar/latest.npz'
            packet.parent.mkdir(parents=True)
            packet.write_bytes(b'broken archive')
            self.assertIsNone(preview_frame(run)['image'])
            metadata = dict(schema='punktnull.rgb_lidar.v1', simulation_only=True,
                            capture_monotonic=time.monotonic()-10., frame_id=23)
            write_packet(packet, [[0, 0, 2]], [[255, 0, 0]], metadata)
            self.assertIsNone(preview_frame(run)['image'])
            replay = preview_frame(run, replay=True)
            self.assertIn('AUFZEICHNUNG', replay['status'])
            self.assertEqual(replay['frame_id'], 23)
            self.assertTrue(base64.b64decode(replay['image']).startswith(b'\x89PNG'))

    def test_live_image_disappears_when_source_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            packet = run/'sim/rgb_lidar/latest.npz'
            meta = dict(schema='punktnull.rgb_lidar.v1', simulation_only=True,
                        capture_monotonic=time.monotonic(), frame_id=1)
            write_packet(packet, [[0, 0, 2]], [[255, 0, 0]], meta)
            self.assertIn('LIVE', preview_frame(run)['status'])
            meta['capture_monotonic'] -= 10.
            write_packet(packet, [[0, 0, 2]], [[255, 0, 0]], meta)
            self.assertIsNone(preview_frame(run)['image'])


if __name__ == '__main__':
    unittest.main()
