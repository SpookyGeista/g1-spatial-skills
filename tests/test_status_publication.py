import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from run_mission import LioMission,old


class StatusPublicationTest(unittest.TestCase):
    def test_one_publication_contains_source_attribution_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            mission=LioMission.__new__(LioMission)
            mission.output=Path(folder);mission.phase='OPERATOR_CONTROL';mission.owner='external POSE / SONIC'
            mission.detail='test';mission.started=time.monotonic();mission.detector=SimpleNamespace(confirmed=None)
            mission.sensor_count=10;mission.grid=SimpleNamespace(seen=np.zeros((2,2),bool))
            mission.args=SimpleNamespace(test='explore')
            (mission.output/'mission_review_replay').mkdir()
            (mission.output/'mission_review_replay/source.json').write_text('{}')
            with patch.object(old,'atomic_json') as publish:
                mission.heartbeat()
            self.assertEqual(publish.call_count,1)
            status=publish.call_args.args[1]
            self.assertEqual(status['pose_input_kind'],'recorded_pose_replay')
            self.assertEqual(status['lidar_rays_per_second'],10000)
            self.assertEqual(status['state'],'OPERATOR_CONTROL')
            self.assertFalse(status['detected'])

    def test_gui_adapter_changes_only_declared_attribution_literals(self):
        from punktnull import gui
        restored=gui._source
        for before,after in reversed(gui._replacements):restored=restored.replace(after,before)
        self.assertEqual(restored,gui._original)


if __name__=='__main__':unittest.main()
