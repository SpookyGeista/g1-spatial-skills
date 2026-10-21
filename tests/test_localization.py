import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from punktnull.localization import PoseHistory,RobotFrames
from punktnull.retained import URDF

class LocalizationTest(unittest.TestCase):
    def test_inverted_sensor_recovers_pelvis_from_encoders(self):
        frames=RobotFrames(URDF);q=[.2,-.1,.15]
        base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',1.1).as_matrix();base[:3,3]=[2,3,.8]
        imu=base@frames.pelvis_to_torso(q)@frames.T_TI
        np.testing.assert_allclose(frames.pelvis(imu,q),base,atol=1e-12)
    def test_estimator_drift_is_preserved_no_truth_reanchor(self):
        hist=PoseHistory(np.eye(4))
        for i in range(3):hist.append(dict(stamp_ns=i*100_000_000,position_m=[i*.1,0,0],quaternion_wxyz=[1,0,0,0]))
        t,age=hist.at(150_000_000);self.assertAlmostEqual(t[0,3],.15);self.assertEqual(age,0)
        with self.assertRaises(RuntimeError):hist.at(1_000_000_000)
    def test_bad_frame_and_quaternion_fail(self):
        with self.assertRaises(ValueError):PoseHistory(np.zeros((3,3)))
        hist=PoseHistory(np.eye(4))
        with self.assertRaises(ValueError):hist.append(dict(stamp_ns=0,position_m=[0,0,0],quaternion_wxyz=[0,0,0,0]))
    def test_camera_waits_for_future_estimate_instead_of_old_turn_pose(self):
        hist=PoseHistory(np.eye(4))
        hist.append(dict(stamp_ns=0,position_m=[0,0,0],quaternion_wxyz=[1,0,0,0]))
        with self.assertRaises(RuntimeError):hist.at(50_000_000,require_bracket=True)
        q=Rotation.from_euler('z',.1).as_quat()[[3,0,1,2]]
        hist.append(dict(stamp_ns=100_000_000,position_m=[.02,0,0],quaternion_wxyz=q))
        pose,age=hist.at(50_000_000,require_bracket=True)
        self.assertEqual(age,0);self.assertAlmostEqual(pose[0,3],.01)
        np.testing.assert_allclose(pose[:3,:3],Rotation.from_euler('z',.05).as_matrix(),atol=1e-12)

if __name__=='__main__':unittest.main()
