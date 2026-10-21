import json
import socket
import struct
import unittest
import numpy as np
from punktnull.sensor_transport import encode,decode,receive

class SensorTransportTest(unittest.TestCase):
    def imu(self):return dict(kind='imu',stamp_ns=5_000_000,angular_velocity_rad_s=[0,0,.2],specific_force_m_s2=[0,0,-9.81])
    def cloud(self):return dict(kind='cloud',scanstart_ns=100_000_000,scan_end_ns=200_000_000,
        xyz_sensor_m=np.array([[1,2,3],[2,3,4]],np.float32),pointdt_ns=np.array([0,95_000_000],np.uint32),
        intensity=np.array([1,2],np.float32),line=np.array([0,0],np.uint8))
    def test_cloud_roundtrip_preserves_discrete_times_and_frame(self):
        event=self.cloud();out=decode(encode(event)[4:])
        np.testing.assert_array_equal(out['xyz_sensor_m'],event['xyz_sensor_m'])
        np.testing.assert_array_equal(out['pointdt_ns'],event['pointdt_ns'])
    def test_no_truth_pose_or_orientation_can_enter_estimator_packet(self):
        event=self.imu();event.update(truth_pose=[1,2,3],orientation=[1,0,0,0])
        out=decode(encode(event)[4:]);self.assertNotIn('truth_pose',out);self.assertNotIn('orientation',out)
    def test_invalid_samples_are_rejected(self):
        event=self.imu();event['specific_force_m_s2'][0]=float('nan')
        with self.assertRaises(ValueError):encode(event)
        event=self.cloud();event['pointdt_ns'][1]=100_000_000
        with self.assertRaises(ValueError):encode(event)
    def test_socket_stream_keeps_back_to_back_frames_separate(self):
        a,b=socket.socketpair()
        try:
            a.sendall(encode(self.imu())+encode(self.cloud()))
            self.assertEqual(receive(b)['kind'],'imu');self.assertEqual(receive(b)['kind'],'cloud')
        finally:a.close();b.close()
    def test_hostile_length_rejected_before_allocation(self):
        a,b=socket.socketpair()
        try:
            a.sendall(struct.pack('!I',2**32-1))
            with self.assertRaises(ValueError):receive(b)
        finally:a.close();b.close()

if __name__=='__main__':unittest.main()
