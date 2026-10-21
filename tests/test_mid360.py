"""CPU analytic backend tests; these are NOT Isaac integration tests."""
import unittest
import numpy as np
from punktnull.sensors.mid360 import SensorConfig,SimMid360,rotation_wxyz,read_mid360_mount

class Plane:
    def cast(self,origin,directions,maximum_range_m):
        # Infinite horizontal floor z=0, independently solved in world coordinates.
        with np.errstate(divide='ignore',invalid='ignore'):
            d=-origin[2]/directions[:,2]
        return np.where((d>0)&(d<=maximum_range_m),d,np.nan)

class SensorTests(unittest.TestCase):
    def step(self,s,t,v=(0,0,0),w=(0,0,0),q=(1,0,0,0)):
        return s.update_physics(t,[0,0,1,*q],v,w)
    def test_stationary_specific_force_inverted(self):
        s=SimMid360(Plane());self.step(s,0);r=self.step(s,.005)['imu']
        expected=s.config.T_torso_lidar[:3,:3].T@np.array([0,0,9.81])
        np.testing.assert_allclose(r['specific_force_m_s2'],expected,atol=1e-10)
        self.assertLess(r['specific_force_m_s2'][2],-9.7)
        self.assertNotIn('orientation',r);self.assertNotIn('position',r)
    def test_acceleration_and_angular_lever(self):
        s=SimMid360(Plane());self.step(s,0,w=(0,0,1));r=self.step(s,.005,v=(.005,0,0),w=(0,0,1))['imu']
        T=s.config.T_torso_lidar@s.config.T_lidar_imu
        a=np.array([1.,0,0])+np.cross([0,0,1],np.cross([0,0,1],T[:3,3]))
        np.testing.assert_allclose(r['specific_force_m_s2'],T[:3,:3].T@(a+[0,0,9.81]),atol=1e-10)
        np.testing.assert_allclose(r['angular_velocity_rad_s'],T[:3,:3].T@[0,0,1])
    def test_scan_clock_and_geometry(self):
        s=SimMid360(Plane());packets=[]
        for i in range(21):packets+=self.step(s,i*.005)['scans']
        self.assertEqual(len(packets),1);p=packets[0]
        self.assertTrue(p['complete']);self.assertEqual(p['rays_queried'],2000)
        self.assertEqual(p['scanstart_ns'],0);self.assertTrue(np.all(p['pointdt_ns']<100000000))
        self.assertTrue(np.all(p['pointdt_ns']%5000000==0))
        world=p['xyz_sensor_m']@s.R_TL.T+np.array(s.config.torso_to_lidar_xyz_m)+[0,0,1]
        np.testing.assert_allclose(world[:,2],0,atol=2e-6)
    def test_clock_faults_rejected(self):
        for stamp in (0,-.005,.01):
            s=SimMid360(Plane());self.step(s,0)
            with self.assertRaises(ValueError):self.step(s,stamp)
    def test_ranges_and_inputs(self):
        with self.assertRaises(ValueError):rotation_wxyz([0,0,0,0])
        with self.assertRaises(ValueError):SensorConfig(rays_per_second=123)
        with self.assertRaises(ValueError):self.step(SimMid360(Plane()),0,v=(float('nan'),0,0))
    def test_extrinsic_direction(self):
        c=SensorConfig();T=np.linalg.inv(c.T_lidar_imu)
        np.testing.assert_allclose(T[:3,3],[-.011,-.02329,.04412])
    def test_partial_first_scan_honest(self):
        s=SimMid360(Plane());p=[]
        for i in range(1,21):p+=self.step(s,i*.005)['scans']
        self.assertFalse(p[0]['complete']);self.assertEqual(p[0]['physics_ticks'],19)

if __name__=='__main__':unittest.main()
