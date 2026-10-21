import unittest
from types import SimpleNamespace
import numpy as np
from punktnull.sensors.mid360 import PhysxRaycaster

class OpticalWindowTests(unittest.TestCase):
    def test_only_sensor_housing_excluded_other_robot_still_occludes(self):
        backend=PhysxRaycaster.__new__(PhysxRaycaster)
        backend.housing_path='/robot/sensor_housing';backend.ignored_housing_hits=0;backend.accepted_hits={}
        def query(origin,direction,maximum,callback):
            for path,distance in [('/wall',4),('/robot/sensor_housing',0),('/robot/hand',.4)]:
                callback(SimpleNamespace(collision=path,distance=distance))
        backend.query=SimpleNamespace(raycast_all=query)
        distances=backend.cast([0,0,0],[[1,0,0]],40)
        np.testing.assert_allclose(distances,[.4]);self.assertEqual(backend.ignored_housing_hits,1)
        self.assertEqual(backend.accepted_hits,{'/robot/hand':1})

if __name__=='__main__':unittest.main()
