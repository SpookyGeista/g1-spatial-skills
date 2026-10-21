import unittest
import numpy as np
from punktnull.inspection import InspectionExplorer,LioOccupancyGrid
from navigation import GridConfig


class InspectionTest(unittest.TestCase):
    def setUp(self):
        self.grid=LioOccupancyGrid(GridConfig())
        self.explorer=InspectionExplorer(self.grid,[0,0])
        # Explicit synthetic observed-free grid for progression unit tests.
        # This is not a sensor or controller integration test.
        self.grid.last_update=1.
        size=self.grid.layers()['safe'].shape
        self.grid.layers=lambda:{'safe':np.ones(size,bool),'clearance':np.ones(size)*2}

    def arrive(self,index,now=1.):
        self.explorer.last_position=None
        xy=self.explorer.inspection_plan['rooms'][index]['observation_xy_m']
        command=self.explorer.step(xy,0,now,pose_time=now)
        self.assertEqual(command.status,'SCAN_REQUIRED')
        return xy

    def test_arrival_alone_never_completes_room_and_return_preserves_completion(self):
        self.explorer.finish_scan()
        for i in range(3):
            self.arrive(i)
            self.assertEqual(len(self.explorer.completed_rooms),i)
            self.explorer.finish_scan()
        self.assertEqual(self.explorer.step([0,3.25],0,1).status,'EXPLORATION_EXHAUSTED')
        self.explorer.return_to_start();self.explorer.last_position=None
        self.assertEqual(self.explorer.step([0,0],0,1).status,'RETURNED')
        self.assertEqual(len(self.explorer.completed_rooms),3)

    def test_room_cannot_be_credited_outside_observation_zone(self):
        self.explorer.finish_scan();self.arrive(0)
        self.explorer.scan_origin=np.array([0.,0.])
        with self.assertRaises(RuntimeError):self.explorer.finish_scan()
        self.assertEqual(self.explorer.completed_rooms,[])

    def test_post_stop_drift_replans_same_room_without_credit(self):
        self.explorer.finish_scan();self.arrive(0)
        result=self.explorer.admit_settled_room_scan([-2.10+.185927,-.10],2.)
        self.assertFalse(result['admitted'])
        self.assertEqual(result['arrival']['distance_m'],0.)
        self.assertFalse(self.explorer.scan_pending)
        self.assertEqual(self.explorer.room_index,0)
        self.assertEqual(self.explorer.completed_rooms,[])
        self.grid.last_update=2.
        self.arrive(0,now=2.)
        self.assertTrue(self.explorer.admit_settled_room_scan([-2.05,-.10],2.)['admitted'])
        self.explorer.scan_origin=np.array([-2.05,-.10]);self.explorer.finish_scan()
        self.assertTrue(self.explorer.completed_rooms[0]['settled_admission']['admitted'])

    def test_post_stop_retries_are_bounded_and_require_assistance(self):
        self.explorer.finish_scan()
        for attempt in range(3):
            self.arrive(0)
            self.explorer.admit_settled_room_scan([-1.90,-.10],1.)
        self.assertEqual(self.explorer.step([-1.90,-.10],0,1.).status,'INSPECTION_SCAN_BLOCKED')
        self.assertEqual(self.explorer.completed_rooms,[])

    def test_finish_rechecks_strict_scan_origin_even_if_admission_bypassed(self):
        self.explorer.finish_scan();self.arrive(0)
        self.explorer.scan_origin=np.array([-2.10+.181,-.10])
        with self.assertRaises(RuntimeError):self.explorer.finish_scan()

    def test_rejected_scan_requires_recovery_without_skipping_room(self):
        self.explorer.finish_scan();self.arrive(0)
        self.explorer.reject_scan([-2.1,-.1],1,'Insufficient turning clearance')
        self.assertEqual(self.explorer.step([-2.1,-.1],0,1).status,'INSPECTION_SCAN_BLOCKED')
        self.assertTrue(self.explorer.reanchor_from_handover([0,0],1))
        self.explorer.scan_pending=True;self.explorer.scan_origin=np.array([0.,0.])
        self.explorer.finish_scan()
        self.assertEqual(self.explorer.room_index,0)
        self.assertEqual(self.explorer.completed_rooms,[])
        self.arrive(0);self.explorer.finish_scan()
        self.assertEqual(self.explorer.room_index,1)

    def test_unrequested_scan_and_stale_handover_rejected(self):
        self.explorer.finish_scan()
        with self.assertRaises(RuntimeError):self.explorer.finish_scan()
        self.assertFalse(self.explorer.reanchor_from_handover([0,0],3))

    def test_injected_review_needs_checked_handover_and_preserves_next_room(self):
        self.explorer.inject_review_after_room='west'
        self.explorer.finish_scan();self.arrive(0);self.explorer.finish_scan()
        self.assertEqual(self.explorer.step([-2.1,-.1],0,1).status,'OPERATOR_REVIEW_REQUIRED')
        self.assertFalse(self.explorer.reanchor_from_handover([-2.1,-.1],3))
        self.assertFalse(self.explorer.injected_review_resolved)
        self.assertTrue(self.explorer.reanchor_from_handover([-2.1,-.1],1))
        self.assertTrue(self.explorer.injected_review_resolved)
        self.assertEqual(self.explorer.room_index,1)
        self.assertEqual([r['id'] for r in self.explorer.completed_rooms],['west'])

if __name__=='__main__':unittest.main()
