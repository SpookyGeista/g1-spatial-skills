"""Geometry contract checks; explicitly not a simulated passage test."""
import unittest
import numpy as np
from punktnull.terrain.service_passage import FIXTURE_CONTRACT, collision_boxes, load_layout, obstacle_specs

class TestServicePassage(unittest.TestCase):
    def test_clearance_and_real_supports(self):
        boxes={b['prim'].split('/ServicePassage/')[1]:b for b in collision_boxes()}
        roof=boxes['Duct/SolidHousing']
        self.assertAlmostEqual(roof['aabb_min'][2],.65)
        np.testing.assert_allclose([roof['aabb_min'][0],roof['aabb_max'][0]],[1.2,2.0])
        self.assertEqual(len([k for k in boxes if 'Post' in k]),4)
        for name,box in boxes.items():
            if 'SouthPost' in name:self.assertAlmostEqual(box['aabb_max'][1],-.775)
            if 'NorthPost' in name:self.assertAlmostEqual(box['aabb_min'][1],.775)
            if 'Foot' in name:self.assertGreaterEqual(min(abs(box['aabb_min'][1]),abs(box['aabb_max'][1])),.775-1e-12)
        self.assertFalse(FIXTURE_CONTRACT['physical_passage_validated'])

    def test_geometry_records_match_build_specs(self):
        for spec,box in zip(obstacle_specs(),collision_boxes()):
            np.testing.assert_allclose(np.subtract(box['aabb_max'],box['aabb_min']),spec['size'])
            self.assertTrue(all(s>0 for s in spec['size']))
        self.assertEqual(len({b['prim'] for b in collision_boxes()}),len(collision_boxes()))

    def test_layout_camera_and_floor_unchanged(self):
        cfg=load_layout()
        self.assertEqual(cfg['floor'],{'center':[2.6,.25,-.10],'size':[9.8,6.1,.20]})
        self.assertEqual(cfg['robot_start_xy_m'],[0.,0.])
        self.assertEqual(cfg['lighting'],{'dome_intensity':560,'key_intensity':1800})
        self.assertIn('TopDown',cfg['cameras'])
        cfg['fixture_contract']['clear_width_m']=999
        self.assertEqual(FIXTURE_CONTRACT['clear_width_m'],1.55)

if __name__=='__main__':unittest.main()
