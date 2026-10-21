"""CPU fixture preflight tests. These do not execute the integrated robot."""
import copy
import math
import unittest
import numpy as np
from punktnull.terrain import inspection_passage as scene
from tools.check_inspection_passage import transformed_envelope,all_variant_boxes,scan_clearance

class InspectionPassageGeometry(unittest.TestCase):
    def test_real_roof_covers_opening_and_shoring_has_no_floor_lip(self):
        boxes={b['prim'].removeprefix(scene.BASE+'/'):b for b in scene.collision_boxes() if b['prim'].startswith(scene.BASE+'/')}
        self.assertEqual(len(boxes),9)
        roof=boxes['Duct/SolidHousing']
        np.testing.assert_allclose([roof['aabb_min'][1],roof['aabb_max'][1]],scene.ROOF_Y_M)
        self.assertAlmostEqual(roof['aabb_min'][2],.65)
        self.assertLess(roof['aabb_min'][0],-scene.CLEAR_WIDTH_M/2)
        self.assertGreater(roof['aabb_max'][0],scene.CLEAR_WIDTH_M/2)
        for name,box in boxes.items():
            if 'Shoring/' in name:
                inner=box['aabb_max'][0] if 'West' in name else box['aabb_min'][0]
                self.assertAlmostEqual(abs(inner),1.1)
        door=next(d for d in scene.load_layout()['doorways'] if d['id']=='north')
        self.assertEqual(door['clear_width_m'],2.2)
        self.assertEqual(door['structural_clear_height_m'],2.2)
        self.assertTrue(door['upright_blocked'])

    def test_only_declared_original_colliders_change_all_eighteen_conditions(self):
        changes=set(scene._relocations())|set(scene._structure_changes())
        count=0
        for variant in (0,1,2):
            for room in ('west','east','north'):
                for present in (False,True):
                    cfg=scene.load_layout();cfg.update(variant=variant,target_room=room,target_present=present)
                    old={s['name']:s for s in scene._base().obstacle_specs(cfg)}
                    new={s['name']:{k:v for k,v in s.items() if k!='prim'} for s in scene.obstacle_specs(cfg)}
                    self.assertEqual({k for k in old if old[k]!=new[k]},changes)
                    self.assertEqual(len(new)-len(old),9)
                    self.assertEqual(cfg['floor'],scene._base().load_layout()['floor'])
                    # Workbench and case are physically distinct in every variation.
                    b=scene.collision_boxes(cfg)
                    bench=[x for x in b if any(n in x['prim'] for n in ('Workbench','BenchLeg'))]
                    targets=[x for x in b if '/InspectionCase/' in x['prim']]
                    self.assertEqual(len(targets),4 if present else 0)
                    for x in bench:
                        for y in targets:
                            overlap=np.minimum(x['aabb_max'],y['aabb_max'])-np.maximum(x['aabb_min'],y['aabb_min'])
                            self.assertFalse(np.all(overlap>1e-9))
                    count+=1
        self.assertEqual(count,18)

    def test_contract_is_deepcopied_and_never_claims_skill_acceptance(self):
        cfg=scene.load_layout();cfg['inspection_passage']['mission_contract']['ingress']['approach_xy_m'][0]=999
        self.assertNotEqual(scene.MISSION_CONTRACT['ingress']['approach_xy_m'][0],999)
        self.assertFalse(scene.MISSION_CONTRACT['skill_admitted'])
        self.assertEqual(scene.MISSION_CONTRACT['minimum_margin_m'],.08)
        self.assertEqual(scene.MISSION_CONTRACT['ingress']['entry_tolerance_m'],.10)
        self.assertEqual(scene.MISSION_CONTRACT['egress']['heading_tolerance_deg'],3.)
        self.assertTrue(scene.MISSION_CONTRACT['two_passes_required'])

    def test_uncertainty_contains_interior_angular_extremum_and_translation(self):
        # A point crosses the +Y extremum at90deg INSIDE a 70..100deg interval.
        bounds=np.array([[[[2.,0.,.2],[2.,0.,.3]]]])
        entry={'approach_xy_m':[1.,-1.],'heading_deg':85.}
        b=transformed_envelope(bounds,entry,.1,15.)[0,0]
        self.assertAlmostEqual(b[1,1],1.1)
        for angle in np.linspace(70,100,37):
            p=np.array([2*math.cos(math.radians(angle))+1,2*math.sin(math.radians(angle))-1])
            for dx in (-.1,.1):
                for dy in (-.1,.1):
                    self.assertTrue(np.all(p+[dx,dy]>=b[0,:2]-1e-12))
                    self.assertTrue(np.all(p+[dx,dy]<=b[1,:2]+1e-12))
        np.testing.assert_allclose(b[:,2],[.2,.3])

    def test_north_scan_keeps_roof_occupied_and_one_metre_reserve(self):
        boxes,variants=all_variant_boxes();check=scan_clearance(boxes)
        self.assertEqual(len(variants),18)
        self.assertTrue(check['roof_included_in_occupied_projection'])
        self.assertGreaterEqual(check['ideal_truth_grid_minimum_occupied_clearance_within_010m_plus_cell_halfdiag_m'],1.)
        self.assertGreater(check['nominal_aabb_point_clearance_m'],1.)

if __name__=='__main__':unittest.main()
