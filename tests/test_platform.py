"""CPU contract checks only, not physical platform success."""
import unittest
from punktnull.skills.stairs import stable_landing
from punktnull.skills.platform import TOP,PlatformSession
from punktnull.skills.reference import ReferenceClip
from punktnull.skills.legacy import WORKSPACE
from punktnull.terrain.platform import collision_boxes,FIXTURE_CONTRACT
from tools.platform_trial import install
class PlatformTests(unittest.TestCase):
    def test_fixed_single_real_riser_contract(self):
        boxes=collision_boxes();self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0]['aabb_max'][2],.120)
        self.assertEqual(boxes[0]['aabb_min'][0],.270)
        self.assertGreater(boxes[0]['size'][0],2.)
    def test_support_height_is_separate_from_two_riser_contract(self):
        s=dict(base=[.5,0,.90,1,0,0,0,0,0,0,0,0,0],dq=[0]*29,
            support=False,command_mode='active',tilt_deg=0,stair_contacts=dict(fresh=True,both_on_landing=True))
        self.assertTrue(stable_landing(s,support_height=.120))
        self.assertFalse(stable_landing(s))
    def test_generated_single_step_stop_and_contract(self):
        c=ReferenceClip.load(WORKSPACE/'assets/platform_p12_reference_002')
        self.assertAlmostEqual(c.root[-1,2],.895,places=4)
        self.assertEqual(PlatformSession(c).minimum_displacement,.30)
    def test_exact_native_supervisor_reuse(self):install()
    def test_component_registry_does_not_admit_building_skill(self):
        import json
        from punktnull.skills.platform120_spec import PLATFORM120_SPEC
        from punktnull.skills.session import REGISTRY,SkillSession
        self.assertFalse(PLATFORM120_SPEC['autonomous_mission_admission'])
        self.assertEqual(REGISTRY['stairs'].status,'failed_two_riser_sim')
        for path in PLATFORM120_SPEC['evidence']:
            self.assertTrue(json.loads((WORKSPACE/path).read_text())['accepted'])
        with self.assertRaisesRegex(ValueError,'no executable validated backend'):
            SkillSession(ReferenceClip.load(PLATFORM120_SPEC['source']),skill='platform120',experimental=True)
if __name__=='__main__':unittest.main()
