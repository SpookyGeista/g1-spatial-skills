"""CPU adapter contracts, not rendered camera QA."""
import unittest
import numpy as np
from punktnull.skills.runtime import FixedSkillSpectator,skill_camera_profile


class CameraTests(unittest.TestCase):
    def test_passage_defaults_and_explicit_opt_out(self):
        scene={'fixture_contract':{'id':'service_passage_p01_v1'}}
        side=skill_camera_profile(scene,{})
        self.assertEqual(side['mode'],'side')
        self.assertEqual(side['eye'],[1.6,-5.8,1.1])
        self.assertEqual(skill_camera_profile({}, {})['mode'],'chase')
        self.assertEqual(skill_camera_profile(scene,{'PUNKTNULL_SKILL_CAMERA':'chase'})['mode'],'chase')

    def test_explicit_configuration_and_invalid_geometry(self):
        side=skill_camera_profile({}, {'PUNKTNULL_SKILL_CAMERA':'side','PUNKTNULL_SKILL_CAMERA_EYE':'1,-5,1.2','PUNKTNULL_SKILL_CAMERA_WIDTH_M':'6.1'})
        self.assertEqual(side['eye'],[1.,-5.,1.2]);self.assertEqual(side['orthographic_width_m'],6.1)
        for env in ({'PUNKTNULL_SKILL_CAMERA':'cinematic'},
                    {'PUNKTNULL_SKILL_CAMERA':'side','PUNKTNULL_SKILL_CAMERA_EYE':'1,2'},
                    {'PUNKTNULL_SKILL_CAMERA':'side','PUNKTNULL_SKILL_CAMERA_EYE':'nan,0,1'},
                    {'PUNKTNULL_SKILL_CAMERA':'side','PUNKTNULL_SKILL_CAMERA_EYE':'1.6,0,1.1'},
                    {'PUNKTNULL_SKILL_CAMERA':'side','PUNKTNULL_SKILL_CAMERA_WIDTH_M':'0'}):
            with self.assertRaises(ValueError):skill_camera_profile({},env)

    def test_first_camera_preserved_and_fixed_pose_logged(self):
        class Retained:
            def __init__(self):
                self.first={'eye':np.array([9.,8.,7.]),'target':np.array([2.,3.,4.]),'up':np.array([0.,0.,1.])}
                self.calls=[]
            def update(self,*args):
                self.calls.append(args)
                return {'first':self.first,'chase':{'eye':np.array([8.,8.,8.])},'clip':{'fraction':.1,'blocked_by':'roof'}}
        old=Retained();profile=skill_camera_profile({}, {'PUNKTNULL_SKILL_CAMERA':'side'});adapter=FixedSkillSpectator(old,profile)
        root=np.arange(7.);torso=np.arange(7.)+10;view=adapter.update(root,torso,.1)
        self.assertIs(view['first'],old.first)
        self.assertIs(old.calls[0][0],root);self.assertIs(old.calls[0][1],torso)
        np.testing.assert_array_equal(view['chase']['eye'],profile['eye'])
        self.assertEqual(view['clip']['retained_follow_diagnostic_not_applied']['blocked_by'],'roof')
        self.assertIsNone(view['clip']['unoccluded'])
        self.assertFalse(view['clip']['full_mesh_visibility_verified'])
        view['chase']['eye'][0]=99
        np.testing.assert_array_equal(adapter.update(root,torso,.1)['chase']['eye'],profile['eye'])

if __name__=='__main__':unittest.main()
