"""Rendering-only camera selection/projection restore and ray diagnostics."""
import unittest
import numpy as np
from punktnull.terrain.mission_camera import MissionSkillSpectator,assess_points,PROFILES

class Retained:
 def __init__(self):self.calls=0
 def update(self,root,dt):
  self.calls+=1
  return {'eye':np.array([9.,9.,9.]),'target':np.zeros(3),'up':np.array([0.,0.,1.])},{'retained':True}

class MissionCameraTests(unittest.TestCase):
 def test_measured_height_gate_and_restoration_without_sensor_access(self):
  status={'state':'SKILL_GROUND','autonomous_skill':{'direction':'egress'}};calls=[];old=Retained()
  camera=MissionSkillSpectator(old,[],lambda:status,calls.append)
  root=np.array([0.,2.2,.78,1.,0.,0.,0.])
  _,meta=camera.update(root,.1);self.assertEqual(meta['mission_skill_camera'],'retained')
  root[2]=.25;pose,meta=camera.update(root,.1)
  self.assertEqual(calls,[18.]);self.assertEqual(meta['mode'],'mission_low_portal')
  self.assertFalse(meta['full_mesh_visibility_verified']);np.testing.assert_array_equal(pose['eye'],PROFILES['egress']['eye'])
  root[2]=.55;camera.update(root,.1);self.assertEqual(calls,[18.])
  root[2]=.61;_,meta=camera.update(root,.1);self.assertEqual(calls,[18.,24.])
  self.assertEqual(meta['mission_skill_camera'],'retained');self.assertEqual(old.calls,4)

 def test_unknown_gate_or_mission_phase_never_selects_profile(self):
  for status in ({'state':'EXPLORING','autonomous_skill':{'direction':'ingress'}},
                 {'state':'SKILL_GROUND','autonomous_skill':None},
                 {'state':'SKILL_GROUND','autonomous_skill':{'direction':'stairs'}}):
   camera=MissionSkillSpectator(Retained(),[],lambda:status)
   _,meta=camera.update([0.,0.,.2,1.,0.,0.,0.],.1)
   self.assertEqual(meta['mission_skill_camera'],'retained')

 def test_real_opaque_box_and_off_frame_point_are_reported(self):
  box={'prim':'opaque_roof','aabb_min':[-1.,.4,0.],'aabb_max':[1.,.6,1.]}
  check=assess_points([0.,0.,.5],[0.,1.,.5],[0.,0.,1.],[[0.,1.,.5],[10.,1.,.5]],[box])
  self.assertTrue(check['occluded']);self.assertFalse(check['full_proxy_in_frame'])
  self.assertEqual(check['blocked_by'],['opaque_roof'])

if __name__=='__main__':unittest.main()
