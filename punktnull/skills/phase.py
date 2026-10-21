"""Retained error/ground-phase calculation on an already validated ReferenceClip."""
import numpy as np
from .legacy import ground_module

class SkillReference:
    def __init__(self,clip):
        old=ground_module();self.source=clip.source
        self.xyz=clip.root;self.quat=old.normalize(clip.quaternion)
        self.joints=clip.data['joint_pos'][:,old.MJ_TO_IL]
        self.frames=clip.frames;self.tilt=old.tilt_degrees(self.quat)
        if self.xyz[0,2]<.60 or self.xyz[-1,2]<.60 or self.tilt[0]>25 or self.tilt[-1]>25:
            raise ValueError('Skill reference requires upright start and finish')
        self.ground=(self.xyz[:,2]<.45)|(self.tilt>45)
        self.permission=self.ground.copy()
        for offset in range(1,11):
            self.permission[offset:] |= self.ground[:-offset]
            self.permission[:-offset] |= self.ground[offset:]

    def errors(self,frame,xyz,quat,joints,heading):
        old=ground_module()
        return (float(abs(xyz[2]-self.xyz[frame,2])),
                float(old.quaternion_error_degrees(quat,old.rotate_heading(self.quat[frame],heading))),
                float(np.sqrt(np.mean((np.asarray(joints)-self.joints[frame])**2))))
