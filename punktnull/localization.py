"""Estimated pose + measured waist encoders; no continuous simulator localization.

An explicit one-time initial-frame registration is allowed by this prototype.
Thereafter FAST-LIO drift is preserved. Invalid/stale estimates fail closed.
"""
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .sensors.mid360 import SensorConfig,rotation_rpy,rotation_wxyz

def transform(position,quaternion_wxyz):
    position=np.asarray(position,float)
    if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('Invalid estimated translation')
    t=np.eye(4);t[:3,:3]=rotation_wxyz(quaternion_wxyz);t[:3,3]=position
    return t

class RobotFrames:
    def __init__(self,urdf):
        root=ET.parse(urdf).getroot();self.joints=[]
        parent='pelvis'
        for name in ('waist_yaw_joint','waist_roll_joint','waist_pitch_joint'):
            j=root.find("joint[@name='"+name+"']")
            if j is None or j.find('parent').get('link')!=parent:raise ValueError('Unexpected G1 waist chain')
            parent=j.find('child').get('link');o=j.find('origin')
            fixed=np.eye(4);fixed[:3,:3]=rotation_rpy([float(x) for x in o.get('rpy','0 0 0').split()])
            fixed[:3,3]=[float(x) for x in o.get('xyz','0 0 0').split()]
            axis=np.asarray([float(x) for x in j.find('axis').get('xyz').split()])
            self.joints.append((fixed,axis))
        if parent!='torso_link':raise ValueError('Unexpected torso link')
        c=SensorConfig();self.T_TI=c.T_torso_lidar@c.T_lidar_imu

    def pelvis_to_torso(self,waist_q):
        if len(waist_q)!=3 or not np.isfinite(waist_q).all():raise ValueError('Need three finite waist encoder readings')
        t=np.eye(4)
        for q,(fixed,axis) in zip(waist_q,self.joints):
            joint=np.eye(4);joint[:3,:3]=Rotation.from_rotvec(float(q)*axis).as_matrix()
            t=t@fixed@joint
        return t

    def pelvis(self,T_WI,waist_q):
        return T_WI@np.linalg.inv(self.T_TI)@np.linalg.inv(self.pelvis_to_torso(waist_q))

class PoseHistory:
    def __init__(self,T_map_initial):
        self.registration=np.asarray(T_map_initial,float)
        if self.registration.shape!=(4,4) or not np.isfinite(self.registration).all():raise ValueError('Bad initial frame registration')
        r=self.registration[:3,:3]
        if not np.allclose(r.T@r,np.eye(3),atol=1e-6) or abs(np.linalg.det(r)-1)>1e-6 or not np.allclose(self.registration[3],[0,0,0,1]):
            raise ValueError('Initial registration must be a proper rigid transform')
        self.records=[]

    def append(self,record):
        if 'rejected' in record:return
        stamp=int(record['stamp_ns'])
        t=transform(record['position_m'],record['quaternion_wxyz'])
        if self.records and stamp<=self.records[-1][0]:return
        self.records.append((stamp,t));self.records=self.records[-1200:]

    def brackets(self,stamp_ns):
        return bool(self.records and self.records[0][0]<=stamp_ns<=self.records[-1][0])

    def at(self,stamp_ns,max_age_ns=150_000_000,*,require_bracket=False):
        if not self.records:raise RuntimeError('FAST-LIO not initialized')
        if require_bracket and not self.brackets(stamp_ns):
            raise RuntimeError('FAST-LIO does not yet bracket this sensor observation')
        stamp_ns=int(stamp_ns);times=np.array([r[0] for r in self.records],dtype=np.int64)
        i=int(np.searchsorted(times,stamp_ns))
        if i==0 or i==len(times):
            k=0 if i==0 else len(times)-1
            if abs(stamp_ns-times[k])>max_age_ns:raise RuntimeError('FAST-LIO pose unavailable/stale at observation time')
            return self.registration@self.records[k][1],int(abs(stamp_ns-times[k]))
        a,b=self.records[i-1],self.records[i];f=(stamp_ns-a[0])/(b[0]-a[0])
        if b[0]-a[0]>2*max_age_ns:raise RuntimeError('FAST-LIO interpolation gap')
        t=np.eye(4);t[:3,3]=(1-f)*a[1][:3,3]+f*b[1][:3,3]
        rotations=Rotation.from_matrix(np.stack([a[1][:3,:3],b[1][:3,:3]]))
        t[:3,:3]=Slerp([0,1],rotations)(f).as_matrix()
        return self.registration@t,0
