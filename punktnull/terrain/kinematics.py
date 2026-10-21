"""Actual G1 URDF FK and bounds using the imported cylinder→capsule setting.

These are conservative geometric bounds, not a simulated contact-success test.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import struct
import re
import numpy as np
from scipy.spatial.transform import Rotation
from punktnull.sensors.mid360 import rotation_rpy

def origin(element):
    t=np.eye(4)
    if element is not None:
        t[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=rotation_rpy(np.fromstring(element.get('rpy','0 0 0'),sep=' '))
    return t

def stl_bounds(path,scale):
    raw=Path(path).read_bytes()
    n=struct.unpack('<I',raw[80:84])[0] if len(raw)>=84 else 0
    if n and len(raw)==84+n*50:
        dtype=np.dtype(dict(names=['v'],formats=[('<f4',(3,3))],offsets=[12],itemsize=50))
        v=np.frombuffer(raw,dtype=dtype,offset=84)['v'].reshape(-1,3).astype(float)
    else:
        v=np.array([[float(x) for x in row.split()] for row in re.findall(r'vertex\s+([^\r\n]+)',raw.decode())])
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():raise ValueError('Invalid STL '+str(path))
    v*=scale;low,high=v.min(0),v.max(0)
    return (low+high)/2,(high-low)/2

class CollisionFK:
    def __init__(self,path,joint_names):
        root=ET.parse(path).getroot();self.names=tuple(joint_names);self.joints=[];self.collisions=[]
        for j in root.findall('joint'):
            kind=j.get('type');name=j.get('name')
            if kind not in ('fixed','revolute','continuous'):raise ValueError('Unsupported joint '+name)
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ') if kind!='fixed' else np.zeros(3)
            self.joints.append(dict(name=name,kind=kind,parent=j.find('parent').get('link'),child=j.find('child').get('link'),
                                   origin=origin(j.find('origin')),axis=axis))
        for link in root.findall('link'):
            for i,c in enumerate(link.findall('collision')):
                g=list(c.find('geometry'))[0];kind=g.tag
                if kind=='cylinder':size=np.array([float(g.get('radius')),float(g.get('length'))])
                elif kind=='sphere':size=np.array([float(g.get('radius'))])
                elif kind=='box':size=np.fromstring(g.get('size'),sep=' ')
                elif kind=='mesh':
                    name=g.get('filename')
                    if not name.startswith('package://robot_description/'):raise ValueError('Unknown mesh source '+name)
                    mesh=Path(path).parents[2]/name.removeprefix('package://robot_description/')
                    center,half=stl_bounds(mesh,np.fromstring(g.get('scale','1 1 1'),sep=' '));size=np.r_[center,half]
                else:raise ValueError('Unmodelled collision '+kind+' on '+link.get('name'))
                self.collisions.append(dict(link=link.get('name'),index=i,kind=kind,size=size,origin=origin(c.find('origin'))))

    def links(self,xyz,quaternion_wxyz,q):
        xyz=np.asarray(xyz);quat=np.asarray(quaternion_wxyz);q=np.asarray(q);n=len(xyz)
        if xyz.shape!=(n,3) or quat.shape!=(n,4) or q.shape!=(n,len(self.names)) or not np.isfinite(np.c_[xyz,quat,q]).all():
            raise ValueError('Invalid measured/reference state shapes')
        t=np.broadcast_to(np.eye(4),(n,4,4)).copy();t[:,:3,3]=xyz
        t[:,:3,:3]=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
        poses={'pelvis':t};remaining=self.joints.copy()
        while remaining:
            progress=False
            for j in remaining.copy():
                if j['parent'] not in poses:continue
                motion=np.broadcast_to(np.eye(4),(n,4,4)).copy()
                if j['kind']!='fixed':motion[:,:3,:3]=Rotation.from_rotvec(q[:,self.names.index(j['name']),None]*j['axis']).as_matrix()
                poses[j['child']]=poses[j['parent']]@j['origin']@motion
                remaining.remove(j);progress=True
            if not progress:raise ValueError('URDF graph does not connect to pelvis')
        return poses

    def bounds(self,poses):
        result=[]
        for c in self.collisions:
            t=poses[c['link']]@c['origin'];r=t[:,:3,:3];center=t[:,:3,3]
            if c['kind']=='cylinder':half=c['size'][0]+np.abs(r[:,:,2])*(c['size'][1]/2)
            elif c['kind']=='sphere':half=np.full_like(center,c['size'][0])
            elif c['kind']=='mesh':
                center=center+np.einsum('nij,j->ni',r,c['size'][:3]);half=np.einsum('nij,j->ni',np.abs(r),c['size'][3:])
            else:half=np.einsum('nij,j->ni',np.abs(r),c['size']/2)
            result.append(np.stack([center-half,center+half],axis=1))
        return np.stack(result,axis=1)
