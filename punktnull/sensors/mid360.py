"""PhysX scene-query MID360-like sensor. No estimator pose/orientation output.

Approximation: deterministic low-discrepancy angular sequence, collision surfaces,
no optical material response/noise, 5ms simultaneous ray batches, not Livox optics.
Sensor-local XYZ is expressed in the actual inverted mid360_link frame. Misses
and hits inside the near blind range are absent; geometry is never used to fill
unobserved space. This module imports Isaac only when creating PhysxRaycaster.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np


def rotation_wxyz(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > 1e-3:
        raise ValueError("Expected finite unit wxyz quaternion")
    w,x,y,z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def rotation_rpy(rpy):
    r,p,y = rpy
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def read_mid360_mount(urdf):
    """Read the actual imported URDF, refusing another sensor parent/config."""
    root = ET.parse(urdf).getroot()
    joints = [j for j in root.findall('joint') if j.find('child').get('link') == 'mid360_link']
    if len(joints) != 1 or joints[0].get('type') != 'fixed' or joints[0].find('parent').get('link') != 'torso_link':
        raise ValueError('Expected exactly one fixed torso_link -> mid360_link')
    origin=joints[0].find('origin')
    xyz=tuple(float(x) for x in origin.get('xyz').split())
    rpy=tuple(float(x) for x in origin.get('rpy').split())
    if len(xyz)!=3 or len(rpy)!=3:raise ValueError('Invalid mount')
    return xyz,rpy


@dataclass(frozen=True)
class SensorConfig:
    rays_per_second: int = 20000
    scan_hz: int = 10
    physics_dt_s: float = .005
    minimum_range_m: float = .1
    maximum_range_m: float = 40.
    elevation_min_deg: float = -7.
    elevation_max_deg: float = 52.
    torso_to_lidar_xyz_m: tuple = (.0002835,.00003,.41618)
    torso_to_lidar_rpy_rad: tuple = (0.,3.101,3.1415)
    # p_I = R_LI p_L + T_LI. R_LI=I in upstream mid360.yaml.
    # The sensor's IMU origin in LiDAR coordinates is therefore -T_LI.
    lidar_to_imu_xyz_m: tuple = (.011,.02329,-.04412)
    lidar_to_imu_rpy_rad: tuple = (0.,0.,0.)
    gravity_world_m_s2: tuple = (0.,0.,-9.81)

    @property
    def T_torso_lidar(self):
        """Homogeneous transform maps LiDAR-local points into torso coordinates."""
        t=np.eye(4);t[:3,:3]=rotation_rpy(self.torso_to_lidar_rpy_rad);t[:3,3]=self.torso_to_lidar_xyz_m
        return t

    @property
    def T_lidar_imu(self):
        """Homogeneous transform maps internal-IMU points into LiDAR coordinates."""
        t=np.eye(4);t[:3,:3]=rotation_rpy(self.lidar_to_imu_rpy_rad);t[:3,3]=self.lidar_to_imu_xyz_m
        return t

    def __post_init__(self):
        if self.rays_per_second not in (10000,20000,200000):
            raise ValueError('Select explicit development10k/20k or nominal200k ray budget')
        if self.scan_hz != 10 or abs(self.physics_dt_s-.005)>1e-10:
            raise ValueError('Current implementation requires10Hz scans and200Hz state')
        if not (0 < self.minimum_range_m < self.maximum_range_m):raise ValueError('Invalid range')
        if not (-90 < self.elevation_min_deg < self.elevation_max_deg < 90):raise ValueError('Invalid elevation')


class PhysxRaycaster:
    """Return closest real collision hit distances from the running PhysX scene.

    Opaque collisions include robot bodies. The optional G1 optical-window proxy
    excludes only its identified sensor-housing capsule from sensing, not physics.
    The direction is normalized, all distances and stage units must be metres.
    Visual-only meshes, glass optics and reflected/multiple returns are unmodeled.
    """
    def __init__(self, *, g1_optical_window=False):
        import omni.physx
        from isaaclab.sim import SimulationContext
        context=SimulationContext.instance()
        if context is not None and not context.cfg.enable_scene_query_support:
            raise RuntimeError("Set SimulationCfg(enable_scene_query_support=True) before simulation creation")
        self.query = omni.physx.get_physx_scene_query_interface()
        self.housing_path=None;self.ignored_housing_hits=0;self.accepted_hits={}
        if g1_optical_window:
            # The imported LiDAR housing capsule contains the optical origin.
            # Represent its unmodelled optical window without changing physics.
            import omni.usd
            from pxr import UsdPhysics
            path='/World/G1/torso_link/collisions/mesh_3/capsule'
            prim=omni.usd.get_context().get_stage().GetPrimAtPath(path)
            if (not prim or prim.GetTypeName()!='Capsule' or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or abs(float(prim.GetAttribute('radius').Get())-.068)>1e-6
                or abs(float(prim.GetAttribute('height').Get())-.010)>1e-6):
                raise RuntimeError('G1 optical housing differs; inspect before enabling sensor-window proxy')
            self.housing_path=path

    def metadata(self):
        return dict(optical_window_proxy=bool(self.housing_path),excluded_sensor_housing_collision=self.housing_path,
            housing_remains_physically_collidable=True,ignored_housing_hits=self.ignored_housing_hits,
            closest_accepted_colliders=self.accepted_hits,
            exclusion_reason='Imported G1 capsule encloses LiDAR origin; approximate optical window only. All other robot/environment occlusions retained.')

    def cast(self, origin_world_m, directions_world, maximum_range_m):
        distances=np.full(len(directions_world),np.nan,dtype=float)
        origin=tuple(float(x) for x in origin_world_m)
        for i,direction in enumerate(directions_world):
            direction=tuple(float(x) for x in direction)
            if self.housing_path:
                nearest=[float('inf'),None]
                def report(hit):
                    path=str(hit.collision)
                    if path==self.housing_path:self.ignored_housing_hits+=1
                    elif float(hit.distance)<nearest[0]:nearest[:]=[float(hit.distance),path]
                    return True
                self.query.raycast_all(origin,direction,float(maximum_range_m),report)
                if math.isfinite(nearest[0]):
                    distances[i]=nearest[0]
                    self.accepted_hits[nearest[1]]=self.accepted_hits.get(nearest[1],0)+1
            else:
                hit=self.query.raycast_closest(origin,direction,float(maximum_range_m))
                if hit['hit']:distances[i]=float(hit['distance'])
        return distances


class SimMid360:
    """Own one continuous simulation clock; instantiate afresh after a reset.

    update_physics(time_s, torso_pose_wxyz, v_world, omega_world) takes the actual
    torso link origin pose [x,y,z,qw,qx,qy,qz] and velocities at that origin.
    It returns {imu: dict|None, scans: list[dict]}. The first IMU sample is withheld
    until a previous velocity exists. A packet closes before sampling the next
    scan boundary. Incomplete final scans are never represented as complete.
    """
    def __init__(self, raycaster, config=None):
        self.config=config or SensorConfig()
        self.raycaster=raycaster
        self.R_TL=rotation_rpy(self.config.torso_to_lidar_rpy_rad)
        self.R_LI=rotation_rpy(self.config.lidar_to_imu_rpy_rad)
        self.previous=None
        self.scan_start_ns=None
        self.parts=[]
        self.ray_index=0
        self.scan_rays=0
        self.scan_ticks=0
        self.first_scan=True
        self.total_ray_queries=0
        self.total_returns=0
        self.completed_scans=0

    def metadata(self):
        return {'schema_version':1,'simulation_only':True,'backend':'PhysX closest accepted collision raycast',
                'raycaster':self.raycaster.metadata() if hasattr(self.raycaster,'metadata') else {},
                'config':asdict(self.config),'T_torso_lidar':self.config.T_torso_lidar.tolist(),
                'T_lidar_imu':self.config.T_lidar_imu.tolist(),'point_frame':'mid360_link; actual inverted URDF mount',
                'imu_frame':'mid360_internal_imu; upstream-configured synthetic fixed extrinsic',
                'imu_acceleration':'specific force R_WI.T @ (a_IMU_world - gravity_world)',
                'imu_derivative':'backward velocity differences at200Hz; acceleration has approx2.5ms differentiation lag',
                'timestamp_basis':'integer simulation nanoseconds; every ray in one physics tick shares that tick timestamp',
                'scan_interval':'half-open [scan_start_ns,scan_start_ns+100000000)',
                'pattern':'nonrepeating deterministic low-discrepancy proxy; not calibrated Livox proprietary optics',
                'angular_coverage_sensor_frame_deg':{'azimuth':[0,360],'elevation':[-7,52]},
                'development_mode':self.config.rays_per_second!=200000,
                'intensity':'constant1.0 simulation placeholder; no material response',
                'line':'proxy0, not physical Livox laser channel','range_model':'single first collision,100mm blind minimum,40m cap',
                'limitations':['collision surfaces rather than visual optical surfaces','no noise or dropout model',
                    'no glass/multipath/material reflectance','internalIMU extrinsic copied from upstream config,not calibrated hardware',
                    'no truth pose/orientation in estimator records','ray budget is simulated cadence,not wall-time throughput claim'],
                'counts':{'ray_queries':self.total_ray_queries,'returns':self.total_returns,'completed_scans':self.completed_scans}}

    def _directions(self,n):
        i=np.arange(self.ray_index,self.ray_index+n,dtype=np.float64)+.5
        azimuth=np.remainder(i*.6180339887498949,1.)*2*np.pi
        lo,hi=np.sin(np.radians([self.config.elevation_min_deg,self.config.elevation_max_deg]))
        sin_el=lo+(hi-lo)*np.remainder(i*.4142135623730951,1.)
        cos_el=np.sqrt(1.-sin_el*sin_el)
        self.ray_index+=n
        return np.column_stack((cos_el*np.cos(azimuth),cos_el*np.sin(azimuth),sin_el))

    def _finish_scan(self):
        if self.parts:
            xyz=np.concatenate([p[0] for p in self.parts]).astype(np.float32)
            dt=np.concatenate([p[1] for p in self.parts]).astype(np.uint32)
        else:
            xyz=np.empty((0,3),np.float32);dt=np.empty(0,np.uint32)
        complete=self.scan_ticks==round(1/self.config.scan_hz/self.config.physics_dt_s)
        packet={'schema_version':1,'scanstart_ns':int(self.scan_start_ns),'scan_end_ns':int(self.scan_start_ns+100000000),
                'xyz_sensor_m':xyz,'pointdt_ns':dt,'intensity':np.ones(len(xyz),np.float32),
                'line':np.zeros(len(xyz),np.uint8),'rays_queried':self.scan_rays,'physics_ticks':self.scan_ticks,
                'complete':complete,'simulation_only':True,'frame_id':'mid360_link'}
        self.parts=[];self.scan_rays=0;self.scan_ticks=0
        self.completed_scans+=int(complete)
        self.first_scan=False
        return packet

    def update_physics(self,time_s,torso_pose_wxyz,torso_velocity_world,torso_angular_velocity_world):
        c=self.config
        if not np.isfinite(time_s) or time_s<0:raise ValueError('Invalid simulation time')
        stamp=int(round(float(time_s)*1e9))
        pose=np.asarray(torso_pose_wxyz,dtype=float);v=np.asarray(torso_velocity_world,dtype=float);w=np.asarray(torso_angular_velocity_world,dtype=float)
        if pose.shape!=(7,) or v.shape!=(3,) or w.shape!=(3,) or not all(np.isfinite(a).all() for a in (pose,v,w)):
            raise ValueError('Nonfinite or malformed measured torso state')
        R_WT=rotation_wxyz(pose[3:]);R_WL=R_WT@self.R_TL;R_WI=R_WL@self.R_LI
        imu=None
        if self.previous is not None:
            dt=(stamp-self.previous[0])/1e9
            if abs(dt-c.physics_dt_s)>2e-7:
                raise ValueError('Simulation clock reset/duplicate/gap: new adapter required for reset; do not fabricate samples')
            a=(v-self.previous[1])/dt;alpha=(w-self.previous[2])/dt
            r=R_WT@(np.asarray(c.torso_to_lidar_xyz_m)+self.R_TL@np.asarray(c.lidar_to_imu_xyz_m))
            a_imu=a+np.cross(alpha,r)+np.cross(w,np.cross(w,r))
            imu={'schema_version':1,'timestamp_ns':stamp,'frame_id':'mid360_internal_imu',
                 'angular_velocity_rad_s':(R_WI.T@w).tolist(),
                 'specific_force_m_s2':(R_WI.T@(a_imu-np.asarray(c.gravity_world_m_s2))).tolist(),
                 'simulation_only':True}
        scan_bin=(stamp//100000000)*100000000
        scans=[]
        if self.scan_start_ns is None:self.scan_start_ns=scan_bin
        elif scan_bin!=self.scan_start_ns:
            if scan_bin!=self.scan_start_ns+100000000:raise ValueError('Skipped scan boundary')
            scans.append(self._finish_scan());self.scan_start_ns=scan_bin
        directions=self._directions(round(c.rays_per_second*c.physics_dt_s))
        origin=pose[:3]+R_WT@np.asarray(c.torso_to_lidar_xyz_m)
        distances=np.asarray(self.raycaster.cast(origin,directions@R_WL.T,c.maximum_range_m),dtype=float)
        if distances.shape!=(len(directions),):raise ValueError('Ray backend returned incompatible distances')
        valid=np.isfinite(distances)&(distances>=c.minimum_range_m)&(distances<=c.maximum_range_m)
        xyz=directions[valid]*distances[valid,None]
        self.parts.append((xyz,np.full(len(xyz),stamp-self.scan_start_ns,dtype=np.uint32)))
        self.scan_rays+=len(directions);self.scan_ticks+=1
        self.total_ray_queries+=len(directions);self.total_returns+=len(xyz)
        self.previous=(stamp,v.copy(),w.copy())
        return {'imu':imu,'scans':scans}


def save_scan(path,packet):
    """Exclusive NPZ sink; estimator arrays only, no pose/world map."""
    import json
    arrays={k:v for k,v in packet.items() if isinstance(v,np.ndarray)}
    scalar={k:v for k,v in packet.items() if not isinstance(v,np.ndarray)}
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays,metadata=json.dumps(scalar))
