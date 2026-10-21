#!/usr/bin/env python3
"""Publish genuine simulated measurements to the real FAST-LIO ROS2 node.

Run with /usr/bin/python3 after sourcing ros2_ws/activate.sh. No motor/device IO.
Can replay sensor events or accept live sensor-only TCP on loopback.
"""
from __future__ import annotations
import argparse
import json
import queue
import socket
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from punktnull.sensor_transport import receive, encode, decode

def events(path):
    path = Path(path)
    with path.open() as source:
        for line in source:
            e = json.loads(line)
            if e['kind'] == 'cloud':
                with np.load(path.parent/e.pop('file'), allow_pickle=False) as data:
                    e.update({key:data[key].copy() for key in
                              ('xyz_sensor_m','pointdt_ns','intensity','line')})
                    metadata=json.loads(str(data['metadata'].item()))
                    if int(metadata['scanstart_ns']) != int(e['scanstart_ns']):
                        raise ValueError('Conflicting stored scan timestamps')
                    e['scan_end_ns']=int(metadata['scan_end_ns'])
                    e['complete']=bool(metadata['complete'])
            e.setdefault('stamp_ns', e.get('timestamp_ns', e.get('scanstart_ns')))
            yield decode(encode(e)[4:])  # Same sensor-only validation as live TCP.

def event_time(e):
    return int(e['scan_end_ns'] if e['kind']=='cloud' else e['stamp_ns'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--port',type=int,default=16720)
    parser.add_argument('--rate',type=float,default=1.)
    parser.add_argument('--tail-seconds',type=float,default=3.)
    parser.add_argument('--seconds',type=float,default=300.)
    args = parser.parse_args()
    if args.output.exists() or args.rate <= 0:
        parser.error('Use a new output directory and a positive replay rate')
    args.output.mkdir(parents=True)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, PointCloud2
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from livox_ros_driver2.msg import CustomMsg, CustomPoint
    from builtin_interfaces.msg import Time
    def stamp(ns):
        return Time(sec=int(ns//1_000_000_000),nanosec=int(ns%1_000_000_000))
    rclpy.init()
    node = Node('punktnull_sim_sensor_bridge')
    qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
    lidar = node.create_publisher(CustomMsg,'/livox/lidar',qos)
    imu = node.create_publisher(Imu,'/livox/imu',qos)
    clock = node.create_publisher(Clock,'/clock',qos)
    counts = dict(imu=0,cloud=0,points=0,odometry=0,registered_clouds=0,
                  registered_points=0,clock_rejections=0,rejected_odometry=0)
    failure = None; stop = threading.Event(); incoming = queue.Queue(maxsize=800)
    odom_file = (args.output/'odometry.jsonl').open('w',buffering=1)
    cloud_file = (args.output/'registered_index.jsonl').open('w',buffering=1)
    publication_file=(args.output/'publication_index.jsonl').open('w',buffering=1)
    last_odom_ns = -1
    def odometry(msg):
        nonlocal last_odom_ns
        ns = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        p,q = msg.pose.pose.position,msg.pose.pose.orientation
        value = dict(stamp_ns=ns,frame_id=msg.header.frame_id,child_frame_id=msg.child_frame_id,
                     position_m=[p.x,p.y,p.z],quaternion_wxyz=[q.w,q.x,q.y,q.z],
                     covariance=list(msg.pose.covariance),received_monotonic=time.monotonic(),
                     latest_sensor_clock_ns=publish.last_clock,
                     simulation_lag_ns=publish.last_clock-ns,
                     source='HKU FAST-LIO algorithm / Ericsii ROS2 port; measurement-only inputs')
        values = np.asarray(value['position_m']+value['quaternion_wxyz']+value['covariance'])
        if not np.isfinite(values).all() or ns <= last_odom_ns:
            counts['rejected_odometry']+=1
            value['rejected']='Nonfinite or nonmonotonic estimator output'
            odom_file.write(json.dumps(value)+'\n'); return
        last_odom_ns = ns; counts['odometry'] += 1
        odom_file.write(json.dumps(value,allow_nan=False)+'\n')
        tmp=args.output/'lio_state.tmp.json';tmp.write_text(json.dumps(value,allow_nan=False))
        tmp.replace(args.output/'lio_state.json')
    def registered(msg):
        # Persist real PointCloud2 XYZ without importing a different PCL stack.
        fields={f.name:f for f in msg.fields}
        if not all(k in fields for k in ('x','y','z')) or msg.point_step < 12:return
        endian='>' if msg.is_bigendian else '<'
        dtype=np.dtype(dict(names=['x','y','z'],formats=[endian+'f4']*3,
                            offsets=[fields[k].offset for k in ('x','y','z')],itemsize=msg.point_step))
        rows=[]
        for row in range(msg.height):
            raw=np.frombuffer(msg.data,dtype=dtype,count=msg.width,offset=row*msg.row_step)
            rows.append(np.column_stack([raw[k] for k in ('x','y','z')]))
        points=np.concatenate(rows);ns=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        counts['registered_clouds']+=1;counts['registered_points']+=len(points)
        name=f"registered_{counts['registered_clouds']:06d}.npz"
        np.savez_compressed(args.output/name,xyz=points,stamp_ns=ns)
        cloud_file.write(json.dumps(dict(file=name,stamp_ns=ns,frame_id=msg.header.frame_id,points=len(points)))+'\n')
    node.create_subscription(Odometry,'/Odometry',odometry,qos)
    node.create_subscription(PointCloud2,'/cloud_registered',registered,qos)
    def listener():
        nonlocal failure
        try:
            with socket.socket() as server:
                server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                server.bind(('127.0.0.1',args.port));server.listen(1);server.settimeout(.5)
                print('LIO_BRIDGE_LISTENING',args.port,flush=True)
                while not stop.is_set():
                    try:conn,_=server.accept();break
                    except socket.timeout:continue
                else:return
                with conn:
                    # Isaac may render its warm-up frames before the first physics sample.
                    # Startup allowance applies only to the first complete sensor packet.
                    conn.settimeout(30)
                    first_packet = True
                    while not stop.is_set():
                        try:
                            packet = receive(conn)
                            if first_packet:
                                conn.settimeout(2)
                                first_packet = False
                            incoming.put(packet,timeout=1)
                        except EOFError:incoming.put(None);return
        except Exception as exc:failure=repr(exc);stop.set()
    def publish(e):
        e=decode(encode(e)[4:])
        kind=e['kind'];ns=int(e['stamp_ns'])
        if ns <= publish.last_input[kind]:
            raise ValueError('Nonadvancing '+kind+' input timestamp')
        if kind=='cloud' and not e['complete']:
            raise ValueError('Incomplete scan is not estimator input')
        # /clock always advances at acquisition end, never rolls back to a scan start.
        now_ns=event_time(e)
        if now_ns < publish.last_clock:
            counts['clock_rejections']+=1
            raise ValueError('Acquisition clock moved backward')
        else:
            clock.publish(Clock(clock=stamp(now_ns)));publish.last_clock=now_ns
        if kind=='imu':
            m=Imu();m.header.stamp=stamp(ns);m.header.frame_id='mid360_internal_imu'
            m.orientation_covariance[0]=-1. # No orientation measurement supplied.
            m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z=map(float,e['angular_velocity_rad_s'])
            m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z=map(float,e['specific_force_m_s2'])
            imu.publish(m)
        elif kind=='cloud':
            m=CustomMsg();m.header.stamp=stamp(int(e['scanstart_ns']));m.header.frame_id='mid360_link'
            m.timebase=int(e['scanstart_ns']);m.lidar_id=0
            points=np.asarray(e['xyz_sensor_m']);dt=np.asarray(e['pointdt_ns'])
            intensity=np.clip(np.asarray(e['intensity']),0,255).astype(np.uint8)
            lines=np.asarray(e['line'])
            m.points=[CustomPoint(offset_time=int(t),x=float(p[0]),y=float(p[1]),z=float(p[2]),
                                  reflectivity=int(i),tag=0,line=int(l))
                      for p,t,i,l in zip(points,dt,intensity,lines)]
            m.point_num=len(m.points);counts['points']+=m.point_num;lidar.publish(m)
        else:raise ValueError('Unknown event kind')
        counts[kind]+=1
        publish.last_input[kind]=ns
        publication_file.write(json.dumps(dict(kind=kind,stamp_ns=ns,acquisition_end_ns=now_ns,
            published_monotonic=time.monotonic(),points=len(e['pointdt_ns']) if kind=='cloud' else None))+'\n')
    publish.last_clock=-1
    publish.last_input={'imu':-1,'cloud':-1}
    try:
        # DDS discovery and endpoint verification before measurements are sent.
        deadline=time.monotonic()+15
        while not (imu.get_subscription_count() and lidar.get_subscription_count()):
            rclpy.spin_once(node,timeout_sec=.05)
            if time.monotonic()>deadline:raise RuntimeError('FAST-LIO input subscribers unavailable')
        print('LIO_INPUT_SUBSCRIBERS_READY',flush=True)
        started=time.monotonic();end=None
        if args.replay:
            stream=iter(events(args.replay));pending=next(stream,None)
            origin=event_time(pending) if pending is not None else 0
        else:
            thread=threading.Thread(target=listener,daemon=True);thread.start()
            pending=None
        while not stop.is_set():
            now=time.monotonic()
            if now-started > args.seconds:raise RuntimeError('Sensor bridge timeout')
            if args.replay:
                if pending is not None and (event_time(pending)-origin)*1e-9/args.rate <= now-started:
                    publish(pending);pending=next(stream,None)
                if pending is None and end is None:end=now
            elif end is None:
                try:
                    e=incoming.get_nowait()
                    if e is None:end=now
                    else:publish(e)
                except queue.Empty:pass
            rclpy.spin_once(node,timeout_sec=0.0001)
            if end is not None and now-end>=args.tail_seconds:break
        if failure:raise RuntimeError(failure)
    except Exception as exc:
        failure=repr(exc);print('LIO_BRIDGE_ERROR',failure,flush=True)
    finally:
        stop.set();odom_file.close();cloud_file.close();publication_file.close()
        result=dict(counts=counts,error=failure,inputs='simulated LiDAR points and internal IMU only',
                    orientation_measurements=False,groundtruth_corrections=False,physical_device=False)
        (args.output/'bridge_result.json').write_text(json.dumps(result,indent=2))
        node.destroy_node();rclpy.shutdown()
    return 1 if failure or counts['odometry']==0 or counts['rejected_odometry'] else 0

if __name__=='__main__':raise SystemExit(main())
