#!/usr/bin/env python3
"""CPU-only native estimator startup check. Publishes no sensor/control data."""
import json, os, signal, subprocess, time
from pathlib import Path
import rclpy
w=Path(__file__).resolve().parent
report=w/'logs/startup_probe.json'
if report.exists():raise SystemExit('Refusing to overwrite existing startup evidence')
if os.environ.get('ROS_LOCALHOST_ONLY')!='1' or os.environ.get('ROS_DOMAIN_ID')!='97':raise SystemExit('Probe requires isolated loopback domain97')
command=[str(w/'install/lib/fast_lio/fastlio_mapping'),'--ros-args','--params-file',str(w/'install/share/fast_lio/config/mid360.yaml'),'-p','use_sim_time:=true','-p','pcd_save.pcd_save_en:=false','-p','mapping.extrinsic_est_en:=false']
r={'scope':'Native binary startup/discovery only; no LiDAR, IMU, simulator or actuator data published','command':command,'domain':97,'sensor_data_published':False,'started':False}
p=None;node=None
with (w/'logs/startup_native.log').open('x') as log:
 try:
  rclpy.init();node=rclpy.create_node('null_fastlio_startup_audit')
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=w/'runtime')
  until=time.monotonic()+8
  while time.monotonic()<until and p.poll() is None:
   rclpy.spin_once(node,timeout_sec=.1)
   if 'laser_mapping' in node.get_node_names():
    r['started']=True
    time.sleep(.3)
    break
  r['subscriptions']=[{'name':name,'types':types} for name,types in node.get_subscriber_names_and_types_by_node('laser_mapping','/')] if r['started'] else []
  r['publishers']=[{'name':name,'types':types} for name,types in node.get_publisher_names_and_types_by_node('laser_mapping','/')] if r['started'] else []
  r['input_qos']={}
  for topic in ['/livox/lidar','/livox/imu']:
   r['input_qos'][topic]=[{'node':e.node_name,'type':e.topic_type,'reliability':str(e.qos_profile.reliability),'depth':e.qos_profile.depth} for e in node.get_subscriptions_info_by_topic(topic)]
 except Exception as e:r['error']=repr(e)
 finally:
  if p and p.poll() is None:
   p.send_signal(signal.SIGINT)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:
    r['required_terminate']=True;p.terminate()
    try:p.wait(timeout=3)
    except subprocess.TimeoutExpired:r['required_kill']=True;p.kill();p.wait(timeout=3)
  r['return_code']=p.returncode if p else None
  if node:node.destroy_node()
  if rclpy.ok():rclpy.shutdown()
names={x['name'] for x in r['subscriptions']}
r['pass']=r['started'] and {'/livox/lidar','/livox/imu','/clock'}<=names and r['return_code']==0 and 'error' not in r
report.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2));raise SystemExit(0 if r['pass'] else 1)
