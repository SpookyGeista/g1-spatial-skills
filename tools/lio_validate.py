#!/usr/bin/env python3
"""Run/evaluate real FAST-LIO on retained Isaac sensor records. No GPU or actuators.

Evaluation truth is loaded only after estimator/bridge shutdown. Exactly one
initial rigid-frame alignment is used; no trajectory fit or truth correction.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
WS=ROOT/'ros2_ws'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readlines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def qmatrix(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.01:raise ValueError('Invalid quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def rigid(matrix):
    x=np.asarray(matrix,dtype=float)
    if x.shape!=(4,4) or not np.isfinite(x).all() or not np.allclose(x[3],[0,0,0,1],atol=1e-6) or not np.allclose(x[:3,:3].T@x[:3,:3],np.eye(3),atol=1e-5) or abs(np.linalg.det(x[:3,:3])-1)>1e-5:raise ValueError('Invalid rigid transform')
    return x
def interp_truth(times,poses,stamp):
    i=int(np.searchsorted(times,stamp))
    if i<len(times) and times[i]==stamp:return poses[i]
    if i==0 or i==len(times):raise ValueError('Estimate outside truth time range')
    a,b=poses[i-1],poses[i];u=float(stamp-times[i-1])/float(times[i]-times[i-1])
    # Short 5ms rotation interpolation on SO(3), not Euler-angle interpolation.
    rel=a[:3,:3].T@b[:3,:3];angle=math.acos(float(np.clip((np.trace(rel)-1)/2,-1,1)))
    if angle<1e-10:r=a[:3,:3]
    else:
        axis=np.array([rel[2,1]-rel[1,2],rel[0,2]-rel[2,0],rel[1,0]-rel[0,1]])/(2*math.sin(angle))
        K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        r=a[:3,:3]@(np.eye(3)+math.sin(u*angle)*K+(1-math.cos(u*angle))*(K@K))
    out=np.eye(4);out[:3,:3]=r;out[:3,3]=(1-u)*a[:3,3]+u*b[:3,3];return out

def scan_index(capture):
    scans=[]
    for e in readlines(capture/'events.jsonl'):
        if e['kind']!='cloud':continue
        with np.load(capture/e['file'],allow_pickle=False) as z:
            meta=json.loads(str(z['metadata'].item()))
            dt=z['pointdt_ns'];points=z['xyz_sensor_m']
            if not len(dt) or points.shape!=(len(dt),3) or not np.isfinite(points).all():raise ValueError('Invalid capture cloud')
            scans.append(dict(file=e['file'],start_ns=int(meta['scanstart_ns']),end_ns=int(meta['scan_end_ns']),last_point_ns=int(meta['scanstart_ns'])+int(dt.max()),points=len(dt),complete=bool(meta['complete'])))
    return scans

def pcd_check(path):
    if not path.exists():return {'present':False,'pass':False}
    raw=path.read_bytes();pos=0;fields={}
    while True:
        end=raw.find(b'\n',pos)
        if end<0:raise ValueError('PCD header missing DATA')
        row=raw[pos:end].decode('ascii').strip();pos=end+1
        if row and not row.startswith('#'):
            key,*value=row.split();fields[key]=value
            if key=='DATA':break
    n=int(fields['POINTS'][0]);size=list(map(int,fields['SIZE']));count=list(map(int,fields.get('COUNT',['1']*len(size))))
    names=fields['FIELDS'];offsets=np.cumsum([0]+[a*b for a,b in zip(size,count)])
    if fields['DATA']!=['binary']:raise ValueError('Expected native binary PCD')
    exact=len(raw)-pos==n*int(offsets[-1]);finite=False
    if exact and n:
        dtype=np.dtype({'names':['x','y','z'],'formats':['<f4']*3,'offsets':[int(offsets[names.index(k)]) for k in ['x','y','z']],'itemsize':int(offsets[-1])})
        a=np.frombuffer(raw,dtype=dtype,count=n,offset=pos);finite=all(np.isfinite(a[k]).all() for k in ['x','y','z'])
    return {'present':True,'points':n,'bytes':len(raw),'sha256':sha(path),'binary_payload_length_matches':exact,'finite_xyz':bool(finite),'pass':bool(n>=100 and exact and finite)}

def evaluate(run,output_name='evaluation.json'):
    if Path(output_name).name!=output_name or not output_name.endswith('.json'):raise ValueError('Use a JSON basename')
    target=run/output_name
    if target.exists():raise FileExistsError(target)
    manifest=json.loads((run/'manifest.json').read_text());capture=Path(manifest['capture'])
    errors=[];result={'scope':'Actual native FAST-LIO on actual PhysX synthetic sensor-rig capture; not fullG1, physicalMID360 or mission acceptance','thresholds':{'translation_rmse_m':.20,'translation_max_m':.40,'rotation_max_deg':10.,'post_initialization_scan_coverage':.90,'initialize_by_sim_s':2.0,'map_min_points':100},'alignment':'One SE(3) alignment at first native estimate, evaluation only; no trajectory fitting; no truth supplied to estimator','checks':{}}
    try:
        input_changes=[]
        for name,digest in manifest['input_sha256'].items():
            actual=run/manifest.get('source_snapshots',{}).get(name,'') if name in manifest.get('source_snapshots',{}) else Path(name)
            if not actual.exists() or sha(actual)!=digest:input_changes.append(name)
        result['input_changes']=input_changes;result['checks']['inputs_unchanged']=not input_changes
        bridge=json.loads((run/'bridge/bridge_result.json').read_text());result['bridge']=bridge
        rows=readlines(run/'bridge/odometry.jsonl');rejected=[r for r in rows if 'rejected' in r];rows=[r for r in rows if 'rejected' not in r]
        result['checks']['native_processes_passed']=manifest.get('bridge_return_code')==0 and manifest.get('native_return_code')==0 and not manifest.get('error')
        result['checks']['bridge_input_integrity']=bridge['error'] is None and bridge['counts']['clock_rejections']==0 and not rejected and bridge['counts'].get('rejected_odometry',0)==0
        if not rows:raise ValueError('No accepted native odometry')
        stamps=np.array([r['stamp_ns'] for r in rows],dtype=np.int64)
        result['checks']['odometry_strictly_monotonic']=bool(np.all(np.diff(stamps)>0))
        truth=readlines(capture/'evaluation_truth.jsonl');times=np.array([r['stamp_ns'] for r in truth],dtype=np.int64);poses=[rigid(r['imu_to_world']) for r in truth]
        if not np.all(np.diff(times)>0):raise ValueError('Nonmonotonic truth')
        estimates=[]
        for r in rows:
            x=np.eye(4);x[:3,:3]=qmatrix(r['quaternion_wxyz']);x[:3,3]=r['position_m'];estimates.append(rigid(x))
        align=interp_truth(times,poses,int(stamps[0]))@np.linalg.inv(estimates[0]);result['fixed_evaluation_alignment']=align.tolist();result['alignment_stamp_ns']=int(stamps[0])
        translations=[];angles=[];aligned=[]
        for t,x in zip(stamps,estimates):
            measured=interp_truth(times,poses,int(t));actual=align@x;err=np.linalg.norm(actual[:3,3]-measured[:3,3]);angle=math.degrees(math.acos(float(np.clip((np.trace(actual[:3,:3].T@measured[:3,:3])-1)/2,-1,1))))
            translations.append(float(err));angles.append(angle);aligned.append(actual[:3,3].tolist())
        scans=scan_index(capture);first=[i for i,s in enumerate(scans) if s['start_ns']<=stamps[0]<=s['end_ns']]
        if not first:raise ValueError('First estimate not associated with an input scan')
        scans_after=scans[first[-1]:];matched=[any(s['start_ns']<=int(t)<s['end_ns'] for t in stamps) for s in scans_after]
        coverage=sum(matched)/len(matched);init_s=(int(stamps[0])-scans[0]['start_ns'])*1e-9
        result['trajectory']={'accepted_odometry':len(rows),'first_stamp_ns':int(stamps[0]),'last_stamp_ns':int(stamps[-1]),'initialization_sim_seconds':init_s,'translation_rmse_m':float(np.sqrt(np.mean(np.square(translations)))),'translation_max_m':max(translations),'rotation_max_deg':max(angles),'post_initialization_scans':len(scans_after),'matched_post_initialization_scans':sum(matched),'post_initialization_coverage':coverage,'missing_post_initialization_scan_files':[s['file'] for s,ok in zip(scans_after,matched) if not ok]}
        m=result['trajectory'];result['checks'].update(initialized_during_stationary_prefix=init_s<=2.,translation_rmse=m['translation_rmse_m']<=.20,translation_max=m['translation_max_m']<=.40,rotation_max=m['rotation_max_deg']<=10.,scan_coverage=coverage>=.90,all_scans_complete=all(s['complete'] for s in scans))
        publication=readlines(run/'bridge/publication_index.jsonl');clouds=[r for r in publication if r['kind']=='cloud'];imus=[r for r in publication if r['kind']=='imu']
        result['checks']['all_recorded_inputs_published']=len(clouds)==len(scans) and len(imus)==sum(e['kind']=='imu' for e in readlines(capture/'events.jsonl'))
        lag=[r['simulation_lag_ns']*1e-9 for r in rows];latency=[]
        for r in rows:
            match=[c for c in clouds if c['stamp_ns']<=r['stamp_ns']<c['acquisition_end_ns']]
            if match:latency.append(r['received_monotonic']-match[-1]['published_monotonic'])
        result['latency']={'simulation_lag_s_max':max(lag),'simulation_lag_s_p95':float(np.percentile(lag,95)),'publication_to_receipt_wall_s_max':max(latency) if latency else None,'publication_to_receipt_wall_s_p95':float(np.percentile(latency,95)) if latency else None,'wall_latency_samples':len(latency),'meaning':'Lag includes processing and DDS; replay rate and workstation scheduling affect wall values'}
        result['map']=pcd_check(run/'map.pcd');result['checks']['native_map_saved_nonempty']=result['map']['pass'] and bool(manifest.get('map_service',{}).get('success'))
        registered=readlines(run/'bridge/registered_index.jsonl');count=0
        for row in registered:
            with np.load(run/'bridge'/row['file'],allow_pickle=False) as z:
                p=z['xyz'];assert p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all();count+=len(p)
        result['registered_clouds']={'files':len(registered),'points':count};result['checks']['finite_nonempty_registered_clouds']=count>0
        result['checks']['sensor_capture_checks_passed']=all(json.loads((capture/'smoke_summary.json').read_text())['checks'].values())
        np.savez_compressed(run/('errors_'+target.stem+'.npz'),stamp_ns=stamps,translation_error_m=translations,rotation_error_deg=angles,aligned_position_m=aligned)
    except Exception as e:errors.append(repr(e))
    result['errors']=errors;result['accepted']=bool(result['checks']) and all(result['checks'].values()) and not errors
    result['failed_checks']=[k for k,v in result['checks'].items() if not v]
    write(target,result);return result

def stop_child(p):
    if p.poll() is None:
        p.send_signal(signal.SIGINT)
        try:p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=3)
    return p.returncode

def save_map():
    import rclpy
    from std_srvs.srv import Trigger
    rclpy.init();node=rclpy.create_node('punktnull_map_capture');result={}
    try:
        client=node.create_client(Trigger,'/map_save')
        if not client.wait_for_service(timeout_sec=5):raise RuntimeError('Native map_save service missing')
        future=client.call_async(Trigger.Request());rclpy.spin_until_future_complete(node,future,timeout_sec=5)
        if not future.done():raise RuntimeError('Native map_save response timeout')
        answer=future.result();result={'success':answer.success,'message':answer.message}
    except Exception as e:result={'success':False,'error':repr(e)}
    finally:node.destroy_node();rclpy.shutdown()
    return result

def execute(capture,output,rate):
    import fcntl
    if ROOT not in output.parents or output.exists():raise ValueError('Use a new run directory under punktNULL')
    if not (capture/'smoke_summary.json').exists():raise ValueError('Capture not finalized')
    if rate<=0 or not math.isfinite(rate):raise ValueError('Invalid rate')
    lock=(WS/'runtime/estimator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output.mkdir(parents=True);(output/'sources').mkdir()
    config=WS/'config/mid360_simdev.yaml';shutil.copy2(config,output/'config.yaml')
    exe=WS/'install/lib/fast_lio/fastlio_mapping'
    files=[capture/'events.jsonl',capture/'evaluation_truth.jsonl',capture/'smoke_summary.json',exe,config,ROOT/'tools/lio_bridge.py',Path(__file__).resolve()]+sorted(capture.glob('scan_*.npz'))
    manifest={'capture':str(capture),'scope':'Recorded genuine Isaac measurements into real native FAST-LIO; syntheticrig not G1mission','replay_rate':rate,'input_sha256':{str(f):sha(f) for f in files},'groundtruth_sent_to_estimator':False,'domain':97,'error':None,'source_snapshots':{}}
    for source in [ROOT/'tools/lio_bridge.py',Path(__file__).resolve(),config]:
        relative='sources/'+source.name;shutil.copy2(source,output/relative);manifest['source_snapshots'][str(source)]=relative
    write(output/'plan.json',manifest)
    os.environ.update(ROS_DOMAIN_ID='97',ROS_LOCALHOST_ONLY='1',ROS_LOG_DIR=str(output/'ros_logs'))
    env=os.environ.copy();env.update(ROS_DOMAIN_ID='97',ROS_LOCALHOST_ONLY='1',ROS_LOG_DIR=str(output/'ros_logs'))
    native=[str(exe),'--ros-args','--params-file',str(output/'config.yaml'),'-p','use_sim_time:=true','-p','pcd_save.pcd_save_en:=true','-p','map_file_path:='+str(output/'map.pcd')]
    bridge=['/usr/bin/python3',str(ROOT/'tools/lio_bridge.py'),'--output',str(output/'bridge'),'--replay',str(capture/'events.jsonl'),'--rate',str(rate),'--tail-seconds','3','--seconds','90']
    manifest['native_command']=native;manifest['bridge_command']=bridge
    np_=bp=None
    with (output/'native.log').open('x') as nl,(output/'bridge.log').open('x') as bl:
        try:
            np_=subprocess.Popen(native,stdout=nl,stderr=subprocess.STDOUT,env=env,cwd=output)
            bp=subprocess.Popen(bridge,stdout=bl,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            manifest['bridge_return_code']=bp.wait(timeout=100)
            manifest['map_service']=save_map()
        except (Exception,KeyboardInterrupt) as e:manifest['error']=repr(e)
        finally:
            if bp is not None:manifest['bridge_return_code']=stop_child(bp)
            if np_ is not None:manifest['native_return_code']=stop_child(np_)
    for folder,dest in [('Log','native_debug'),('PCD','native_exit_clouds')]:
        for f in (WS/'runtime'/folder).glob('*'):
            if f.is_file():
                (output/dest).mkdir(exist_ok=True);shutil.copy2(f,output/dest/f.name)
    write(output/'manifest.json',manifest);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return evaluate(output)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--evaluate',type=Path);p.add_argument('--output-name',default='evaluation.json');p.add_argument('--rate',type=float,default=1.)
    a=p.parse_args()
    if a.evaluate:r=evaluate(a.evaluate.resolve(),a.output_name)
    elif a.execute and a.capture and a.output:r=execute(a.capture.resolve(),a.output.resolve(),a.rate)
    else:p.error('Use --capture PATH --output NEW_PATH --execute, or --evaluate RUN')
    print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1
if __name__=='__main__':raise SystemExit(main())
