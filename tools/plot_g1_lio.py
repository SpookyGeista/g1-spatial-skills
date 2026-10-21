#!/usr/bin/env python3
"""Render actual native G1 LIO PCD/odometry. No CAD, truth trajectory or fitted alignment."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'evidence/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_g1_lio import digest,json_file,json_lines,pcd_audit,rigid,stamps


def native_xyz(path):
    """Read XYZ by actual PCD offsets, after strict schema/payload validation."""
    details=pcd_audit(path)
    with path.open('rb') as f:
        fields={}
        while True:
            row=f.readline().decode('ascii').strip()
            if row and not row.startswith('#'):
                key,*value=row.split();fields[key]=value
                if key=='DATA':break
        payload=f.read()
    sizes=list(map(int,fields['SIZE']));counts=list(map(int,fields.get('COUNT',['1']*len(sizes))))
    offsets=np.cumsum([0]+[a*b for a,b in zip(sizes,counts)])
    dtype=np.dtype(dict(names=['x','y','z'],formats=['<f4']*3,
        offsets=[int(offsets[fields['FIELDS'].index(k)]) for k in ['x','y','z']],itemsize=int(offsets[-1])))
    data=np.frombuffer(payload,dtype=dtype,count=details['points'])
    return np.column_stack([data[k] for k in ['x','y','z']]),details


def plot(run,output):
    run,output=Path(run).resolve(),Path(output).resolve();sidecar=output.with_suffix('.json')
    if output.suffix.lower()!='.png':raise ValueError('Use a PNG output')
    if output.exists() or sidecar.exists():raise FileExistsError('Refusing to overwrite figure/provenance')
    if not output.parent.is_dir():raise ValueError('Output parent must exist')
    origin=json_file(run/'localization_origin.json');t=rigid(origin['T_map_initial_imu'])
    sensor_metadata=json_file(run/'sim/mid360/metadata.json')
    rays_per_second=sensor_metadata['config']['rays_per_second']
    if type(rays_per_second) is not int or rays_per_second<=0:raise ValueError('Invalid recorded ray budget')
    budget_label=f'{rays_per_second:,}-ray/s MID-360-style simulation proxy'
    points,details=native_xyz(run/'fastlio_map.pcd')
    rows=json_lines(run/'lio/odometry.jsonl');stamp=stamps(rows)
    positions=np.asarray([r['position_m'] for r in rows],float)
    if positions.shape!=(len(rows),3) or not np.isfinite(positions).all():raise ValueError('Invalid native odometry XYZ')
    # Exact saved runtime registration, shared by map and trajectory.
    positions=positions@t[:3,:3].T+t[:3,3]
    stride=max(1,int(np.ceil(len(points)/65000)))
    shown=points[::stride].astype(float)@t[:3,:3].T+t[:3,3]
    fig=plt.figure(figsize=(15,7.4),facecolor='#f8fafc')
    grid=fig.add_gridspec(1,2,width_ratios=[1.2,1])
    ax=fig.add_subplot(grid[0],projection='3d');bx=fig.add_subplot(grid[1])
    zmin,zmax=float(shown[:,2].min()),float(shown[:,2].max())
    scatter=ax.scatter(*shown.T,c=shown[:,2],cmap='viridis',vmin=zmin,vmax=zmax,s=.7,alpha=.35,rasterized=True)
    ax.plot(*positions.T,color='#e83f32',linewidth=1.7,label='Native FAST-LIO IMU trajectory')
    ax.scatter(*positions[0],c='#151a23',s=35,marker='o',label='First estimate')
    ax.scatter(*positions[-1],c='#e83f32',s=70,marker='*',edgecolors='white',linewidths=.6,label='Last estimate')
    ax.set(xlabel='Registered map X [m]',ylabel='Y [m]',zlabel='Z [m]',title='Native registered map · 3D')
    extent=np.ptp(shown,axis=0);ax.set_box_aspect(np.maximum(extent,[1,1,.5]));ax.view_init(27,-62)
    ax.legend(loc='upper left',fontsize=8,framealpha=.85)
    bx.scatter(shown[:,0],shown[:,1],c=shown[:,2],cmap='viridis',vmin=zmin,vmax=zmax,s=.7,alpha=.38,rasterized=True)
    bx.plot(positions[:,0],positions[:,1],color='#e83f32',linewidth=1.5,label='Native estimated IMU position')
    bx.scatter(*positions[0,:2],c='#151a23',s=40,marker='o',zorder=5)
    bx.scatter(*positions[-1,:2],c='#e83f32',s=80,marker='*',edgecolors='white',linewidths=.7,zorder=6)
    bx.set(xlabel='Registered map X [m]',ylabel='Y [m]',title='Plan view · actual measured returns')
    bx.set_aspect('equal');bx.grid(alpha=.12);bx.legend(loc='upper right',fontsize=8)
    cb=fig.colorbar(scatter,ax=[ax,bx],fraction=.024,pad=.04,shrink=.78)
    cb.set_label('Registered point height [m]',fontsize=9)
    fig.suptitle('Actual G1 simulation | Native FAST-LIO map and estimated trajectory',x=.49,y=.973,fontsize=16,weight='bold',color='#111827')
    fig.text(.04,.912,'Unitree G1 + SONIC  •  HKU-MaRS FAST-LIO / Ericsii ROS 2 port  •  '+budget_label,fontsize=10,color='#344054')
    fig.text(.04,.070,f"{details['points']:,} native PCD points; deterministic 1/{stride} display subset ({len(shown):,} points).  {len(rows):,} native odometry samples.",fontsize=10)
    fig.text(.04,.042,'Exact saved runtime frame registration only. No CAD geometry, truth map/trajectory or posthoc fit added.',fontsize=10)
    fig.text(.04,.016,'Simulation integration evidence: approximate collision-ray sensor, not calibrated physical MID-360 or real-hardware validation.',fontsize=9,color='#475467')
    # Manual panel positions reserve space for captions/colorbar without 3D tight_layout warnings.
    ax.set_position([.015,.16,.50,.69]);bx.set_position([.57,.18,.32,.64]);cb.ax.set_position([.92,.22,.014,.53])
    fig.savefig(output,dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
    provenance=dict(run=str(run),figure=str(output),scope='Actual native PCD + native odometry only, no truth trajectory/CAD inputs',
        native_map=details,display_stride=stride,display_points=len(shown),odometry_samples=len(rows),
        odometry_first_stamp_ns=int(stamp[0]),odometry_last_stamp_ns=int(stamp[-1]),
        runtime_registration=origin,sensor_rays_per_second=rays_per_second,sensor_development_mode=sensor_metadata.get('development_mode'),source_sha256={name:digest(run/name) for name in ['fastlio_map.pcd','lio/odometry.jsonl','localization_origin.json','sim/mid360/metadata.json']},
        plotter_sha256=digest(__file__),evaluator_helper_sha256=digest(Path(__file__).with_name('evaluate_g1_lio.py')),figure_sha256=digest(output),
        map_frame='exact runtime T_map_initial_imu; no posthoc alignment or scale change')
    with sidecar.open('x') as f:json.dump(provenance,f,indent=2,allow_nan=False);f.write('\n')
    return provenance


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=plot(a.run,a.output);print(result['figure'])

if __name__=='__main__':main()
