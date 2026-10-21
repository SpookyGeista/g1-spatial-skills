#!/usr/bin/env python3
"""Plot preserved native outputs and held-out evaluation; no fabricated geometry."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lio_validate import interp_truth,rigid,readlines
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
output=run/'map_trajectory.png'
if output.exists():raise SystemExit('Refusing to overwrite figure')
e=json.loads((run/'evaluation.json').read_text());m=json.loads((run/'manifest.json').read_text());capture=Path(m['capture'])
truth=readlines(capture/'evaluation_truth.jsonl');tt=np.array([r['stamp_ns'] for r in truth],dtype=np.int64);tp=[rigid(r['imu_to_world']) for r in truth]
with np.load(run/'errors_evaluation.npz',allow_pickle=False) as z:
 t=z['stamp_ns'];estimated=z['aligned_position_m'];err=z['translation_error_m'];angle=z['rotation_error_deg']
measured=np.asarray([interp_truth(tt,tp,int(x))[:3,3] for x in t]);alignment=np.asarray(e['fixed_evaluation_alignment'])
clouds=[]
for row in readlines(run/'bridge/registered_index.jsonl'):
 with np.load(run/'bridge'/row['file'],allow_pickle=False) as z:clouds.append(z['xyz'])
cloud=np.concatenate(clouds);stride=max(1,int(np.ceil(len(cloud)/18000)));shown=cloud[::stride]@alignment[:3,:3].T+alignment[:3,3]
fig=plt.figure(figsize=(15,5.8),facecolor='#f8fafc');gs=fig.add_gridspec(1,3,width_ratios=[1.13,1,.88])
ax=fig.add_subplot(gs[0],projection='3d');ax.scatter(*shown.T,c=shown[:,2],cmap='viridis',s=.7,alpha=.5,rasterized=True)
ax.plot(*measured.T,color='#f59e0b',linewidth=2.2,label='Held-out PhysX IMU')
ax.plot(*estimated.T,color='#ef4444',linewidth=1.3,label='FAST-LIO, initial alignment')
ax.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]',title='Native registered returns (3D)');ax.view_init(28,-65)
ax.set_box_aspect((8,6,3));ax.legend(loc='upper left',fontsize=7)
bx=fig.add_subplot(gs[1]);bx.scatter(shown[:,0],shown[:,1],c=shown[:,2],cmap='viridis',s=.6,alpha=.4,rasterized=True)
bx.plot(measured[:,0],measured[:,1],color='#f59e0b',linewidth=2.5,label='Held-out truth')
bx.plot(estimated[:,0],estimated[:,1],color='#ef4444',linewidth=1.4,label='Native estimate')
bx.scatter([estimated[0,0]],[estimated[0,1]],marker='o',s=35,c='black',label='Initial alignment')
bx.set(xlabel='X [m]',ylabel='Y [m]',title='Plan view: measured map + trajectory');bx.set_aspect('equal');bx.grid(alpha=.15);bx.legend(fontsize=7)
cx=fig.add_subplot(gs[2]);cx.plot(t*1e-9,err*100,color='#dc2626',label='Translation error [cm]');cx.axhline(40,color='#dc2626',linestyle=':',alpha=.6,label='Maximum threshold [40 cm]')
cy=cx.twinx();cy.plot(t*1e-9,angle,color='#2563eb',label='Orientation error [deg]');cy.axhline(10,color='#2563eb',linestyle=':',alpha=.6)
cx.set(xlabel='Simulation time [s]',ylabel='Translation error [cm]',title='One fixed initial alignment only');cy.set_ylabel('Orientation error [deg]');cx.set_ylim(0,42);cy.set_ylim(0,10.5);cx.grid(alpha=.2)
lines=cx.get_lines()+cy.get_lines()[:1];cx.legend(lines,[x.get_label() for x in lines],loc='upper left',fontsize=7)
tr=e['trajectory'];fig.suptitle('Actual PhysX sensor rig + native FAST-LIO | 20k rays/s development capture (not 200k/s nominal)',fontsize=12,weight='bold',y=.97)
fig.text(.035,.052,f"{len(cloud):,} native registered points; deterministic 1/{stride} display subset.  No CAD/ground-truth map inserted.",fontsize=9)
fig.text(.035,.022,f"RMSE {tr['translation_rmse_m']*100:.2f} cm | max {tr['translation_max_m']*100:.2f} cm | max angle {tr['rotation_max_deg']:.2f} deg | scan outputs {tr['matched_post_initialization_scans']}/{tr['post_initialization_scans']} | rig diagnostic, not full-G1/physical validation",fontsize=9)
fig.subplots_adjust(left=.025,right=.96,top=.85,bottom=.17,wspace=.30);fig.savefig(output,dpi=150);plt.close(fig)
print(output)
