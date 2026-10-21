#!/usr/bin/env python3
"""Posthoc CPU fixture proposal against saved URDF AABBs; no contact claims."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from punktnull.terrain.service_passage import FIXTURE_CONTRACT,collision_boxes


def evaluate(bounds):
    if bounds.ndim!=4 or bounds.shape[2:]!=(2,3) or not np.isfinite(bounds).all():raise ValueError('Invalid body bounds')
    rows=[];total=0
    for box in collision_boxes():
        low=np.asarray(box['aabb_min']);high=np.asarray(box['aabb_max'])
        gap=np.maximum(np.maximum(bounds[:,:,0]-high,low-bounds[:,:,1]),0.)
        distance=np.linalg.norm(gap,axis=-1)
        overlap=np.all(np.minimum(bounds[:,:,1],high)-np.maximum(bounds[:,:,0],low)>1e-9,axis=-1)
        rows.append({'prim':box['prim'],'sample_geometry_overlaps':int(overlap.sum()), 'minimum_sampled_aabb_distance_m':float(distance.min())})
        total+=int(overlap.sum())
    x0,x1=FIXTURE_CONTRACT['roof_x_interval_m']
    cross=(bounds[:,:,0,0]<=x1)&(bounds[:,:,1,0]>=x0)
    if not cross.any():raise ValueError('No body geometry crosses proposed roof interval')
    max_z=float(bounds[:,:,1,2][cross].max())
    ylow=float(bounds[:,:,0,1][cross].min());yhigh=float(bounds[:,:,1,1][cross].max())
    return {'sample_count':len(bounds),'collision_geometries':bounds.shape[1],
            'sample_geometry_overlaps':total,'all_sampled_aabbs_clear':total==0,
            'roof_crossing_max_geometry_z_m':max_z,
            'roof_vertical_margin_m':FIXTURE_CONTRACT['roof_underside_z_m']-max_z,
            'roof_crossing_y_interval_m':[ylow,yhigh],
            'minimum_side_margin_m':min(ylow+.775,.775-yhigh),'obstacles':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,default=ROOT/'evidence/terrain/crawl_geometry_003');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refuse to overwrite an existing report directory')
    a.output.mkdir(parents=True)
    report={'scope':'Collision fixture geometric preflight only; not executed PhysX passage validation',
            'fixture':FIXTURE_CONTRACT,'collision_boxes':collision_boxes(), 'sources_sha256':{},'datasets':{},
            'limits':['Sampled URDF FK conservative AABBs; no exact mesh contact forces or intersample interpolation proof.',
                      'Measured source is a prior flat-floor run. It did not contain this duct or shoring.',
                      'Collision geometry only; unseen visual enclosure or actual sensor hardware size is not validated.',
                      'Configured fixture geometry must not be used as sensor observations or navigation truth.']}
    for name in ('reference','measured'):
        path=a.geometry/(name+'_bounds.npz');report['sources_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        bounds=np.load(path,allow_pickle=False)['bounds'];report['datasets'][name]=evaluate(bounds)
        if name=='reference':
            standing=bounds[:1].copy();standing[:,:,:,0]+=1.6
            report['upright_at_roof_center']={'sample_geometry_overlaps':evaluate(standing)['sample_geometry_overlaps'],'source':'Counterfactual translation of reference standing frame0, not simulated execution'}
    src=ROOT/'punktnull/terrain/service_passage.py';report['sources_sha256'][str(src)]=hashlib.sha256(src.read_bytes()).hexdigest()
    (a.output/'fixture_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('scope','datasets','upright_at_roof_center')},indent=2))

if __name__=='__main__':main()
