#!/usr/bin/env python3
"""Generate a declared crawl target variant; no control/effort limit changes."""
import argparse
import hashlib
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from punktnull.skills.reference import ReferenceClip,write_skill_reference
from punktnull.skills.session import REGISTRY
from punktnull.skills.legacy import ground_module
from punktnull.skills.limits import load_joint_limits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--margin-rad',type=float,default=.03)
    args=parser.parse_args()
    if not .01<=args.margin_rad<=.06:parser.error('Bounded inward target margin must be.01..06rad')
    clip=ReferenceClip.load(REGISTRY['crawl'].source);limits=load_joint_limits()
    q=clip.data['joint_pos'][:,ground_module().MJ_TO_IL].copy();original=q.copy()
    joints=['left_ankle_pitch_joint','right_ankle_pitch_joint']
    for name in joints:
        i=limits.names.index(name)
        t=np.clip((q[:,i]-limits.lower[i])/.10,0.,1.)
        smooth=6*t**5-15*t**4+10*t**3
        q[:,i]+=args.margin_rad*(1-smooth)
    changes=[dict(operation='smooth inward ankle lower-stop target margin',joints=joints,
        maximum_delta_rad=float(np.max(abs(q-original))),requested_margin_rad=args.margin_rad,
        taper_interval_rad=.10,formula='delta=margin*(1-(6t^5-15t^4+10t^3)), t=clip((q-lower)/.10,0,1)',
        reason='Archived actual get-up pitch stop excursions while reference reaches exact lower stop',
        root_trajectory_unchanged=True,controller_unchanged=True,physical_limits_unchanged=True,
        actual_tracking_contact_and_limit_validation='pending actual simulation')]
    sources={str(clip.source/name):digest for name,digest in clip.hashes.items()}
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    new=write_skill_reference(args.output,clip.root,clip.quaternion,q,sources,changes,skill='crawl')
    print('Generated experimental reference:',new.source,'digest:',new.digest)


if __name__=='__main__':main()
