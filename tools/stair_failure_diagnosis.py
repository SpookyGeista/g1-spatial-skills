#!/usr/bin/env python3
"""Post-run only: exact joint, step force and measured-foot failure report."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from punktnull.skills.limits import load_joint_limits
from punktnull.skills.stairs import URDF
from punktnull.terrain.kinematics import CollisionFK

def diagnose(run):
 run=Path(run);z=np.load(run/'sim/trace.npz');a=z['data'];c=list(z['columns']);col=lambda k:a[:,c.index(k)]
 get=lambda key,n:a[:,[c.index(key+str(i)) for i in range(n)]]
 L=load_joint_limits();q=get('q_',29);base=get('root_',13);released=col('support')==0
 bad=np.maximum(L.lower-q,q-L.upper);indices=np.argwhere((bad>1e-6)&released[:,None])
 violations=[dict(sample=int(i),joint=L.names[j],actual_rad=float(q[i,j]),lower_rad=float(L.lower[j]),upper_rad=float(L.upper[j]),
  excess_rad=float(bad[i,j]),sim_time=float(col('sim_time')[i])) for i,j in indices]
 contact=json.loads((run/'sim/wall_contacts.json').read_text());pairs=contact.get('contact_pairs_above_threshold',[])
 fk=CollisionFK(URDF,L.names);bounds=fk.bounds(fk.links(base[:,:3],base[:,3:7],q));events=[]
 for pair in pairs:
  t=pair['events'][0]['start_sim_time_s'];i=int(np.argmin(abs(col('sim_time')-t)))
  ids=[j for j,g in enumerate(fk.collisions) if g['link']==pair['body']]
  events.append(dict(body=pair['body'],obstacle=pair['obstacle_prim'],first_contact_time=t,
   closest_pose_time=float(col('sim_time')[i]),measured_toe_front_x=float(bounds[i,ids,1,0].max()),
   measured_foot_low_z=float(bounds[i,ids,0,2].min()),peak_force_vector_world_n=pair['peak_force_vector_world_n'],
   force_scope='Filtered ankle collider normal forces, not all-body contact'))
 return dict(schema=1,task='Failed actual staircase trial diagnosis, no acceptance change',
  actual_limit_violations=violations,first_contacts=events,
  overall_outcome=json.loads((run/'manifest.json').read_text())['outcome'],
  sources_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'sim/trace.npz',run/'sim/wall_contacts.npz',run/'sim/wall_contacts.json',run/'manifest.json')},
  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  geometry_source_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),
  limitations=['Collision FK is a CPU counterpart using logged poses; force vectors come from actual PhysX sensors.',
   'No continuous-time, hardware or all-body contact guarantee.'])
if __name__=='__main__':
 run=Path(sys.argv[1]);out=run/'diagnostics/failure_diagnosis.json';out.parent.mkdir(exist_ok=True)
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(diagnose(run),indent=2)+'\n');print(out)
