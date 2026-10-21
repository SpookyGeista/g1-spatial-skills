"""Pure evidence regression tests; synthetic data are never simulation proof."""
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from punktnull.mission_skill_evidence import native_rows,audit_authorization,audit_submission,passage_geometry,evaluate

class MissionSkillEvidenceTests(unittest.TestCase):
    def auth(self):
        phases=['PRIMING','ALIGNING','GROUND','RETURN_READY','COMPLETE']
        return [dict(monotonic=10+i*.02,owner='skill' if i<4 else 'none',owner_token='token' if i<4 else None,
            phase=p,released_owner_token='token' if i==4 else None,source='/clip',source_digest='digest',clearance_ok=True) for i,p in enumerate(phases)]

    def test_bound_complete_sequence_and_wrong_token_rejected(self):
        rows=self.auth();self.assertEqual(audit_authorization(rows,'/clip','digest',10,11)['token'],'token')
        rows[2]['owner_token']='different'
        with self.assertRaisesRegex(ValueError,'Owner changed'):audit_authorization(rows,'/clip','digest',10,11)

    def test_final_release_must_bind_retiring_token(self):
        rows=self.auth();rows[-1]['released_owner_token']='other'
        with self.assertRaisesRegex(ValueError,'bound'):audit_authorization(rows,'/clip','digest',10,11)

    def test_missing_release_stale_clearance_and_clock_gap_fail(self):
        rows=self.auth()[:-1]
        with self.assertRaises(ValueError):audit_authorization(rows,'/clip','digest',10,11)
        rows=self.auth();rows[2]['clearance_ok']=False
        with self.assertRaisesRegex(ValueError,'clearance'):audit_authorization(rows,'/clip','digest',10,11)
        rows=self.auth();rows[2]['monotonic']=10.7
        with self.assertRaisesRegex(ValueError,'clock'):audit_authorization(rows,'/clip','digest',10,11)

    def stream(self):
        return [dict(monotonic=10+i*.02,source_frame=i,wire_index=i,source_digest='digest') for i in range(3)]

    def test_held_endpoint_evidence_must_cover_release(self):
        rows=self.stream();self.assertEqual(audit_submission(rows,'digest',3,10,10.05)['samples'],3)
        with self.assertRaisesRegex(ValueError,'freshness'):audit_submission(rows,'digest',3,10,10.3)
        rows.append(dict(monotonic=10.06,source_frame=2,wire_index=3,source_digest='digest'))
        self.assertEqual(audit_submission(rows,'digest',3,10,10.07)['samples'],4)

    def test_skipped_or_restarted_reference_is_not_completion(self):
        rows=self.stream();rows[1]['source_frame']=2
        with self.assertRaisesRegex(ValueError,'cursor'):audit_submission(rows,'digest',3,10,10.05)
        rows=self.stream();rows[2]['wire_index']=1
        with self.assertRaisesRegex(ValueError,'cursor'):audit_submission(rows,'digest',3,10,10.05)

    def write_native(self,folder,indices=(0,1,2)):
        fields={'motion_name':'motion_name','encoder_mode':'encoder_mode_0','motion_playing':'playing_0'}
        values={'motion_name':'streamed','encoder_mode':0,'motion_playing':1}
        for name,field in fields.items():
            with (folder/(name+'.csv')).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['index','time_monotonic_ms',field]);writer.writeheader()
                for i in indices:writer.writerow(dict(index=i,time_monotonic_ms=10000+i*20,**{field:values[name]}))

    def test_native_all_rows_required_not_intersection(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.write_native(folder);self.assertEqual(len(native_rows(folder)),3)
            path=folder/'encoder_mode.csv';path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'Incomplete'):native_rows(folder)

    def test_native_frame_gap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.write_native(folder,(0,2))
            with self.assertRaisesRegex(ValueError,'gap'):native_rows(folder)

    def geometry(self):
        xyz=np.array([[0.,0.,.3],[0.,2.,.3],[0.,4.,.3]])
        lo=xyz-np.array([.1,.1,.1]);hi=xyz+np.array([.1,.1,.1]);bounds=np.stack([lo,hi],axis=1)[:,None]
        roof=dict(prim='roof',aabb_min=[-1,1.9,.65],aabb_max=[1,2.55,1.])
        return bounds,xyz,[roof]

    def test_full_body_clearance_and_exit_required(self):
        bounds,xyz,boxes=self.geometry();r=passage_geometry(bounds,xyz,boxes,[1.9,2.55],.65,'ingress')
        self.assertTrue(r['whole_body_entered_and_exited']);self.assertTrue(r['every_crossing_body_below_roof'])
        self.assertEqual(r['sampled_aabb_overlaps'],[])
        bounds[-1,0,0,1]=2.5
        self.assertFalse(passage_geometry(bounds,xyz,boxes,[1.9,2.55],.65,'ingress')['whole_body_entered_and_exited'])

    def test_upper_body_hit_cannot_pass_from_low_pelvis(self):
        bounds,xyz,boxes=self.geometry();bounds[1,0,1,2]=.8
        result=passage_geometry(bounds,xyz,boxes,[1.9,2.55],.65,'ingress')
        self.assertFalse(result['every_crossing_body_below_roof']);self.assertTrue(result['sampled_aabb_overlaps'])

    def test_missing_actual_evidence_cannot_enable_crawl_exception(self):
        with tempfile.TemporaryDirectory() as d:
            result=evaluate(d)
            self.assertFalse(result['accepted']);self.assertIn('complete_integrated_skill_evidence',result['failed_checks'])

if __name__=='__main__':unittest.main()
