"""Census classification tests; no simulation/ROS execution."""
import hashlib,json
from pathlib import Path
import tempfile
import unittest
from tools.summarize_runs import versioned_reports,row_for,census,series_summary

class CensusTest(unittest.TestCase):
    def write(self,p,data):p.write_text(json.dumps(data));return hashlib.sha256(p.read_bytes()).hexdigest()
    def test_numeric_version_order_keeps_old_failures(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name,value in [('evaluation_g1_lio.json',False),('evaluation_g1_lio_v2.json',True),('evaluation_g1_lio_v10.json',False)]:
                self.write(p/name,dict(audit_pass=value,checks={'bounded':value}))
            report,_=versioned_reports(p,'evaluation_g1_lio','audit_pass','x')
            self.assertEqual(report['verdict'],'FAIL');self.assertTrue(report['selected'].endswith('_v10.json'))
            self.assertEqual(len(report['history']),3);self.assertEqual(report['history'][1]['verdict'],'PASS')
    def test_malformed_latest_does_not_fall_back_to_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.write(p/'skill_evaluation.json',dict(accepted=True,checks={'ok':True}))
            (p/'skill_evaluation_v2.json').write_text('{')
            r,_=versioned_reports(p,'skill_evaluation','accepted','x')
            self.assertEqual(r['verdict'],'INVALID_REPORT')
    def test_duplicate_explicit_version_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name in ['skill_evaluation.json','skill_evaluation_v1.json']:self.write(p/name,dict(accepted=True))
            r,_=versioned_reports(p,'skill_evaluation','accepted','x')
            self.assertEqual(r['verdict'],'AMBIGUOUS_VERSION');self.assertIsNone(r['selected'])
    def test_missing_audit_never_confers_success(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.write(p/'manifest.json',dict(outcome='FOUND_AND_RETURNED'))
            r=row_for(p,'lio');self.assertEqual(r['status'],'UNVERIFIED_MISSING');self.assertTrue(r['completed'])
    def test_failed_manifest_conflicts_with_pass_report(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);sha=self.write(p/'manifest.json',dict(outcome='FAILED'))
            self.write(p/'evaluation_g1_lio.json',dict(audit_pass=True,checks={'ok':True},input_sha256={'manifest.json':sha}))
            self.assertEqual(row_for(p,'lio')['status'],'CONFLICTING_EXECUTION_AND_REPORT')
    def test_changed_manifest_does_not_reuse_old_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.write(p/'manifest.json',dict(outcome='FOUND_AND_RETURNED'))
            self.write(p/'evaluation_g1_lio.json',dict(audit_pass=True,input_sha256={'manifest.json':'wrong'}))
            self.assertEqual(row_for(p,'lio')['status'],'UNVERIFIED_STALE_MANIFEST_REPORT')
    def test_false_check_invalidates_claimed_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.write(p/'skill_evaluation.json',dict(accepted=True,checks={'tracking':False}))
            r,_=versioned_reports(p,'skill_evaluation','accepted','x')
            self.assertEqual(r['verdict'],'INCONSISTENT_REPORT')
    def test_running_entries_excluded_from_completed_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'lio_001').mkdir();(p/'lio_002').mkdir()
            self.write(p/'lio_001/manifest.json',dict(outcome='RUNNING'))
            self.write(p/'lio_002/manifest.json',dict(outcome='FAILED'))
            r=census(p);self.assertEqual(r['counts']['completed'],1);self.assertEqual(r['counts']['in_progress_or_unfinalized'],1)
            self.assertEqual(r['counts']['completed_statuses'],{'EXECUTION_FAILED':1})
    def test_explicit_series_keeps_missing_and_failed_trials_in_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for number,outcome,accepted in [('010','FOUND_AND_RETURNED',True),('011','FOUND_AND_RETURNED',False)]:
                p=root/('lio_'+number+'_test');p.mkdir();self.write(p/'manifest.json',dict(outcome=outcome))
                self.write(p/'evaluation_g1_lio.json',dict(audit_pass=accepted,checks={'gate':accepted},sensor_lio_integration_pass=True,physics={'timing':{'overall_ratio':1.,'windows':[]}}))
            r=census(root,['initial=010,011,012'])['explicit_series'][0]
            self.assertEqual(r['name'],'initial');self.assertEqual(r['expected_trials'],3);self.assertEqual(r['completed'],2);self.assertEqual(r['bounded_audit_passes'],1)
            self.assertEqual(r['missing_runs'],['012']);self.assertFalse(r['all_bounded_audits_pass'])
    def test_negative_fault_pass_never_becomes_completed_mission_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'lio_017_fault';p.mkdir()
            sha=self.write(p/'manifest.json',dict(outcome='FAILED'))
            folder=p/'fault_test_unavailable_pose';folder.mkdir()
            self.write(folder/'summary.json',dict(negative_test_pass=True,fault='unavailable_pose',mission_outcome='FAILED',
                mission_completed=False,live_human_acceptance=False,checks={'hold':True},source_sha256={'manifest.json':sha}))
            r=census(root);row=r['rows'][0]
            self.assertEqual(row['status'],'EXECUTION_FAILED')
            self.assertEqual(row['negative_fault_audits'][0]['audit']['verdict'],'PASS')
            self.assertEqual(r['counts']['separate_negative_fault_audits'],{'PASS':1})
            self.assertNotIn('AUDITED_PASS_IN_REPORTED_SCOPE',r['counts']['completed_statuses'])
    def test_fault_success_cannot_claim_live_or_completed_mission(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);sha=self.write(p/'manifest.json',dict(outcome='FAILED'))
            folder=p/'fault_test_explicit_abort';folder.mkdir()
            self.write(folder/'summary.json',dict(negative_test_pass=True,fault='explicit_abort',mission_outcome='FAILED',
                mission_completed=True,live_human_acceptance=True,source_sha256={'manifest.json':sha}))
            r=row_for(p,'lio')
            self.assertEqual(r['negative_fault_audits'][0]['audit']['verdict'],'INCONSISTENT_REPORT')
            self.assertEqual(r['status'],'EXECUTION_FAILED')
    def test_duplicate_series_reference_rejected(self):
        with self.assertRaises(ValueError):series_summary([],'initial=010,010')
if __name__=='__main__':unittest.main()
