"""CPU regression: release bookkeeping must not hide active source loss."""
import unittest
from tools.native_reference_evidence import native_reference_phase

class NativeReferenceEvidenceTests(unittest.TestCase):
    def rows(self):return [(10.06+i*.02,'streamed',0,1) for i in range(100)]
    def test_explicit_release_pause_is_bounded(self):
        rows=self.rows()+[(12.04,'streamed',0,0),(12.06,'planner_motion',0,0)]
        self.assertTrue(native_reference_phase(rows,10.,12.05)['active_phase'] is False)
        self.assertTrue(native_reference_phase(rows,10.,12.03)['accepted'])
    def test_play_loss_during_owned_phase_fails(self):
        rows=self.rows();rows[42]=(rows[42][0],'streamed',0,0)
        self.assertFalse(native_reference_phase(rows,10.,12.05)['accepted'])
    def test_wrong_encoder_or_missing_row_fails(self):
        for kind in ('encoder','gap','owner'):
            rows=self.rows()
            if kind=='encoder':rows[42]=(rows[42][0],'streamed',2,1)
            elif kind=='gap':del rows[40:44]
            else:rows[42]=(rows[42][0],'planner_motion',0,1)
            self.assertFalse(native_reference_phase(rows,10.,12.05)['accepted'])
    def test_late_or_excess_release_pause_fails(self):
        self.assertFalse(native_reference_phase(self.rows()+[(12.20,'streamed',0,0)],10.,12.05)['accepted'])
        self.assertFalse(native_reference_phase(self.rows()+[(12.06+i*.001,'streamed',0,0) for i in range(6)],10.,12.05)['accepted'])
