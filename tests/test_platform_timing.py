import unittest
import numpy as np
from tools.platform_timing_audit import timing_metrics
class PlatformTimingTests(unittest.TestCase):
    def test_real_time_clock_passes(self):
        x=np.arange(0,12,.005);r=timing_metrics(x,x,1,11)
        self.assertTrue(r['accepted']);self.assertEqual(len(r['full_5s_windows']),2)
    def test_aggregate_cannot_hide_slow_window(self):
        x=np.arange(0,12,.005);sim=np.where(x<=6,x*.90,5.4+(x-6)*1.1)
        r=timing_metrics(x,sim,1,11);self.assertAlmostEqual(r['ratio'],1.)
        self.assertFalse(r['accepted'])
    def test_resets_or_missing_bounds_fail(self):
        x=np.arange(0,12,.005)
        self.assertFalse(timing_metrics(x,x,1,13)['accepted'])
        y=x.copy();y[50]=0
        self.assertFalse(timing_metrics(x,y,1,11)['accepted'])
