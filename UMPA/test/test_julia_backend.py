import os
import unittest

import numpy as np

os.environ.setdefault("UMPA_DISABLE_JULIA", "1")

import UMPA


class JuliaBackendTest(unittest.TestCase):
    def test_match_df_outputs(self):
        sample = np.ones((2, 10, 10), dtype=float)
        ref = np.ones((2, 10, 10), dtype=float) * 2
        model = UMPA.model.UMPAModelDF(sample, ref, window_size=0, max_shift=0)
        result = model.match(step=2)
        self.assertIn("df", result)
        self.assertEqual(result["T"].shape, (5, 5))
        self.assertAlmostEqual(result["T"][0, 0], 0.5)
        coverage = model.coverage(step=2)
        self.assertEqual(coverage.shape, result["T"].shape)

    def test_match_nodf_outputs(self):
        sample = np.ones((1, 8, 8), dtype=float)
        ref = np.ones((1, 8, 8), dtype=float)
        model = UMPA.model.UMPAModelNoDF(sample, ref, window_size=0, max_shift=0)
        result = model.match()
        self.assertNotIn("df", result)
        self.assertEqual(result["T"].shape, (8, 8))
