import os
import unittest
from unittest import mock

import numpy as np

import UMPA


def _reset_backend():
    UMPA.model.reset_backend()


class JuliaBackendTest(unittest.TestCase):
    def test_match_df_outputs(self):
        with mock.patch.dict(os.environ, {"UMPA_DISABLE_JULIA": "1"}):
            _reset_backend()
            sample = np.ones((2, 10, 10), dtype=float)
            ref = np.ones((2, 10, 10), dtype=float) * 2
            model = UMPA.model.UMPAModelDF(sample, ref, window_size=1, max_shift=1)
            result = model.match(step=2)
            self.assertIn("df", result)
            expected_shape = (
                1 + (model.extent[0] - 1) // 2,
                1 + (model.extent[1] - 1) // 2,
            )
            self.assertEqual(result["T"].shape, expected_shape)
            self.assertAlmostEqual(result["T"][0, 0], 0.5)
            coverage = model.coverage(step=2)
            self.assertEqual(coverage.shape, result["T"].shape)

    def test_match_nodf_outputs(self):
        with mock.patch.dict(os.environ, {"UMPA_DISABLE_JULIA": "1"}):
            _reset_backend()
            sample = np.ones((1, 8, 8), dtype=float)
            ref = np.ones((1, 8, 8), dtype=float)
            model = UMPA.model.UMPAModelNoDF(sample, ref, window_size=0, max_shift=0)
            result = model.match()
            self.assertNotIn("df", result)
            self.assertEqual(result["T"].shape, (8, 8))
