import unittest
from unittest.mock import patch
import numpy as np
from gradient_boosting import features


class FeatureTests(unittest.TestCase):
    def test_known_triangle_has_correct_units_and_shape(self):
        values, reason = features.pulse_features([0, 1, 2, 1, 0], 0, 2, 4, fs=2)
        self.assertEqual(reason, "ok")
        np.testing.assert_allclose(values, [1, 2, .5, 1, .5, .5, .25, 2, .5, 2, 2], atol=1e-12)
        self.assertEqual(len(features.FEATURE_NAMES), 17)

    def test_normalized_shape_resists_fixed_boundary_gain_and_offset(self):
        signal = np.array([0, 1, 2, 1, 0]) + np.linspace(5, 7, 5)
        first, _ = features.pulse_features(signal, 0, 2, 4, fs=2)
        second, _ = features.pulse_features(signal * 3.2 + 7.5, 0, 2, 4, fs=2)
        np.testing.assert_allclose(first[[2, 5, 6, 8, 9, 10]], second[[2, 5, 6, 8, 9, 10]], atol=1e-12)

    def test_incomplete_pulses_and_detector_failures_keep_missing_values(self):
        self.assertTrue(np.isnan(features.pulse_features([0, 1, 0], 0, 1, 3)[0]).all())
        signal = np.sin(np.linspace(0, 10, 625))
        with patch.object(features.nk, "ppg_findpeaks", side_effect=ValueError("detector failed")):
            result = features.extract_window(signal)
        self.assertTrue(np.isfinite(result["values"][:3]).all())
        self.assertTrue(np.isnan(result["values"][3:]).all())
        self.assertEqual(result["status"], "peak_detection_error")

    def test_basic_features_follow_original_peak_formula(self):
        signal = np.array([0., 1, 0, 2, 0, 1, 0])
        actual = features.from_peaks(signal, [1, 3, 5], fs=2)
        np.testing.assert_allclose(actual, [signal.mean(), signal.std(), 2, 1, 60, 1.5])
