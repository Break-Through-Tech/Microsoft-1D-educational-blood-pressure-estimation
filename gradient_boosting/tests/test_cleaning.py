import unittest
import numpy as np
from gradient_boosting import cleaning, data


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.config = data.load_config()
        t = np.arange(625) / 125
        self.ppg = 2 + np.sin(2 * np.pi * 1.2 * t)
        self.abp = 90 + 25 * np.sin(2 * np.pi * 1.2 * t)

    def test_interpolation_and_raw_quality_gates(self):
        np.testing.assert_allclose(cleaning.interpolate_zeros([0, 2, 0, 4, 0]), [2, 2, 3, 4, 4])
        self.assertEqual(cleaning.raw_status(np.zeros(625), self.config)[1]["reason"], "excess_zeros")
        self.assertEqual(cleaning.raw_status(np.ones(625), self.config)[1]["reason"], "flat_signal")
        bad = self.ppg.copy()
        bad[0] = np.nan
        self.assertEqual(cleaning.raw_status(bad, self.config)[1]["reason"], "nonfinite_signal")
        with self.assertRaises(ValueError):
            cleaning.raw_status(self.ppg[:10], self.config)

    def test_ppg_and_reference_checks_are_independent(self):
        ppg, reference, included = cleaning.clean_window(self.ppg, self.abp, self.config)
        self.assertTrue(included)
        other, invalid, included = cleaning.clean_window(self.ppg, np.ones(625) * 90, self.config)
        self.assertTrue(other["eligible"])
        self.assertFalse(invalid["valid"])
        self.assertFalse(included)
        np.testing.assert_array_equal(ppg["cleaned"], other["cleaned"])
        failed, good, included = cleaning.clean_window(np.zeros(625), self.abp, self.config)
        self.assertFalse(failed["eligible"])
        self.assertTrue(good["valid"])
        self.assertFalse(included)

    def test_labels_use_the_shared_complete_beat_detector(self):
        from data.bp_utils import extract_bp_labels
        result = cleaning.clean_reference(self.abp, self.config)
        sbp, dbp, _, _ = extract_bp_labels(result["cleaned"])
        np.testing.assert_allclose([result["sbp"], result["dbp"]], [sbp, dbp], atol=1e-4)
