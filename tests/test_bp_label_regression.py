"""Regression checks: PR #46's refactored label code must match the PR #41 method on main."""

import unittest

import numpy as np
from scipy.signal import find_peaks

from data.bp_utils import complete_beats, extract_bp_labels

FS = 125


def pr41_extract_bp_labels(abp_window, fs=FS):
    """Frozen copy of extract_bp_labels from main (PR #41), used as the reference answer."""
    abp_window = np.asarray(abp_window, dtype=np.float64)
    low, high = np.percentile(abp_window, [5, 95])
    prominence = max(10.0, 0.5 * (high - low))
    distance = int(round(0.33 * fs))
    peaks, _ = find_peaks(abp_window, prominence=prominence, distance=distance)
    beat_sbp = []
    beat_dbp = []
    for previous_peak, peak in zip(peaks[:-1], peaks[1:]):
        beat_sbp.append(abp_window[peak])
        beat_dbp.append(abp_window[previous_peak:peak].min())
    n_peaks = len(peaks)
    n_beats = len(beat_sbp)
    if n_beats < 2:
        return np.nan, np.nan, n_peaks, n_beats
    return float(np.median(beat_sbp)), float(np.median(beat_dbp)), n_peaks, n_beats


def random_abp(rng, fs=FS, seconds=5):
    """Synthetic ABP with random rate, pressures, secondary bump, start phase, drift and noise."""
    t = np.arange(int(seconds * fs)) / fs
    bpm = rng.uniform(40, 170)
    dbp = rng.uniform(40, 100)
    sbp = dbp + rng.uniform(15, 90)
    phase = (t * bpm / 60 + rng.uniform()) % 1
    pulse = np.exp(-((phase - 0.15) / 0.06) ** 2)
    bump = rng.uniform(0, 0.4) * (sbp - dbp) * np.exp(-((phase - 0.45) / 0.04) ** 2)
    drift = rng.uniform(-5, 5) * t / seconds
    noise = rng.normal(0, rng.uniform(0, 3), t.size)
    return dbp + (sbp - dbp) * pulse + bump + drift + noise


class LabelRefactorRegressionTests(unittest.TestCase):
    def assert_same_labels(self, window, fs=FS):
        expected = pr41_extract_bp_labels(window, fs)
        actual = extract_bp_labels(window, fs)
        np.testing.assert_equal(actual[:2], expected[:2])  # exact, and NaN matches NaN
        self.assertEqual(actual[2:], expected[2:])

    def test_matches_pr41_on_varied_synthetic_windows(self):
        rng = np.random.default_rng(41)
        for i in range(300):
            with self.subTest(window=i):
                self.assert_same_labels(random_abp(rng))

    def test_matches_pr41_on_edge_cases(self):
        rng = np.random.default_rng(7)
        cases = {
            "flat": np.full(625, 90.0),
            "one_beat": random_abp(rng, seconds=0.9),
            "pure_noise": 80 + rng.normal(0, 2, 625),
            "large_noise": 80 + rng.normal(0, 30, 625),
        }
        for name, window in cases.items():
            with self.subTest(case=name):
                self.assert_same_labels(window)

    def test_matches_pr41_at_another_sampling_rate(self):
        rng = np.random.default_rng(250)
        for i in range(20):
            with self.subTest(window=i):
                self.assert_same_labels(random_abp(rng, fs=250), fs=250)

    def test_lows_are_the_minimum_between_neighbouring_peaks(self):
        rng = np.random.default_rng(3)
        for i in range(50):
            window = random_abp(rng)
            peaks, lows = complete_beats(window)
            with self.subTest(window=i):
                self.assertEqual(len(lows), max(len(peaks) - 1, 0))
                for a, b, low in zip(peaks[:-1], peaks[1:], lows):
                    self.assertEqual(window[low], window[a:b].min())

    def test_invalid_sampling_rate_is_rejected(self):
        for fs in (0, -125, np.nan, np.inf):
            with self.subTest(fs=fs):
                with self.assertRaisesRegex(ValueError, "fs must be positive and finite"):
                    complete_beats(np.full(625, 90.0), fs)


if __name__ == "__main__":
    unittest.main()
