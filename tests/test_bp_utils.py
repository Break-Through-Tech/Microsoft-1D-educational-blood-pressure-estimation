import contextlib
import io
import unittest

import numpy as np

from data.bp_utils import extract_bp_labels, extract_bp_labels_batch, extract_bp_labels_v1

FS = 125


def make_abp(bpm=75, bump=0.0, start_phase=0.0, seconds=5, sbp=120.0, dbp=80.0):
    """Synthetic ABP: one sharp systolic peak per beat plus an optional smaller secondary bump.

    Each beat sits at `dbp` and rises to `sbp` at 15% of the beat. `bump` adds a smaller peak
    later in the beat, like a dicrotic wave. `start_phase` (0-1) shifts where in the beat the
    window starts, e.g. 0.3 starts on the downslope after a peak.
    """
    t = np.arange(int(seconds * FS)) / FS
    phase = (t * bpm / 60 + start_phase) % 1
    pulse = np.exp(-((phase - 0.15) / 0.06) ** 2)
    secondary = np.exp(-((phase - 0.45) / 0.04) ** 2)
    return dbp + (sbp - dbp) * pulse + bump * secondary


class ExtractBPLabelsTests(unittest.TestCase):
    def test_normal_signal_gives_expected_sbp_dbp(self):
        sbp, dbp, _, _ = extract_bp_labels(make_abp())
        self.assertAlmostEqual(sbp, 120.0, delta=0.5)
        self.assertAlmostEqual(dbp, 80.0, delta=0.5)

    def test_peak_and_beat_counts(self):
        # 75 bpm over 5 s: peaks at 0.12 s, 0.92 s, ... 4.92 s -> 7 peaks, 6 complete beats
        _, _, n_peaks, n_beats = extract_bp_labels(make_abp())
        self.assertEqual(n_peaks, 7)
        self.assertEqual(n_beats, 6)

    def test_secondary_bumps_are_not_counted_as_beats(self):
        abp = make_abp(bump=15.0)
        sbp, dbp, n_peaks, n_beats = extract_bp_labels(abp)
        self.assertEqual(n_peaks, 7)
        self.assertEqual(n_beats, 6)
        self.assertAlmostEqual(sbp, 120.0, delta=0.5)
        self.assertAlmostEqual(dbp, 80.0, delta=0.5)

        # The old method counts every bump as a peak, which pulls SBP down
        old_sbp, _, old_peaks, _ = extract_bp_labels_v1(abp)
        self.assertGreater(old_peaks, n_peaks)
        self.assertLess(old_sbp, sbp - 5)

    def test_fast_heart_rate(self):
        # 150 bpm over 5 s: peaks at 0.06 s, 0.46 s, ... 4.86 s -> 13 peaks, 12 complete beats
        abp = make_abp(bpm=150, bump=10.0)
        sbp, dbp, n_peaks, n_beats = extract_bp_labels(abp)
        self.assertEqual(n_peaks, 13)
        self.assertEqual(n_beats, 12)
        # At 125 Hz the true top falls between samples, so compare with the highest/lowest samples
        self.assertAlmostEqual(sbp, abp.max(), places=6)
        self.assertAlmostEqual(dbp, abp.min(), places=6)

    def test_window_starting_with_partial_beat(self):
        # Window starts on the downslope of a peak, so the first low is cut off by the window edge
        abp = make_abp(start_phase=0.18)
        self.assertGreater(abp[0], 100.0)
        sbp, dbp, n_peaks, n_beats = extract_bp_labels(abp)
        self.assertEqual(n_beats, n_peaks - 1)
        self.assertAlmostEqual(sbp, 120.0, delta=0.5)
        self.assertAlmostEqual(dbp, 80.0, delta=0.5)

    def test_flat_signal_returns_nan(self):
        sbp, dbp, n_peaks, n_beats = extract_bp_labels(np.full(625, 90.0))
        self.assertTrue(np.isnan(sbp))
        self.assertTrue(np.isnan(dbp))
        self.assertEqual((n_peaks, n_beats), (0, 0))

    def test_too_few_complete_beats_returns_nan(self):
        # 1.2 s at 75 bpm: 2 peaks -> only 1 complete beat
        sbp, dbp, n_peaks, n_beats = extract_bp_labels(make_abp(seconds=1.2))
        self.assertTrue(np.isnan(sbp))
        self.assertTrue(np.isnan(dbp))
        self.assertEqual((n_peaks, n_beats), (2, 1))

    def test_batch_matches_single_calls(self):
        windows = np.stack([make_abp(), make_abp(bpm=150, bump=10.0),
                            make_abp(start_phase=0.18), np.full(625, 90.0)])
        with contextlib.redirect_stdout(io.StringIO()):
            sbp_arr, dbp_arr, n_peaks_arr, n_beats_arr = extract_bp_labels_batch(windows)

        for i, window in enumerate(windows):
            sbp, dbp, n_peaks, n_beats = extract_bp_labels(window)
            np.testing.assert_equal(sbp_arr[i], sbp)
            np.testing.assert_equal(dbp_arr[i], dbp)
            self.assertEqual(n_peaks_arr[i], n_peaks)
            self.assertEqual(n_beats_arr[i], n_beats)


if __name__ == "__main__":
    unittest.main()
