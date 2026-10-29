import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.naive_baseline_model import (
    load_data,
    select_recordings,
    valid_labels,
)


class MedianBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive_path = Path(self.temp.name) / "cleaned.npz"
        np.savez(
            self.archive_path,
            ppg=np.ones((5, 3)),
            sbp=np.array([120.0, 140.0, 130.0, 125.0, 150.0]),
            dbp=np.array([80.0, 90.0, 85.0, 82.0, 95.0]),
            source_record_ids=np.array(["A", "B", "C", "D", "E"]),
            split_assignments=np.array(
                ["train", "train", "train", "validation", "validation"]
            ),
        )

    def test_medians_are_fitted_on_train_and_scored_on_validation(self):
        arrays = load_data(self.archive_path)
        train_ids, validation_ids = select_recordings(arrays, 2026, 3, 2)
        training_labels = valid_labels(arrays, "train", train_ids)
        validation_labels = valid_labels(arrays, "validation", validation_ids)
        medians = np.median(training_labels, axis=0)

        np.testing.assert_array_equal(medians, [130.0, 85.0])
        np.testing.assert_array_equal(validation_labels, [[125.0, 82.0], [150.0, 95.0]])
        errors = medians - validation_labels
        self.assertAlmostEqual(np.mean(np.abs(errors[:, 0])), 12.5)
        self.assertAlmostEqual(np.mean(np.abs(errors[:, 1])), 6.5)

    def test_invalid_windows_are_excluded(self):
        with np.load(self.archive_path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["sbp"][0] = np.nan
        arrays["ppg"][1, 0] = np.nan
        np.savez(self.archive_path, **arrays)

        loaded = load_data(self.archive_path)
        labels = valid_labels(loaded, "train", {"A", "B", "C"})
        np.testing.assert_array_equal(labels, [[130.0, 85.0]])

    def test_selects_whole_recordings_from_each_saved_split(self):
        arrays = load_data(self.archive_path)
        train_ids, validation_ids = select_recordings(arrays, 2026, 2, 1)

        self.assertEqual(len(train_ids), 2)
        self.assertEqual(len(validation_ids), 1)
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids <= {"A", "B", "C"})
        self.assertTrue(validation_ids <= {"D", "E"})

    def test_rejects_recording_assigned_to_multiple_splits(self):
        with np.load(self.archive_path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["source_record_ids"][3] = "A"
        np.savez(self.archive_path, **arrays)

        with self.assertRaisesRegex(ValueError, "more than one split"):
            load_data(self.archive_path)


if __name__ == "__main__":
    unittest.main()
