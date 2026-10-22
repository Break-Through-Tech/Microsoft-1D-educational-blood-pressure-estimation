import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CNNModelTests(unittest.TestCase):
    def test_batch_output_shape(self):
        from models.cnn_model import PPGCNN

        torch.manual_seed(0)
        output = PPGCNN()(torch.zeros(8, 1, 625))
        self.assertEqual(tuple(output.shape), (8, 2))


# Rows deliberately interleave recordings and splits: (source ID, split) for each window.
FAKE_ROWS = [("A", "train"), ("C", "validation"), ("B", "train"), ("A", "train"),
             ("D", "test"), ("C", "validation")]


def write_fake_cleaned(path, rows=FAKE_ROWS):
    """Row i has PPG filled with i, SBP 100 + i, DBP 60 + i, so alignment errors are visible."""
    n = len(rows)
    local = {}
    window_ids = []
    for source, _ in rows:
        window_ids.append(f"{source}:window_{local.get(source, 0)}")
        local[source] = local.get(source, 0) + 1
    np.savez(path,
             ppg=np.repeat(np.arange(n, dtype=np.float32)[:, None], 625, axis=1),
             abp=np.zeros((n, 625), np.float32),
             sbp=100 + np.arange(n, dtype=np.float32), dbp=60 + np.arange(n, dtype=np.float32),
             window_ids=np.array(window_ids), source_record_ids=np.array([s for s, _ in rows]),
             split_assignments=np.array([split for _, split in rows]))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CleanedDatasetLoadingTests(unittest.TestCase):
    def setUp(self):
        from models.train_cnn import load_cleaned_dataset

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        write_fake_cleaned(self.directory / "cleaned.npz")
        self.arrays, self.records = load_cleaned_dataset(self.directory / "cleaned.npz")

    def test_selects_requested_recordings_with_aligned_labels_and_ids(self):
        from models.train_cnn import load_windows

        ppg, labels, window_ids, sources = load_windows(self.arrays, "train", {"A"})
        np.testing.assert_array_equal(ppg[:, 0], [0, 3])
        np.testing.assert_array_equal(labels, [[100, 60], [103, 63]])
        self.assertEqual(window_ids.tolist(), ["A:window_0", "A:window_1"])
        self.assertEqual(sources.tolist(), ["A", "A"])

        ppg, labels, window_ids, sources = load_windows(self.arrays, "validation", {"C"})
        np.testing.assert_array_equal(ppg[:, 0], [1, 5])
        np.testing.assert_array_equal(labels[:, 0], [101, 105])
        self.assertEqual(window_ids.tolist(), ["C:window_0", "C:window_1"])

    def test_rejects_test_split(self):
        from models.train_cnn import load_windows

        with self.assertRaisesRegex(ValueError, "only the train and validation"):
            load_windows(self.arrays, "test", {"D"})

    def test_rejects_recording_from_another_split(self):
        from models.train_cnn import load_windows

        with self.assertRaisesRegex(ValueError, "no validation windows"):
            load_windows(self.arrays, "validation", {"A"})

    def test_records_match_sample_recordings_format(self):
        self.assertEqual(self.records, [
            {"source_record_id": "A", "split": "train", "window_count": 2},
            {"source_record_id": "B", "split": "train", "window_count": 1},
            {"source_record_id": "C", "split": "validation", "window_count": 2},
            {"source_record_id": "D", "split": "test", "window_count": 1},
        ])
        self.assertNotIn("abp", self.arrays)

    def test_recording_in_two_splits_is_rejected(self):
        from models.train_cnn import load_cleaned_dataset

        rows = [("A", "train"), ("A", "validation")]
        write_fake_cleaned(self.directory / "crossing.npz", rows)
        with self.assertRaisesRegex(ValueError, "more than one split: A"):
            load_cleaned_dataset(self.directory / "crossing.npz")


if __name__ == "__main__":
    unittest.main()
