import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data.load_files import assign_splits, iter_split_windows, prepare_dataset, scan_records, validate_assignments
from data.load_files import build_datasets, load_mat_records, split_into_windows


def mat_file(path, records):
    with h5py.File(path, "w") as f:
        refs = f.create_dataset(path.stem, (len(records), 1), dtype=h5py.ref_dtype)
        for i, record in enumerate(records):
            refs[i, 0] = f.create_dataset(f"record_{i}", data=record).ref


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = ["Part_1.mat", "Part_2.mat"]
        self.source = []
        for i, length in enumerate((624, 625, 626, 1250, 625, 625, 625, 625)):
            self.source.append(np.arange(length * 3, dtype=float).reshape(length, 3) + i * 10000)
        self.source[5][0, 2] = np.nan
        mat_file(self.root / self.files[0], self.source[:4])
        mat_file(self.root / self.files[1], self.source[4:] + [self.source[1].copy()])
        self.config = dict(seed=2026, fractions=[0.7, 0.15, 0.15], policy_version="test",
                           reviewed_record_ids=["Part_2:record_4"])
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.archive = self.root / "dataset.npz"

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_dataset(self.root, self.files, self.archive, self.config_path)

    def test_windows_and_legacy_wrapper(self):
        for length, windows, tail in ((624, 0, 624), (625, 1, 0), (626, 1, 1), (1250, 2, 0)):
            record = np.arange(length * 3).reshape(length, 3)
            result = split_into_windows(record)
            self.assertEqual(result.shape, (windows, 625, 3))
            self.assertEqual(length - result.shape[0] * 625, tail)
        np.testing.assert_equal(next(load_mat_records(self.root / self.files[0])), self.source[0])

    def test_full_archive_rerun_and_split_access(self):
        expected = build_datasets(self.root, self.files)
        np.savez_compressed(self.archive, ppg=expected[0], abp=expected[1], ecg=expected[2])
        report = self.prepare()
        with np.load(self.archive) as archive:
            for channel, data in zip(("ppg", "abp", "ecg"), expected):
                np.testing.assert_equal(archive[channel], data)
            first_ids = archive["window_ids"].copy()
            first_splits = archive["split_assignments"].copy()
        self.assertEqual(report["records"][-1]["canonical_record_id"], "Part_1:record_1")
        self.assertEqual(report["records"][-1]["split"], "excluded")
        self.assertEqual(report["records"][1]["split"], "train")
        self.assertEqual(report["records"][5]["ecg_nan_count"], 1)
        groups, window_ids = {}, []
        for split in ("train", "validation", "test"):
            batches = list(iter_split_windows(split, 2, self.archive))
            self.assertTrue(batches)
            for batch in batches:
                self.assertLessEqual(len(batch.ppg), 2)
                self.assertEqual(batch.ppg.shape, batch.abp.shape)
                for key, source in zip(batch.window_ids, batch.source_record_ids):
                    self.assertNotEqual(source, "Part_2:record_4")
                    self.assertEqual(groups.setdefault(source, split), split)
                    row = np.flatnonzero(first_ids == key)[0]
                    np.testing.assert_equal(batch.ppg[list(batch.window_ids).index(key)], expected[0][row])
                    window_ids.append(key)
        self.assertEqual(len(window_ids), len(expected[0]) - 1)
        self.assertEqual(len(window_ids), len(set(window_ids)))
        self.prepare()
        with np.load(self.archive) as archive:
            np.testing.assert_equal(archive["window_ids"], first_ids)
            np.testing.assert_equal(archive["split_assignments"], first_splits)

    def test_old_archive_and_invalid_arguments(self):
        np.savez(self.archive, ppg=np.zeros((1, 625)))
        with self.assertRaisesRegex(ValueError, "Rebuild"):
            list(iter_split_windows("train", archive_path=self.archive))
        for split, size in (("invalid", 1), ("train", 0), ("train", True)):
            with self.assertRaises(ValueError):
                list(iter_split_windows(split, size, self.archive))

    def test_malformed_and_missing_reviewed_record(self):
        mat_file(self.root / self.files[0], [np.zeros((625, 2))])
        with self.assertRaisesRegex(ValueError, "Part_1:record_0"):
            scan_records(self.root, self.files, self.config)
        mat_file(self.root / self.files[0], self.source[:4])
        with self.assertRaisesRegex(ValueError, "absent"):
            scan_records(self.root, self.files, {**self.config, "reviewed_record_ids": ["unknown"]})

    def test_failed_rebuild_preserves_existing_archive(self):
        expected = build_datasets(self.root, self.files)
        expected[0][0, 0] += 1
        np.savez(self.archive, ppg=expected[0], abp=expected[1], ecg=expected[2])
        original = self.archive.read_bytes()
        with self.assertRaisesRegex(ValueError, "not replaced"):
            self.prepare()
        self.assertEqual(self.archive.read_bytes(), original)

    def test_duplicate_selection_is_order_independent(self):
        a = scan_records(self.root, self.files, self.config)[0]
        b = scan_records(self.root, list(reversed(self.files)), self.config)[0]
        self.assertEqual(a, b)

    def test_metadata_alignment_and_small_batch_sizes(self):
        self.prepare()
        with np.load(self.archive) as archive:
            content = {key: archive[key] for key in archive.files}
        content["window_ids"][0] = "Part_1:record_999:window_0"
        np.savez(self.archive, **content)
        with self.assertRaisesRegex(ValueError, "Misaligned window_ids"):
            list(iter_split_windows("train", 1, self.archive))

    def test_nonfinite_ppg_preserved_and_measured(self):
        self.source[1][0, 0] = np.nan
        self.source[1][1, 1] = np.inf
        mat_file(self.root / self.files[0], self.source[:4])
        report = self.prepare()
        self.assertEqual(report["records"][1]["ppg_nan_count"], 1)
        self.assertEqual(report["records"][1]["abp_infinity_count"], 1)
        with np.load(self.archive) as archive:
            self.assertTrue(np.isnan(archive["ppg"][0, 0]))
            self.assertTrue(np.isinf(archive["abp"][0, 1]))

    def test_cross_split_duplicate_metadata_rejected(self):
        self.prepare()
        with np.load(self.archive) as archive:
            content = {key: archive[key] for key in archive.files}
        report = json.loads(content["preparation_json"].item())
        report["records"][-1]["split"] = "test"
        content["preparation_json"] = np.array(json.dumps(report))
        content["split_assignments"][-1] = "test"
        np.savez(self.archive, **content)
        with self.assertRaisesRegex(ValueError, "duplicate/split"):
            validate_assignments(self.archive)


if __name__ == "__main__":
    unittest.main()
