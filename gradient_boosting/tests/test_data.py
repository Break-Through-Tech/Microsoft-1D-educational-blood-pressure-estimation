import contextlib
import io
import json
import tempfile
import unittest
import zipfile
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from data.load_files import prepare_dataset
from tests.test_dataset_splits import mat_file
from gradient_boosting import data


def small_dataset(folder):
    config = data.load_config()
    raw = folder / "raw"
    raw.mkdir()
    t = np.arange(1250) / 125
    records = []
    for i in range(8):
        ppg = 2 + (1 + .05 * i) * np.sin(2 * np.pi * 1.2 * t)
        abp = 90 + i + 25 * np.sin(2 * np.pi * 1.2 * t)
        if i == 0:
            ppg[:] = 0
        if i == 1:
            abp[:] = 90
        records.append(np.column_stack([ppg, abp, np.cos(t)]))
    mat_file(raw / "Part_1.mat", records + [records[2].copy()])
    split = folder / "split.json"
    split.write_text(json.dumps({"seed": 2026, "fractions": [.5, .25, .25], "reviewed_record_ids": []}))
    config["paths"].update(raw_dir=str(raw), split_config=str(split), prepared_archive=str(folder / "prepared.npz"), output_dir=str(folder / "run"))
    config.update(workers=1, batch_size=3, bootstrap_samples=25)
    config["tuning"] = {"learning_rates": [.05, .1], "leaf_counts": [15]}
    with contextlib.redirect_stdout(io.StringIO()):
        prepare_dataset(raw, ["Part_1.mat"], folder / "prepared.npz", split)
    return config


class DataTests(unittest.TestCase):
    def test_complete_coverage_failures_resumption_and_raw_reproduction(self):
        with tempfile.TemporaryDirectory(dir=data.OUTPUT_ROOT) as temp:
            config = small_dataset(Path(temp))
            with patch.object(data, "ProcessPoolExecutor", ThreadPoolExecutor), contextlib.redirect_stdout(io.StringIO()):
                for split in ("train", "validation"):
                    data.prepare_split(config, split)
                    data.verify_raw_samples(config, split, 20)
                    with patch.object(data, "process_chunk", side_effect=AssertionError("recleaned")):
                        data.prepare_split(config, split)
                folder = data.table_folder(config, "train")
                (folder / "metadata.json").unlink()
                with patch.object(data, "process_chunk", side_effect=AssertionError("recleaned")):
                    data.prepare_split(config, "train")
            tables = [data.load_table(config, s) for s in ("train", "validation")]
            self.assertEqual(sum(len(t["X"]) for t in tables), 12)
            self.assertTrue(any((~t["ppg_eligible"]).any() for t in tables))
            self.assertTrue(any((t["ppg_eligible"] & ~t["reference_valid"]).any() for t in tables))
            self.assertFalse(any("record_8:" in w for t in tables for w in t["window_ids"]))

    def test_test_preparation_requires_a_saved_model(self):
        with tempfile.TemporaryDirectory(dir=data.OUTPUT_ROOT) as temp:
            config = small_dataset(Path(temp))
            with patch.object(data, "iter_split_windows", side_effect=AssertionError("test requested before saving model")):
                with self.assertRaisesRegex(ValueError, "Train and save"):
                    data.prepare_split(config, "test")

    def test_changed_ledger_and_wrong_ids_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=data.OUTPUT_ROOT) as temp:
            config = small_dataset(Path(temp))
            with patch.object(data, "ProcessPoolExecutor", ThreadPoolExecutor), contextlib.redirect_stdout(io.StringIO()):
                data.prepare_split(config, "train")
            table = data.load_table(config, "train")
            table["window_ids"][0] = table["window_ids"][1]
            with self.assertRaises(ValueError):
                data.validate_table(table, "train", data.source_report(config))
            ledger = data.table_folder(config, "train") / "batch_00000.csv"
            with ledger.open("a") as output:
                output.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "ledger changed"):
                data.load_table(config, "train")

    def test_shared_output_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            data.output_path("data/processed_dataset.npz")

    def test_original_backup_is_checked_before_reusing_old_results(self):
        with tempfile.TemporaryDirectory(dir=data.OUTPUT_ROOT) as temp:
            folder = Path(temp)
            config = small_dataset(folder)
            archive = folder / "old_source.zip"
            original = b"original experiment source"
            with zipfile.ZipFile(archive, "x") as saved:
                saved.writestr("gradient_boosting/features.py", original)
            old = folder / "old_run"
            data.save_json(old / "frozen/freeze.json", {"code": {
                "gradient_boosting/features.py": hashlib.sha256(original).hexdigest()}})
            data.save_json(data.run_dir(config) / "imported_run.json", {
                "processing_signature": data.processing_signature(config), "source_backup": str(archive),
                "backup_sha256": data.checksum(archive), "original_run": str(old),
                "original_freeze_sha256": data.checksum(old / "frozen/freeze.json")})
            self.assertIsNotNone(data.check_backup(config))
            with archive.open("ab") as output:
                output.write(b"changed")
            with self.assertRaisesRegex(ValueError, "backup changed"):
                data.check_backup(config)
