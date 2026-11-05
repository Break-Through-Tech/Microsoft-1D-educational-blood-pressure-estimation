import contextlib
import copy
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from gradient_boosting import data, evaluate, experiments, models, report
from gradient_boosting.tests.test_data import small_dataset


class PipelineTests(unittest.TestCase):
    def test_small_archive_runs_experiments_saves_model_then_tests_without_refitting(self):
        with tempfile.TemporaryDirectory(dir=data.OUTPUT_ROOT) as temp:
            config = small_dataset(Path(temp))
            with patch.object(data, "ProcessPoolExecutor", ThreadPoolExecutor), contextlib.redirect_stdout(io.StringIO()):
                for split in ("train", "validation"):
                    data.prepare_split(config, split)
                for name in ("features", "tuning"):
                    result = experiments.run_experiment(config, name)
                    self.assertFalse(result["uses_test_data"])
                self.assertFalse(data.table_folder(config, "test").exists())
                models.train(config)
                before = data.checksum(data.run_dir(config) / "model.joblib")
                with patch.object(models, "fit_regressor", side_effect=AssertionError("test refitted")), \
                        patch.object(experiments, "run_experiment", side_effect=AssertionError("test tuned")):
                    models.test(config)
                    models.verify_predictions(config, "validation")
                    result = models.verify_predictions(config, "test")
                    models.test(config)
                self.assertEqual(before, data.checksum(data.run_dir(config) / "model.joblib"))
                page = Path(temp) / "results.html"
                summary = report.write_report(config, page)
                self.assertEqual(page.read_text().count("<section>"), 3)
                self.assertIn("What I tried", page.read_text())
                self.assertIn("What I learned", page.read_text())
                self.assertEqual(result["coverage"]["source_windows"], 4)
                changed = copy.deepcopy(config)
                changed["gradient_boosting"]["learning_rate"] = .8
                with self.assertRaises(ValueError):
                    models.load_model(changed)
                with patch.object(models, "package_versions", return_value={}):
                    with self.assertRaises(ValueError):
                        models.load_model(config)
                with (data.run_dir(config) / "model.joblib").open("ab") as output:
                    output.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "saved model changed"):
                    models.load_model(config)

    def test_missing_reference_stays_predicted_unknown_and_unscored(self):
        with tempfile.TemporaryDirectory(dir=data.OUTPUT_ROOT) as temp:
            config = small_dataset(Path(temp))
            with patch.object(data, "ProcessPoolExecutor", ThreadPoolExecutor), contextlib.redirect_stdout(io.StringIO()):
                data.prepare_split(config, "validation")
            table = data.load_table(config, "validation")
            i = int(np.flatnonzero(table["included"])[0])
            table["reference_valid"][i] = table["included"][i] = False
            table["sbp"][i] = table["dbp"][i] = np.nan
            bundle = {"feature_names": list(data.features.FEATURE_NAMES), "target_names": ["sbp", "dbp"],
                      "models": {"training_mean": [120, 80], "training_median": [120, 80]}}
            for name in ("ridge", "gradient_boosting"):
                bundle["models"][name] = [{"columns": [], "constant": n} for n in (120, 80)]
            payload = models.prediction_table(bundle, table)
            result = evaluate.evaluate_predictions(payload, table, config, "validation")
            self.assertEqual(len(payload["window_ids"]), table["ppg_eligible"].sum())
            self.assertTrue(np.isfinite(payload["predictions"]).all())
            self.assertTrue(all(row["retained_windows"] == 0 and row["source_windows"] >= 1
                                for row in result["pressure_ranges"] if row["band"] == "unknown"))
            self.assertTrue(all(row["n_windows"] == table["included"].sum() for row in result["metrics"]))
            with self.assertRaisesRegex(ValueError, "wrong split"):
                evaluate.evaluate_predictions(payload, table, config, "test")
            payload["window_ids"] = payload["window_ids"][::-1]
            with self.assertRaisesRegex(ValueError, "do not match"):
                evaluate.evaluate_predictions(payload, table, config, "validation")
