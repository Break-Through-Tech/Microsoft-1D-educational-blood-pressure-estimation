import unittest
from unittest.mock import patch
import numpy as np
from gradient_boosting import cleaning, data, evaluate, experiments, features, models


class ModelTests(unittest.TestCase):
    def test_training_only_imputation_scaling_and_missing_column_fallback(self):
        config = data.load_config()
        x = np.full((80, 17), np.nan)
        x[:, 0] = np.arange(80)
        x[0, 0] = np.nan
        y = np.arange(80.)
        ridge = models.fit_regressor(x, y, "ridge", config, [0, 1])
        self.assertEqual(ridge["columns"], [0])
        self.assertEqual(ridge["estimator"][0].statistics_[0], 40)
        mean = ridge["estimator"][1].mean_.copy()
        models.predict_regressor(ridge, np.full((2, 17), 100000.))
        np.testing.assert_array_equal(ridge["estimator"][1].mean_, mean)
        fallback = models.fit_regressor(x, y, "gb", config, [1])
        np.testing.assert_array_equal(models.predict_regressor(fallback, x), np.full(80, np.median(y)))

    def test_fixed_seed_predictions_and_target_specific_ties(self):
        config = data.load_config()
        x = np.random.default_rng(7).normal(size=(100, 17))
        y = 120 + 10 * x[:, 0]
        first = models.fit_regressor(x, y, "gb", config)
        second = models.fit_regressor(x, y, "gb", config)
        np.testing.assert_allclose(models.predict_regressor(first, x), models.predict_regressor(second, x), rtol=1e-6, atol=1e-6)
        rows = [{"target": t, "order": i, "macro_mae": score} for t, scores in
                (("sbp", [3, 1, 1]), ("dbp", [1, 2, 3])) for i, score in enumerate(scores)]
        self.assertEqual(experiments.choose_best(rows[::-1], "sbp")["order"], 1)
        self.assertEqual(experiments.choose_best(rows, "dbp")["order"], 0)

    def test_pooled_macro_and_whole_recording_intervals(self):
        records = np.array(["A", "A", "A", "B", "C", "C"])
        errors = np.array([1, 1, 1, 9, 4, 4])
        score = evaluate.regression_metrics(errors, np.zeros(6), records)
        self.assertAlmostEqual(score["pooled_mae"], 20 / 6)
        self.assertAlmostEqual(score["macro_mae"], 14 / 3)
        self.assertAlmostEqual(score["macro_rmse"], np.sqrt(98 / 3))
        duplicate = evaluate.regression_metrics(np.r_[errors, errors[:3]], np.zeros(9), np.r_[records, records[:3]])
        self.assertEqual(duplicate["macro_mae"], score["macro_mae"])
        self.assertNotEqual(duplicate["pooled_mae"], score["pooled_mae"])
        truth = np.tile([120., 80.], (6, 1))
        predictions = np.stack([truth + [4, 6], truth + [3, 5], truth + [2, 4], truth + [1, 3]])
        rows = evaluate.paired_intervals(truth, predictions, records, data.load_config(), "test")
        row = next(r for r in rows if r["baseline"] == "training_median" and r["target"] == "sbp")
        self.assertEqual((row["delta_mae"], row["ci_lower"], row["ci_upper"]), (-2., -2., -2.))

    def test_ppg_only_inference_keeps_unusable_rows(self):
        bundle = {"feature_names": list(features.FEATURE_NAMES), "target_names": ["sbp", "dbp"],
                  "models": {"training_mean": [120, 80], "training_median": [120, 80]}}
        for name in ("ridge", "gradient_boosting"):
            bundle["models"][name] = [{"columns": [], "constant": n} for n in (120, 80)]
        t = np.arange(625) / 125
        raw = np.stack([2 + np.sin(2 * np.pi * 1.2 * t), np.zeros(625)])
        with patch.object(cleaning, "clean_reference", side_effect=AssertionError("PPG prediction used ABP")):
            result = models.predict_ppg(bundle, raw, data.load_config())
        np.testing.assert_array_equal(result["ppg_eligible"], [True, False])
        self.assertTrue(np.isfinite(result["predictions"][:, 0]).all())
        self.assertTrue(np.isnan(result["predictions"][:, 1]).all())
