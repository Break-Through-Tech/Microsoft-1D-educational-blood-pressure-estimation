"""Integrity and learning checks for the feature network; no real waveforms needed."""

import copy
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import sklearn
    import matplotlib
except ImportError:
    torch = None

import numpy as np
import pandas as pd


@unittest.skipIf(torch is None, "Install requirements.txt and CPU PyTorch")
class FeatureMLPTests(unittest.TestCase):
    def setUp(self):
        from models.train_feature_mlp import DEMO_LABEL_VERSION, make_demo_data
        self.table, self.names = make_demo_data()
        self.label_version = DEMO_LABEL_VERSION

    def prepare(self, table=None):
        from models.train_feature_mlp import prepare_rows
        return prepare_rows(self.table if table is None else table, self.names, self.label_version)

    def test_identity_and_target_leakage_are_rejected(self):
        from models.train_feature_mlp import prepare_rows
        duplicate = pd.concat([self.table, self.table.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate window_id"):
            self.prepare(duplicate)
        overlap = self.table.copy()
        overlap.loc[overlap["split"] == "validation", "source_record_id"] = self.table.loc[0, "source_record_id"]
        with self.assertRaisesRegex(ValueError, "recording appears"):
            self.prepare(overlap)
        for forbidden in ("sbp", "source_record_id", "ppg_quality_median", "dbp_target"):
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(ValueError, "cannot be an input"):
                prepare_rows(self.table, [forbidden], self.label_version)

    def test_exclusions_keep_ids_and_reasons(self):
        table = self.table.copy()
        table["usable"] = table["usable"].astype(str)
        table.loc[0, "usable"] = "False"
        table.loc[1, "sbp"] = np.nan
        table.loc[2, "sbp"] = table.loc[2, "dbp"]
        splits, log = self.prepare(table)
        self.assertEqual(log.loc[:2, "exclusion_reason"].tolist(),
                         ["marked_unusable", "nonfinite_label", "sbp_not_above_dbp"])
        self.assertFalse(log.loc[:2, "included"].any())
        self.assertEqual(splits["train"].iloc[0]["window_id"], table.iloc[3]["window_id"])
        self.assertEqual(len(log), len(table))

    def test_unknown_usable_or_label_version_stops_run(self):
        bad = self.table.copy()
        bad["usable"] = "maybe"
        with self.assertRaisesRegex(ValueError, "usable must"):
            self.prepare(bad)
        bad = self.table.copy()
        bad.loc[0, "label_version"] = "different_target"
        with self.assertRaisesRegex(ValueError, "label_version"):
            self.prepare(bad)

    def test_test_values_are_not_used(self):
        test = self.table.iloc[:8].copy()
        test["window_id"] = [f"test:{i}" for i in range(len(test))]
        test["source_record_id"], test["split"] = "heldout_record", "test"
        # These would fail conversion if held-out values were inspected.
        test[self.names + ["sbp", "dbp", "usable", "label_version"]] = "DO_NOT_USE"
        expected, _ = self.prepare()
        actual, log = self.prepare(pd.concat([self.table, test], ignore_index=True))
        for split in expected:
            pd.testing.assert_frame_equal(expected[split], actual[split], check_dtype=False)
        self.assertNotIn("test", log["split"].unique())

    def test_preprocessing_uses_training_only_and_constant_scale(self):
        from models.train_feature_mlp import fit_preprocessing, transform_features
        table = self.table.copy()
        table.loc[table["split"] == "train", self.names[-1]] = 7.0
        table.loc[table["split"] == "validation", self.names[0]] = 1e6
        splits, _ = self.prepare(table)
        state = fit_preprocessing(splits["train"], self.names)
        before = copy.deepcopy(state)
        expected_median = np.nanmedian(splits["train"][self.names[4]])
        self.assertEqual(state["feature_medians"][4], expected_median)
        self.assertEqual(state["feature_scale"][-1], 1.0)
        self.assertAlmostEqual(state["feature_mean"][0], splits["train"][self.names[0]].mean())
        transformed = transform_features(splits["validation"], state)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertGreater(transformed[:, 0].min(), 1e5)
        self.assertEqual(state, before)

    def test_missing_features_have_explicit_failure_modes(self):
        from models.train_feature_mlp import fit_preprocessing
        table = self.table.copy()
        table.loc[table["split"] == "train", self.names[-1]] = np.nan
        splits, _ = self.prepare(table)
        with self.assertRaisesRegex(ValueError, "no usable training values"):
            fit_preprocessing(splits["train"], self.names)
        table.loc[0, self.names] = np.nan
        with self.assertRaisesRegex(ValueError, "has no features"):
            self.prepare(table)
        table = self.table.copy()
        table.loc[0, self.names[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "Infinite train feature"):
            self.prepare(table)

    def test_learning_validation_and_saved_prediction_roundtrip(self):
        from models.train_feature_mlp import (TrainConfig, load_bundle, predict,
                                              run_experiment, train_model)
        config = TrainConfig(epochs=50, patience=10, learning_rate=0.01, dropout=0, threads=1)
        splits, _ = self.prepare()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "artificial_run"
            result, scores = run_experiment(self.table, self.names, self.label_version,
                                            "artificial_v1", directory, config, demo=True)
            history = result["history"]
            self.assertLess(history.iloc[-1]["train_mse_standardized"],
                            history.iloc[0]["train_mse_standardized"] / 5)
            metrics = scores.set_index("model")
            for target in ("sbp_mae", "dbp_mae"):
                self.assertLess(metrics.loc["feature_mlp", target],
                                metrics.loc["training_mean_diagnostic", target] / 2)
            best = int(history.loc[history["validation_mean_mae"].idxmin(), "epoch"])
            self.assertEqual(result["best_epoch"], best)
            weights = copy.deepcopy(result["model"].state_dict())
            before = copy.deepcopy(result["preprocessing"])
            # Future prediction inputs contain ONLY the feature columns.
            features = splits["validation"][self.names]
            predictions = predict(result["model"], features, before, batch_size=17)
            for key, value in result["model"].state_dict().items():
                self.assertTrue(torch.equal(value, weights[key]))
            self.assertEqual(result["preprocessing"], before)
            model, state, metadata = load_bundle(directory / "model.pt")
            np.testing.assert_allclose(predict(model, features, state, 17), predictions, rtol=0, atol=1e-6)
            self.assertTrue(metadata["provenance"]["demo"])
            self.assertIn("issues/29", metadata["issue_url"])
            self.assertEqual(len(pd.read_csv(directory / "window_usage.csv")), len(self.table))
            self.assertTrue((directory / "learning_curves.png").is_file())
            # Same CPU seed/settings reproduce a fresh run, including dropout/shuffling seeds.
            again = train_model(splits["train"], splits["validation"], self.names, config)
            np.testing.assert_array_equal(predict(again["model"], features, again["preprocessing"], 17), predictions)

    def test_missing_or_overlapping_splits_are_not_resplit(self):
        with self.assertRaisesRegex(ValueError, "No validation rows"):
            self.prepare(self.table[self.table["split"] == "train"])
        bad = self.table.copy()
        bad.loc[0, "split"] = "val"
        with self.assertRaisesRegex(ValueError, "split must"):
            self.prepare(bad)

    def test_early_stopping_restores_first_checkpoint_on_plateau(self):
        from models.train_feature_mlp import TrainConfig, train_model
        splits, _ = self.prepare()
        # A positive rate too small to change float32 weights gives an exact plateau.
        config = TrainConfig(epochs=20, patience=2, learning_rate=1e-40, dropout=0, threads=1)
        result = train_model(splits["train"], splits["validation"], self.names, config)
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(len(result["history"]), 3)

    def test_csv_roundtrip_and_duplicate_headers(self):
        from models.train_feature_mlp import read_feature_table
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.csv"
            self.table.to_csv(path, index=False)
            splits, _ = self.prepare(read_feature_table(path))
            self.assertEqual(len(splits["train"]), 512)
            path.write_text("window_id,window_id\na,b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate column"):
                read_feature_table(path)


if __name__ == "__main__":
    unittest.main()
