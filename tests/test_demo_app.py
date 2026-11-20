"""Exercise the live UI against the small, tracked deployment bundle."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "demo/assets"
LIVE_MODELS = {"resnet_l1": "ResNet (L1)", "cnn_c": "Short CNN"}


def bundled_examples():
    import numpy as np
    with np.load(BUNDLE / "samples/ppg.npz", allow_pickle=False) as examples:
        return examples["ppg"], examples["window_ids"].astype(str)


def shown_predictions(app):
    """Map each prediction column's model heading to its displayed metric tiles."""
    shown = {}
    for column in app.columns:
        headings = [element.value.lstrip("# ") for element in column.markdown]
        if headings and headings[0] in LIVE_MODELS.values():
            shown[headings[0]] = {metric.label: metric.value for metric in column.metric}
    return shown


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Install demo/requirements.txt for UI tests")
class DemoTests(unittest.TestCase):
    def test_live_prediction_and_example_switch(self):
        import numpy as np
        from streamlit.testing.v1 import AppTest
        from models.selected_models import SelectedPredictor, load_selected_model

        signals, ids = bundled_examples()
        direct = {name: load_selected_model(name, seed=2026, artifact_dir=BUNDLE) for name in LIVE_MODELS}
        original = SelectedPredictor.predict_ppg
        with patch.object(SelectedPredictor, "predict_ppg", autospec=True, side_effect=original) as calls:
            app = AppTest.from_file(str(ROOT / "demo/streamlit_app.py"), default_timeout=90).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            # No placeholder results before Predict.
            self.assertEqual(len(app.metric), 0)
            previous = None
            for example in (0, 1):
                if example:
                    app.selectbox[0].select(example).run()
                    # A previous example's prediction must never appear under a new waveform.
                    self.assertEqual(len(app.metric), 0)
                calls.reset_mock()
                app.button[0].click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(len(app.metric), 4)

                # Each model ran once, on the same selected waveform and ID.
                self.assertEqual(sorted(call.args[0].model_name for call in calls.call_args_list),
                                 sorted(LIVE_MODELS))
                for call in calls.call_args_list:
                    np.testing.assert_array_equal(call.args[1], signals[example:example + 1])
                    self.assertEqual(list(call.args[2]), [ids[example]])

                shown = shown_predictions(app)
                self.assertEqual(sorted(shown), sorted(LIVE_MODELS.values()))
                for name, label in LIVE_MODELS.items():
                    expected = direct[name].predict_ppg(signals[example:example + 1], [ids[example]]).iloc[0]
                    tiles = shown[label]
                    self.assertTrue(all(value.endswith(" mmHg") for value in tiles.values()))
                    sbp = float(tiles["Estimated systolic blood pressure"].removesuffix(" mmHg"))
                    dbp = float(tiles["Estimated diastolic blood pressure"].removesuffix(" mmHg"))
                    self.assertTrue(np.isfinite([sbp, dbp]).all())
                    self.assertAlmostEqual(sbp, expected["predicted_sbp"], delta=0.051)
                    self.assertAlmostEqual(dbp, expected["predicted_dbp"], delta=0.051)
                # The reference is displayed once, below both models.
                self.assertEqual(sum("Reference for this example" in element.value for element in app.markdown), 1)
                self.assertNotEqual(previous, shown)
                previous = shown

    def test_missing_cnn_checkpoint_keeps_resnet_working(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        import models.selected_models as selected_models

        real_loader = selected_models.load_selected_model

        def loader(model, *args, **kwargs):
            if model == "cnn_c":
                raise ValueError("Missing or changed model artifact: components/cnn_c/seed_2026/model.pt")
            return real_loader(model, *args, **kwargs)

        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with patch.object(selected_models, "load_selected_model", side_effect=loader):
            app = AppTest.from_file(str(ROOT / "demo/streamlit_app.py"), default_timeout=90).run()
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        shown = shown_predictions(app)
        self.assertEqual(len(shown["ResNet (L1)"]), 2)
        self.assertEqual(shown["Short CNN"], {})
        self.assertEqual(len(app.metric), 2)
        self.assertTrue(any("Short CNN is unavailable" in warning.value for warning in app.warning))

    def test_model_comparison_matches_saved_validation_results(self):
        import csv
        import json
        from statistics import mean
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "demo/streamlit_app.py"), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        table = app.dataframe[0].value.set_index("Model")
        names = {"training_median": "Training-median baseline", "lightgbm_16": "LightGBM (16 PPG features)",
                 "cnn_c": "Short CNN", "resnet_l1": "ResNet (L1, standalone)",
                 "recommended": "Recommended ensemble"}
        self.assertEqual(list(table.index), list(names.values()))
        rows = []
        for name in ("selected_metrics.csv", "cnn_metrics.csv"):
            with open(ROOT / "reports/final" / name, newline="") as handle:
                rows += [row for row in csv.DictReader(handle) if row["split"] == "validation"]
        for model, label in names.items():
            for column, shown in (("sbp_mae", "SBP MAE"), ("dbp_mae", "DBP MAE"),
                                  ("sbp_rmse", "SBP RMSE"), ("dbp_rmse", "DBP RMSE")):
                expected = mean(float(row[column]) for row in rows if row["model"] == model)
                self.assertAlmostEqual(table.loc[label, shown], expected, delta=0.005)
        # The saved comparison chart is displayed.
        self.assertEqual(len(app.get("image")), 1)

        weights = json.loads((ROOT / "configs/selected_models.json").read_text())["models"]["recommended"]
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        text = " ".join(element.value for element in app.markdown)
        self.assertIn(f"{weights['cnn_c']:.0%} Short CNN", text)

    def test_bundle_replays_release_prediction_and_rejects_flat_input(self):
        import numpy as np
        from models.selected_models import load_selected_model

        signals, ids = bundled_examples()
        for name in LIVE_MODELS:
            with self.subTest(model=name):
                predictor = load_selected_model(name, seed=2026, artifact_dir=BUNDLE)
                result = predictor.predict_ppg(signals[:1], ids[:1])
                self.assertEqual(result.rejection_reason.iloc[0], "")
                self.assertTrue(np.isfinite(result[["predicted_sbp", "predicted_dbp"]].to_numpy()).all())
                rejected = predictor.predict_ppg(np.zeros((1, 625)), ["flat"])
                self.assertTrue(rejected.rejection_reason.iloc[0])
                self.assertTrue(rejected[["predicted_sbp", "predicted_dbp"]].isna().all().all())
        # Independent recorded release output for ResNet; tolerate CPU rounding differences.
        # The release records no standalone cnn_c output to compare against.
        resnet = load_selected_model("resnet_l1", seed=2026, artifact_dir=BUNDLE)
        np.testing.assert_allclose(
            resnet.predict_ppg(signals[:1], ids[:1])[["predicted_sbp", "predicted_dbp"]].to_numpy(),
            [[124.208170, 69.711548]], atol=0.002, rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
