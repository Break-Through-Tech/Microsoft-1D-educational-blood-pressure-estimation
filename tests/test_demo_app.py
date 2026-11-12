"""Exercise the live UI against the small, tracked deployment bundle."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Install demo/requirements.txt for UI tests")
class DemoTests(unittest.TestCase):
    def test_live_prediction_and_example_switch(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "demo/streamlit_app.py"), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 2)
        first = [metric.value for metric in app.metric]
        self.assertTrue(all("mmHg" in value for value in first))
        app.selectbox[0].select(1).run()
        # A previous example's prediction must never appear under a new waveform.
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.error), 0)
        self.assertNotEqual(first, [metric.value for metric in app.metric])

    def test_bundle_replays_release_prediction_and_rejects_flat_input(self):
        import numpy as np
        from models.selected_models import load_selected_model

        bundle = ROOT / "demo/assets"
        predictor = load_selected_model("resnet_l1", seed=2026, artifact_dir=bundle)
        with np.load(bundle / "samples/ppg.npz", allow_pickle=False) as examples:
            result = predictor.predict_ppg(examples["ppg"][:1], examples["window_ids"][:1])
        self.assertEqual(result.rejection_reason.iloc[0], "")
        # Independent recorded release output; tolerate CPU rounding differences.
        np.testing.assert_allclose(
            result[["predicted_sbp", "predicted_dbp"]].to_numpy(),
            [[124.208170, 69.711548]], atol=0.002, rtol=0,
        )
        rejected = predictor.predict_ppg(np.zeros((1, 625)), ["flat"])
        self.assertTrue(rejected.rejection_reason.iloc[0])
        self.assertTrue(rejected[["predicted_sbp", "predicted_dbp"]].isna().all().all())


if __name__ == "__main__":
    unittest.main()
