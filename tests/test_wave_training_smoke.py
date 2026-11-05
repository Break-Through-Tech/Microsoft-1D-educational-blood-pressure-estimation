"""Tiny synthetic CNN training runs and training-only target statistics; no dataset files needed."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


class FakeCache:
    """Stands in for WaveCache: returns PPG channels for the requested window IDs."""

    def __init__(self, ppg_by_id):
        self.ppg_by_id = ppg_by_id

    def values(self, frame, representation="ppg"):
        from models.training_support import waveform_channels

        ppg = np.stack([self.ppg_by_id[w] for w in frame.window_id])
        return waveform_channels(ppg, representation)


class NoBudget:
    def check(self, selection=True):
        pass


def synthetic_split(split, recordings, windows_per_recording, sbp, dbp, rng):
    rows, ppg = [], {}
    t = np.arange(625) / 125
    for r in range(recordings):
        for w in range(windows_per_recording):
            window_id = f"{split}{r}:window_{w}"
            rate = rng.uniform(1, 2)
            ppg[window_id] = (np.sin(2 * np.pi * rate * t) + rng.normal(0, 0.05, 625)).astype(np.float32)
            rows.append(dict(window_id=window_id, source_record_id=f"{split}{r}", split=split,
                             sbp=sbp + rng.normal(0, 5), dbp=dbp + rng.normal(0, 3)))
    return pd.DataFrame(rows), ppg


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WaveTrainingSmokeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.train, train_ppg = synthetic_split("train", 4, 4, 120, 70, rng)
        # Holdout pressures are deliberately extreme so any leak into training statistics shows.
        self.holdout, holdout_ppg = synthetic_split("validation", 2, 4, 250, 150, rng)
        self.cache = FakeCache({**train_ppg, **holdout_ppg})
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def fit(self, name, parameters):
        from models.train_waveform_comparison import fit_wave

        return fit_wave(self.train, self.holdout, self.cache, parameters, seed=2026, device="cpu",
                        directory=self.root / name, budget=NoBudget(), limit=2)

    def check_completed_run(self, name, info):
        from models.train_waveform_comparison import load_wave_model

        directory = self.root / name
        for split in ("train", "inner"):
            for value in info["metrics"][split].values():
                self.assertTrue(np.isfinite(value))
        self.assertFalse(json.loads((directory / "complete.json").read_text())["test_accessed"])
        self.assertEqual(len(pd.read_csv(directory / "history.csv")), 2)

        _, state, _ = load_wave_model(directory)
        np.testing.assert_allclose(state["target_mean"], self.train[["sbp", "dbp"]].mean().to_numpy())
        np.testing.assert_allclose(state["target_scale"], self.train[["sbp", "dbp"]].std(ddof=0).to_numpy())

    def test_short_cnn_completes_small_training_run(self):
        parameters = dict(representation="ppg", normalization="global", channels=[4, 4, 4],
                          learning_rate=0.001, weight_decay=0.0001, batch_size=8)
        info = self.fit("short_cnn", parameters)
        self.check_completed_run("short_cnn", info)

    def test_multitask_resnet_completes_small_training_run(self):
        parameters = dict(architecture="residual_v1", representation="derivatives",
                          normalization="global", channels=[4, 4, 4], learning_rate=0.001,
                          weight_decay=0.0001, batch_size=8, kernel_size=7, loss="l1",
                          auxiliary_classification=True, auxiliary_lambda=0.2)
        info = self.fit("multitask_resnet", parameters)
        self.check_completed_run("multitask_resnet", info)

    def test_saved_short_cnn_predicts_finite_values_for_new_ppg(self):
        from models.train_waveform_comparison import predict_wave

        parameters = dict(representation="ppg", normalization="global", channels=[4, 4, 4],
                          learning_rate=0.001, weight_decay=0.0001, batch_size=8)
        self.fit("short_cnn", parameters)
        predictions = predict_wave(self.root / "short_cnn", self.holdout, self.cache)
        self.assertEqual(predictions.shape, (len(self.holdout), 2))
        self.assertTrue(np.isfinite(predictions).all())


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FeatureBaselineTests(unittest.TestCase):
    def test_training_mean_baseline_ignores_validation_labels(self):
        from models.train_feature_mlp import fit_preprocessing

        train = pd.DataFrame(dict(f1=[1.0, 2.0, 3.0, 4.0], f2=[0.5, np.nan, 1.5, 2.0],
                                  sbp=[110.0, 120.0, 130.0, 140.0], dbp=[60.0, 70.0, 80.0, 90.0]))
        state = fit_preprocessing(train, ["f1", "f2"])
        self.assertEqual(state["target_mean"], [125.0, 75.0])
        # save_run() reports this exact vector as the training_mean_diagnostic baseline.
        mixed = pd.concat([train, pd.DataFrame(dict(f1=[9.0], f2=[9.0], sbp=[999.0], dbp=[999.0]))])
        self.assertNotEqual(fit_preprocessing(mixed, ["f1", "f2"])["target_mean"], state["target_mean"])


if __name__ == "__main__":
    unittest.main()
