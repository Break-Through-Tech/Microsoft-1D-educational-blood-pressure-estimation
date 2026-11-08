"""Output shape, finite outputs and checkpoint compatibility for the waveform networks."""

import json
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]

# Layer shapes of PPGCNN on main (PR #39); saved short-CNN checkpoints depend on them.
MAIN_PPGCNN_SHAPES = {
    "features.0.weight": (16, 1, 7), "features.0.bias": (16,),
    "features.3.weight": (32, 16, 5), "features.3.bias": (32,),
    "features.6.weight": (64, 32, 5), "features.6.bias": (64,),
    "head.weight": (2, 64), "head.bias": (2,),
}


def selected_parameters(name):
    config = json.loads((ROOT / "configs" / "selected_models.json").read_text(encoding="utf-8"))
    return config["components"][name]["parameters"]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WaveModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x1 = torch.randn(4, 1, 625)
        self.x3 = torch.randn(4, 3, 625)

    def assert_regression_output(self, output, batch=4):
        self.assertEqual(tuple(output.shape), (batch, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_default_ppgcnn_keeps_main_layer_shapes(self):
        from models.cnn_model import PPGCNN

        shapes = {k: tuple(v.shape) for k, v in PPGCNN().state_dict().items()}
        self.assertEqual(shapes, MAIN_PPGCNN_SHAPES)

    def test_main_checkpoint_loads_into_short_cnn_built_from_parameters(self):
        from models.cnn_model import PPGCNN, build_wave_model

        old_weights = PPGCNN().state_dict()
        model = build_wave_model(dict(representation="ppg", channels=[16, 32, 64]))
        model.load_state_dict(old_weights, strict=True)
        model.eval()
        with torch.no_grad():
            self.assert_regression_output(model(self.x1))

    def test_short_cnn_variants_give_finite_two_value_outputs(self):
        from models.cnn_model import PPGCNN

        variants = {
            "one_channel": (PPGCNN(), self.x1),
            "three_channel": (PPGCNN(input_channels=3), self.x3),
            "dilated": (PPGCNN(input_channels=3, dilations=(2, 2, 2)), self.x3),
        }
        for name, (model, x) in variants.items():
            with self.subTest(model=name), torch.no_grad():
                self.assert_regression_output(model.eval()(x))

    def test_selected_short_cnn_and_resnet_configs_build_and_predict(self):
        from models.cnn_model import PPGCNN, ResidualPPGCNN, build_wave_model

        for name, kind in (("cnn_c", PPGCNN), ("multitask_resnet", ResidualPPGCNN),
                           ("resnet_l1", ResidualPPGCNN)):
            model = build_wave_model(selected_parameters(name)).eval()
            with self.subTest(model=name), torch.no_grad():
                self.assertIsInstance(model, kind)
                self.assert_regression_output(model(self.x3))

    def test_multitask_resnet_heads_have_expected_shapes(self):
        from models.cnn_model import build_wave_model

        model = build_wave_model(selected_parameters("multitask_resnet")).eval()
        with torch.no_grad():
            regression, sbp_classes, dbp_classes = model.forward_tasks(self.x3)
            plain = model(self.x3)
        self.assert_regression_output(regression)
        self.assertEqual(tuple(sbp_classes.shape), (4, 5))
        self.assertEqual(tuple(dbp_classes.shape), (4, 5))
        self.assertTrue(torch.isfinite(sbp_classes).all() and torch.isfinite(dbp_classes).all())
        # Inference uses forward(); it must equal the regression head trained in forward_tasks().
        torch.testing.assert_close(plain, regression)

    def test_unknown_architecture_is_rejected(self):
        from models.cnn_model import build_wave_model

        with self.assertRaisesRegex(ValueError, "Unknown waveform architecture"):
            build_wave_model(dict(architecture="vgg", representation="ppg", channels=[16, 32, 64]))


if __name__ == "__main__":
    unittest.main()
