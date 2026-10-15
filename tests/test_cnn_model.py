import unittest

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


if __name__ == "__main__":
    unittest.main()
