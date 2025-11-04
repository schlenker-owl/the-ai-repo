# tests/test_cnn_shapes.py
import torch

from airoad.vision.cnn_torch import SimpleCNN


def test_cnn_output_shape():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)
