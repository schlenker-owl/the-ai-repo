import torch
from airoad.vision.transfer import ResNet18Transfer

def test_resnet_transfer_shape():
    m = ResNet18Transfer(num_classes=10, weights=None, freeze_backbone=True)
    x = torch.randn(4, 3, 64, 64)
    y = m(x)
    assert y.shape == (4, 10)
