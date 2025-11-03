from __future__ import annotations
import torch.nn as nn
from torchvision import models

class ResNet18Transfer(nn.Module):
    """
    Minimal transfer learner: resnet18 backbone (weights=None by default),
    replace fc to num_classes.
    """
    def __init__(self, num_classes: int = 10, weights=None, freeze_backbone: bool = True):
        super().__init__()
        m = models.resnet18(weights=weights)
        if freeze_backbone:
            for p in m.parameters():
                p.requires_grad = False
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        self.model = m

    def forward(self, x):
        return self.model(x)
