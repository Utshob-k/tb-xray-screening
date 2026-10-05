"""Backbone: ImageNet-pretrained ResNet-18 with a single-logit head."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, 1)  # one logit: P(TB) = sigmoid(logit)
    return net


@torch.no_grad()
def predict_proba(model: nn.Module, loader, device: str) -> tuple[list[float], list[int]]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device)).squeeze(1)
        probs += torch.sigmoid(logits).cpu().tolist()
        labels += list(y)
    return probs, [int(v) for v in labels]
