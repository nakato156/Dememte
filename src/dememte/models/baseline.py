"""ResNet18 baseline (supports frozen-backbone and full fine-tune modes)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def make_backbone() -> nn.Sequential:
    """ImageNet-pretrained ResNet18 stripped of its final pooling/FC."""
    base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    return nn.Sequential(*list(base.children())[:-2])


class ResNetBaseline(nn.Module):
    """ResNet18 + GAP + linear classifier. Used in notebook 01 (frozen) and 04 (FT)."""

    def __init__(self, num_classes: int = 102, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = make_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self.freeze_backbone = freeze_backbone
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pooled = self.pool(feats).flatten(1)
        return self.classifier(pooled)
