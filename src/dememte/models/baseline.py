"""ImageNet-pretrained ResNet backbones and simple classifier baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


_BACKBONE_SPECS = {
    "resnet18": {
        "builder": torchvision.models.resnet18,
        "weights": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        "out_channels": 512,
    },
    "resnet50": {
        "builder": torchvision.models.resnet50,
        "weights": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        "out_channels": 2048,
    },
}


def backbone_out_channels(name: str = "resnet50") -> int:
    if name not in _BACKBONE_SPECS:
        raise ValueError(f"Unknown backbone {name!r}. Choices: {sorted(_BACKBONE_SPECS)}")
    return int(_BACKBONE_SPECS[name]["out_channels"])


def make_backbone(name: str = "resnet50", pretrained: bool = True) -> nn.Sequential:
    """Return an ImageNet-pretrained ResNet feature extractor without pool/FC."""
    if name not in _BACKBONE_SPECS:
        raise ValueError(f"Unknown backbone {name!r}. Choices: {sorted(_BACKBONE_SPECS)}")
    spec = _BACKBONE_SPECS[name]
    weights = spec["weights"] if pretrained else None
    base = spec["builder"](weights=weights)
    return nn.Sequential(*list(base.children())[:-2])


def make_imagenet_resnet50(device: str = "cuda") -> nn.Module:
    """Full ImageNet-1K ResNet-50 classifier for ImageNet/ImageNet-C source eval."""
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    return model.to(device)


class ResNetBaseline(nn.Module):
    """ResNet backbone + GAP + linear classifier for fine-tuning experiments."""

    def __init__(
        self,
        num_classes: int = 1000,
        freeze_backbone: bool = True,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = make_backbone(backbone_name, pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(backbone_out_channels(backbone_name), num_classes)
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
