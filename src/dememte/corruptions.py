"""Image-level corruption suite (train augmentation + evaluation grid).

Mirrors the corruption helpers from `Dememte_e5y.ipynb` so that train-time and
eval-time corruptions match the legacy pipeline exactly.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F


# Eval grid: 4 corruption types x 3 severities each.
STRICT_SUITE = {
    "gaussian_noise": [0.5, 1.0, 1.5],
    "pixel_mask": [0.25, 0.5, 0.75],
    "cutout": [0.2, 0.35, 0.5],
    "blur": [0.35, 0.6, 0.85],
}


def _gaussian_blur_batch(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    channels = x.size(1)
    weight = torch.ones(channels, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype) / (kernel_size * kernel_size)
    return F.conv2d(x, weight, padding=kernel_size // 2, groups=channels)


def apply_train_corruption(x: torch.Tensor, prob: float = 0.7) -> torch.Tensor:
    """Stochastic augmentation: with `prob`, pick one mode and apply it batch-wide."""
    if random.random() > prob:
        return x

    mode = random.choice(["gaussian_noise", "pixel_mask", "cutout", "blur"])

    if mode == "gaussian_noise":
        level = random.uniform(0.4, 1.3)
        return x + level * torch.randn_like(x)

    if mode == "pixel_mask":
        level = random.uniform(0.20, 0.65)
        keep = (torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device) > level).float()
        return x * keep

    if mode == "cutout":
        level = random.uniform(0.20, 0.45)
        b, _, h, w = x.shape
        cut_h = max(1, int(h * level))
        cut_w = max(1, int(w * level))
        mask = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
        for i in range(b):
            t = random.randint(0, max(0, h - cut_h))
            l = random.randint(0, max(0, w - cut_w))
            mask[i, :, t : t + cut_h, l : l + cut_w] = 0.0
        return x * mask

    # blur
    level = random.uniform(0.30, 0.80)
    blur = _gaussian_blur_batch(x, kernel_size=7)
    return (1.0 - level) * x + level * blur


def apply_eval_corruption(
    x: torch.Tensor,
    corruption: Optional[str],
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Deterministic eval-time corruption driven by an explicit RNG."""
    if corruption is None or severity == 0:
        return x

    if corruption == "gaussian_noise":
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return x + severity * noise

    if corruption == "pixel_mask":
        rand = torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device, generator=generator)
        keep = (rand > severity).float()
        return x * keep

    if corruption == "cutout":
        b, _, h, w = x.shape
        cut_h = max(1, int(h * severity))
        cut_w = max(1, int(w * severity))
        mask = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
        max_top = max(1, h - cut_h + 1)
        max_left = max(1, w - cut_w + 1)
        tops = torch.randint(0, max_top, (b,), device=x.device, generator=generator)
        lefts = torch.randint(0, max_left, (b,), device=x.device, generator=generator)
        for i in range(b):
            t, l = int(tops[i].item()), int(lefts[i].item())
            mask[i, :, t : t + cut_h, l : l + cut_w] = 0.0
        return x * mask

    if corruption == "blur":
        blur = _gaussian_blur_batch(x, kernel_size=7)
        return (1.0 - severity) * x + severity * blur

    raise ValueError(f"Unknown corruption: {corruption}")
