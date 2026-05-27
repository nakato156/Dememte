"""Test-time adaptation helpers for DeMemte VQSA.

Implements the E7 TTA variants:
- TENT-style entropy minimization over BatchNorm affine parameters.
- EATA-lite reliable/non-redundant entropy minimization without Fisher.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTAStats:
    """Counters accumulated during online test-time adaptation."""

    updates: int = 0
    reliable: int = 0
    selected: int = 0
    seen: int = 0

    @property
    def selection_rate(self) -> float:
        return float(self.selected / max(1, self.seen))


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of a softmax distribution from logits, per sample."""
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def collect_tta_bn_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """Collect BatchNorm affine parameters used by TENT/EATA."""
    params: List[nn.Parameter] = []
    names: List[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in {"weight", "bias"}:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    return params, names


def configure_tta_model(model: nn.Module) -> nn.Module:
    """Freeze the model except BN affine params and force BN batch statistics.

    The full model remains in eval mode so dropout is disabled and EMA VQ
    codebooks are not updated. BatchNorm layers use per-batch statistics by
    clearing running statistics and disabling tracking.
    """
    model.eval()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def make_tta_optimizer(params, lr: float = 2.5e-4, momentum: float = 0.9):
    return torch.optim.SGD(params, lr=lr, momentum=momentum)


class _BaseAdapter(nn.Module):
    def __init__(self, model: nn.Module, optimizer, steps: int = 1, episodic: bool = False):
        super().__init__()
        if steps < 1:
            raise ValueError("TTA adapters require steps >= 1")
        self.model = model
        self.optimizer = optimizer
        self.steps = int(steps)
        self.episodic = bool(episodic)
        self.stats = TTAStats()
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

    def reset(self) -> None:
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.stats = TTAStats()


class TentAdapter(_BaseAdapter):
    """TENT-style online entropy minimization."""

    method_name = "tent_bn"

    @torch.enable_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        if self.episodic:
            self.reset()
        logits, dbg = None, None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits, _, dbg = self.model(x, return_debug=True)
            loss = softmax_entropy(logits).mean()
            loss.backward()
            self.optimizer.step()
            self.stats.updates += 1
            self.stats.reliable += x.size(0)
            self.stats.selected += x.size(0)
            self.stats.seen += x.size(0)
        if return_debug:
            return logits, dbg
        return logits


class EATALiteAdapter(_BaseAdapter):
    """EATA sample filtering without Fisher regularization."""

    method_name = "eata_lite"

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        num_classes: int,
        steps: int = 1,
        episodic: bool = False,
        e_margin: float | None = None,
        d_margin: float = 0.05,
        prob_momentum: float = 0.9,
    ):
        super().__init__(model, optimizer, steps=steps, episodic=episodic)
        self.e_margin = float(0.4 * math.log(num_classes) if e_margin is None else e_margin)
        self.d_margin = float(d_margin)
        self.prob_momentum = float(prob_momentum)
        self.current_model_probs: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self.current_model_probs = None

    @torch.no_grad()
    def _update_model_probs(self, new_probs: torch.Tensor) -> None:
        if new_probs.numel() == 0:
            return
        mean_probs = new_probs.mean(dim=0).detach()
        if self.current_model_probs is None:
            self.current_model_probs = mean_probs
        else:
            self.current_model_probs = self.prob_momentum * self.current_model_probs + (1.0 - self.prob_momentum) * mean_probs

    @torch.enable_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        if self.episodic:
            self.reset()
        logits, dbg = None, None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits, _, dbg = self.model(x, return_debug=True)
            entropies = softmax_entropy(logits)
            reliable_mask = entropies < self.e_margin
            reliable_count = int(reliable_mask.sum().item())
            selected_mask = reliable_mask.clone()

            if self.current_model_probs is not None and reliable_count > 0:
                probs = logits.softmax(dim=1)
                similarities = F.cosine_similarity(
                    self.current_model_probs.unsqueeze(0),
                    probs[reliable_mask],
                    dim=1,
                )
                reliable_indices = reliable_mask.nonzero(as_tuple=False).flatten()
                selected_mask[:] = False
                selected_mask[reliable_indices[torch.abs(similarities) < self.d_margin]] = True

            selected_count = int(selected_mask.sum().item())
            self.stats.reliable += reliable_count
            self.stats.selected += selected_count
            self.stats.seen += x.size(0)

            if selected_count > 0:
                selected_entropies = entropies[selected_mask]
                coeff = torch.exp(-(selected_entropies.detach() - self.e_margin))
                loss = (selected_entropies * coeff).mean()
                loss.backward()
                self.optimizer.step()
                self.stats.updates += 1
                self._update_model_probs(logits.softmax(dim=1)[selected_mask])
            elif reliable_count > 0 and self.current_model_probs is None:
                self._update_model_probs(logits.softmax(dim=1)[reliable_mask])

        if return_debug:
            return logits, dbg
        return logits
