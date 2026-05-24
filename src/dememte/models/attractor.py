"""Attractor memory (pattern completion residual MLP) and adaptive gate."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttractorMemory(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape
        toks = z.flatten(2).transpose(1, 2).contiguous().view(b * h * w, c)
        completed = toks + self.net(toks)
        return completed.view(b, h * w, c).transpose(1, 2).view(b, c, h, w).contiguous()


class AmbiguityGate(nn.Module):
    """Adaptive multiplier in [0,1] gating how much memory correction is applied."""

    def __init__(
        self,
        num_classes: int = 102,
        num_embeddings: int = 1024,
        hidden: int = 16,
        familiarity_midpoint: float = 0.0,
        familiarity_width: float = 1.0,
        ood_tau: float = 2.0,
        ood_beta: float = 4.0,
        gate_init_prob: float = 0.1,
        gate_prior_floor: float = 0.02,
        gate_dropout: float = 0.0,
        use_uncertainty: bool = True,
        use_familiarity: bool = True,
        use_conflict: bool = True,
        use_ood: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.ood_tau = ood_tau
        self.ood_beta = ood_beta
        self.gate_prior_floor = gate_prior_floor
        self.gate_dropout = gate_dropout
        self.use_uncertainty = use_uncertainty
        self.use_familiarity = use_familiarity
        self.use_conflict = use_conflict
        self.use_ood = use_ood

        self.midpoint = nn.Parameter(torch.tensor(float(familiarity_midpoint)))
        self.log_width = nn.Parameter(torch.log(torch.tensor(float(familiarity_width))))
        self.gate_dropout_layer = nn.Dropout(float(gate_dropout)) if gate_dropout > 0 else nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        init_prob = min(max(float(gate_init_prob), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.mlp[-1].bias, math.log(init_prob / (1.0 - init_prob)))

        self.register_buffer("dq_ema_mean", torch.tensor(0.0))
        self.register_buffer("dq_ema_var", torch.tensor(1.0))
        self.register_buffer("dq_ema_counted", torch.tensor(0.0))

    def _update_ema(self, dq_score: torch.Tensor) -> None:
        with torch.no_grad():
            batch_mean = dq_score.mean()
            batch_var = dq_score.var(unbiased=False) if dq_score.numel() > 1 else torch.tensor(0.0, device=dq_score.device)
            momentum = 0.99
            if self.dq_ema_counted.item() == 0:
                self.dq_ema_mean.fill_(batch_mean)
                self.dq_ema_var.fill_(batch_var)
            else:
                self.dq_ema_mean.mul_(momentum).add_(batch_mean * (1 - momentum))
                self.dq_ema_var.mul_(momentum).add_(batch_var * (1 - momentum))
            self.dq_ema_counted.add_(1.0)

    def forward(self, aux_logits: torch.Tensor, dq_map: torch.Tensor, soft_assign: torch.Tensor, update_ema: bool = True):
        b = aux_logits.size(0)
        dq_score = dq_map.flatten(1).mean(dim=1, keepdim=True)
        if self.training and update_ema:
            self._update_ema(dq_score)

        dq_std = torch.sqrt(self.dq_ema_var.clamp_min(0.0)) + 1e-5
        dq_norm = (dq_score - self.dq_ema_mean) / dq_std

        probs = F.softmax(aux_logits, dim=1)
        uncertainty = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True) / math.log(self.num_classes)

        width = torch.exp(self.log_width).clamp_min(1e-4)
        familiarity = torch.exp(-((dq_norm - self.midpoint) ** 2) / (2 * width ** 2))

        conflict = -(soft_assign * torch.log(soft_assign + 1e-8)).sum(dim=-1).mean(dim=(1, 2), keepdim=False).view(b, 1) / math.log(self.num_embeddings)
        ood_risk = torch.sigmoid(self.ood_beta * (dq_norm - self.ood_tau))

        zero = torch.zeros_like(uncertainty)
        gate_inputs = torch.cat([
            uncertainty if self.use_uncertainty else zero,
            familiarity if self.use_familiarity else zero,
            conflict if self.use_conflict else zero,
            (1.0 - ood_risk) if self.use_ood else zero,
        ], dim=1)
        one = torch.ones_like(uncertainty)
        uncertainty_factor = uncertainty if self.use_uncertainty else one
        familiarity_factor = familiarity if self.use_familiarity else one
        ood_factor = (1.0 - ood_risk) if self.use_ood else one
        gate_prior = (uncertainty_factor * familiarity_factor * ood_factor).clamp(0.0, 1.0)
        gate_hidden = self.mlp[1](self.mlp[0](gate_inputs))
        gate_hidden = self.gate_dropout_layer(gate_hidden)
        gate_raw = torch.sigmoid(self.mlp[2](gate_hidden))
        floor = min(max(float(self.gate_prior_floor), 0.0), 1.0)
        gate_score = floor + (1.0 - floor) * gate_prior * gate_raw
        gate = gate_score.view(b, 1, 1, 1)
        signals = {
            "dq_norm": dq_norm.view(b, 1, 1, 1),
            "uncertainty": uncertainty.view(b, 1, 1, 1),
            "familiarity": familiarity.view(b, 1, 1, 1),
            "conflict": conflict.view(b, 1, 1, 1),
            "ood_risk": ood_risk.view(b, 1, 1, 1),
            "gate_prior": gate_prior.view(b, 1, 1, 1),
            "gate_raw": gate_raw.view(b, 1, 1, 1),
            "gate_inputs": gate_inputs,
        }
        return gate, signals


def gate_entropy_regularizer(gate: torch.Tensor) -> torch.Tensor:
    g = gate.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(g * torch.log(g) + (1.0 - g) * torch.log(1.0 - g))
    return -entropy.mean()
