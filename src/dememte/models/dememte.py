"""DeMemteAttractor — full model orchestrating ResNet backbone + VQ memory + gate."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attractor import AttractorMemory, AmbiguityGate
from .baseline import make_backbone
from .vq import LatentProjector, LatentUnprojector, VectorQuantizer2D, sigreg_latent_loss


class DeMemteAttractor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 102,
        latent_dim: int = 128,
        num_embeddings: int = 1024,
        attractor_hidden: int = 512,
        gate_hidden: int = 16,
        commitment_cost: float = 0.25,
        vq_temperature: float = 1.0,
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
        disable_attractor: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = LatentProjector(512, latent_dim)
        self.vq = VectorQuantizer2D(num_embeddings, latent_dim, commitment_cost, vq_temperature)
        self.attractor = AttractorMemory(latent_dim, attractor_hidden)
        self.gate = AmbiguityGate(
            num_classes=num_classes,
            num_embeddings=num_embeddings,
            hidden=gate_hidden,
            familiarity_midpoint=familiarity_midpoint,
            familiarity_width=familiarity_width,
            ood_tau=ood_tau,
            ood_beta=ood_beta,
            gate_init_prob=gate_init_prob,
            gate_prior_floor=gate_prior_floor,
            gate_dropout=gate_dropout,
            use_uncertainty=use_uncertainty,
            use_familiarity=use_familiarity,
            use_conflict=use_conflict,
            use_ood=use_ood,
        )
        self.unprojector = LatentUnprojector(latent_dim, 512)
        self.aux_classifier = nn.Linear(512, num_classes)
        self.classifier = nn.Linear(512, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.disable_attractor = bool(disable_attractor)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def _mask_features(self, feats: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        if mask_ratio <= 0:
            return feats
        b, _, h, w = feats.shape
        keep = (torch.rand(b, 1, h, w, device=feats.device) > mask_ratio).float()
        return feats * keep

    def encode_z(self, x: torch.Tensor):
        feats = self.backbone(x)
        z = self.projector(feats)
        return feats, z

    def pretrain_latent(self, x: torch.Tensor, feature_mask_ratio: float = 0.0, update_ema: bool = True, sigreg_sketch_dim: int = 64):
        feats = self.backbone(x)
        vq_input = self._mask_features(feats, feature_mask_ratio if self.training else 0.0)
        z = self.projector(vq_input)
        zq, vq_loss, dq_map, soft_assign = self.vq(z)
        rec_feats = self.unprojector(zq)
        if self.training and update_ema:
            aux_logits = self.aux_classifier(self.pool(feats).flatten(1))
            self.gate(aux_logits, dq_map, soft_assign, update_ema=True)
        recon_loss = F.mse_loss(rec_feats, feats.detach())
        sigreg_loss = sigreg_latent_loss(z, sigreg_sketch_dim)
        return recon_loss, vq_loss, sigreg_loss

    def forward(self, x: torch.Tensor, target_z=None, update_ema: bool = True, return_debug: bool = False, feature_mask_ratio: float = 0.0):
        feats = self.backbone(x)
        vq_feats = self._mask_features(feats, feature_mask_ratio if self.training else 0.0)
        z = self.projector(vq_feats)
        zq, vq_loss, dq_map, soft_assign = self.vq(z)
        z_completed = z if self.disable_attractor else self.attractor(z)
        delta = z_completed - z.detach()

        feats_flat = self.pool(feats).flatten(1)
        aux_logits = self.aux_classifier(feats_flat)
        gate, signals = self.gate(aux_logits, dq_map, soft_assign, update_ema=update_ema)

        z_final = z + gate * delta
        feats_final = self.unprojector(z_final)
        logits = self.classifier(self.pool(feats_final).flatten(1))
        target = z.detach() if target_z is None else target_z.detach()
        denoise_loss = F.mse_loss(z_completed, target)

        if return_debug:
            debug = {
                "gate": gate,
                "delta": delta,
                "z": z,
                "z_completed": z_completed,
                "dq_map": dq_map,
                "logits_base": aux_logits,
                "clean_feats": feats,
                "enhanced_feats": feats_final,
                **signals,
            }
            return logits, denoise_loss, vq_loss, debug
        return logits, denoise_loss, vq_loss


def make_dememte_variant(config, device: str = "cuda") -> DeMemteAttractor:
    """Build a DeMemteAttractor from an E5Config / AblationConfig."""
    model = DeMemteAttractor(
        backbone=make_backbone(),
        num_classes=config.num_classes,
        latent_dim=config.latent_dim,
        num_embeddings=config.num_embeddings,
        attractor_hidden=config.attractor_hidden,
        gate_hidden=config.gate_hidden,
        commitment_cost=config.commitment_cost,
        vq_temperature=config.vq_temperature,
        familiarity_midpoint=config.familiarity_midpoint,
        familiarity_width=config.familiarity_width,
        ood_tau=config.ood_tau,
        ood_beta=config.ood_beta,
        gate_init_prob=config.gate_init_prob,
        gate_prior_floor=config.gate_prior_floor,
        gate_dropout=config.gate_dropout,
        use_uncertainty=config.use_uncertainty,
        use_familiarity=config.use_familiarity,
        use_conflict=config.use_conflict,
        use_ood=config.use_ood,
        disable_attractor=config.disable_attractor,
    ).to(device)
    model.set_backbone_trainable(False)
    return model


def make_dememte_e5(config=None, device: str = "cuda") -> DeMemteAttractor:
    """Convenience: build the E5 winner with default config if none provided."""
    from ..config import E5Config
    return make_dememte_variant(config or E5Config(), device=device)


def reset_gate_calibration_from_config(model: DeMemteAttractor, config) -> None:
    with torch.no_grad():
        model.gate.midpoint.fill_(float(config.familiarity_midpoint))
        model.gate.log_width.fill_(math.log(float(config.familiarity_width)))
    model.gate.ood_tau = float(config.ood_tau)
    model.gate.ood_beta = float(config.ood_beta)
    model.gate.gate_prior_floor = float(config.gate_prior_floor)
