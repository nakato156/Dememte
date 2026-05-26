"""DeMemte VQSA: ResNet backbone + vector quantization + self-attention fusion."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from .baseline import make_backbone
from .vq import LatentProjector, VectorQuantizer2D


class SelfAttentionBlock(nn.Module):
    """Small encoder block that exposes attention maps for diagnostics."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class VQSAFusion(nn.Module):
    """Quantize feature maps, fuse original/quantized descriptors, and refine by self-attention."""

    VALID_FUSIONS = {"concat", "replace", "add"}

    def __init__(
        self,
        in_channels: int = 512,
        latent_dim: int = 256,
        num_embeddings: int = 1024,
        commitment_cost: float = 0.25,
        vq_temperature: float = 1.0,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_mode: str = "concat",
        use_codebook: bool = True,
        use_self_attention: bool = True,
    ):
        super().__init__()
        if fusion_mode not in self.VALID_FUSIONS:
            raise ValueError(f"Unknown VQSA fusion mode: {fusion_mode!r}. Choices: {sorted(self.VALID_FUSIONS)}")
        self.latent_dim = latent_dim
        self.fusion_mode = fusion_mode
        self.use_codebook = bool(use_codebook)
        self.use_self_attention = bool(use_self_attention)
        self.projector = LatentProjector(in_channels, latent_dim)
        self.vq = VectorQuantizer2D(num_embeddings, latent_dim, commitment_cost, vq_temperature)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention = nn.ModuleList([
            SelfAttentionBlock(latent_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def _zero_loss(self, z: torch.Tensor) -> torch.Tensor:
        return z.sum() * 0.0

    def _tokens_from_pooled(self, z_pool: torch.Tensor, zq_pool: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "concat":
            return torch.stack([z_pool, zq_pool], dim=1)
        if self.fusion_mode == "replace":
            return torch.stack([zq_pool, zq_pool], dim=1)
        fused = z_pool + zq_pool
        return torch.stack([fused, fused], dim=1)

    def forward(self, feats: torch.Tensor):
        z = self.projector(feats)
        if self.use_codebook:
            zq, vq_loss, codebook_loss, commitment_loss, dq_map, soft_assign = self.vq(z)
        else:
            zq = z
            vq_loss = codebook_loss = commitment_loss = self._zero_loss(z)
            dq_map = torch.zeros(z.size(0), 1, z.size(2), z.size(3), device=z.device, dtype=z.dtype)
            soft_assign = None

        z_pool = self.pool(z).flatten(1)
        zq_pool = self.pool(zq).flatten(1)
        tokens = self._tokens_from_pooled(z_pool, zq_pool)
        attn_weights: List[torch.Tensor] = []
        if self.use_self_attention:
            for block in self.attention:
                tokens, weights = block(tokens)
                attn_weights.append(weights)

        fused = tokens.flatten(1)
        debug = {
            "z": z,
            "zq": zq,
            "z_pool": z_pool,
            "zq_pool": zq_pool,
            "tokens": tokens,
            "attention_weights": torch.stack(attn_weights, dim=1) if attn_weights else None,
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "dq_map": dq_map,
            "soft_assign": soft_assign,
        }
        return fused, debug


class DeMemteVQSA(nn.Module):
    """Strict VQSA classifier for quality-independent feature learning."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 102,
        latent_dim: int = 256,
        num_embeddings: int = 1024,
        commitment_cost: float = 0.25,
        vq_temperature: float = 1.0,
        vqsa_heads: int = 4,
        vqsa_layers: int = 2,
        vqsa_dropout: float = 0.1,
        vqsa_fusion_mode: str = "concat",
        vqsa_use_codebook: bool = True,
        vqsa_use_self_attention: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.vqsa = VQSAFusion(
            in_channels=512,
            latent_dim=latent_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            vq_temperature=vq_temperature,
            num_heads=vqsa_heads,
            num_layers=vqsa_layers,
            dropout=vqsa_dropout,
            fusion_mode=vqsa_fusion_mode,
            use_codebook=vqsa_use_codebook,
            use_self_attention=vqsa_use_self_attention,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * latent_dim, 512),
            nn.GELU(),
            nn.Dropout(vqsa_dropout),
            nn.Linear(512, num_classes),
        )

    @property
    def projector(self) -> LatentProjector:
        return self.vqsa.projector

    @property
    def vq(self) -> VectorQuantizer2D:
        return self.vqsa.vq

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def encode_z(self, x: torch.Tensor):
        feats = self.backbone(x)
        z = self.projector(feats)
        return feats, z

    def forward(self, x: torch.Tensor, return_debug: bool = False, **_):
        feats = self.backbone(x)
        fused, debug = self.vqsa(feats)
        logits = self.classifier(fused)
        if return_debug:
            debug = {**debug, "features": feats, "fused": fused}
            return logits, debug["vq_loss"], debug
        return logits


def make_dememte_variant(config, device: str = "cuda") -> DeMemteVQSA:
    """Build the strict VQSA DeMemte variant from an E5/Ablation config."""
    model = DeMemteVQSA(
        backbone=make_backbone(),
        num_classes=config.num_classes,
        latent_dim=config.latent_dim,
        num_embeddings=config.num_embeddings,
        commitment_cost=config.commitment_cost,
        vq_temperature=config.vq_temperature,
        vqsa_heads=config.vqsa_heads,
        vqsa_layers=config.vqsa_layers,
        vqsa_dropout=config.vqsa_dropout,
        vqsa_fusion_mode=config.vqsa_fusion_mode,
        vqsa_use_codebook=config.vqsa_use_codebook,
        vqsa_use_self_attention=config.vqsa_use_self_attention,
    ).to(device)
    model.set_backbone_trainable(bool(getattr(config, "vqsa_train_backbone", False)))
    return model


def make_dememte_e5(config=None, device: str = "cuda") -> DeMemteVQSA:
    """Convenience: build the strict VQSA default config if none is provided."""
    from ..config import E5Config
    return make_dememte_variant(config or E5Config(), device=device)


def attention_entropy(attention_weights: torch.Tensor | None) -> torch.Tensor:
    if attention_weights is None:
        return torch.tensor(0.0)
    probs = attention_weights.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1).mean() / math.log(probs.size(-1))
