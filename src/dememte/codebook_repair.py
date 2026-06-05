"""E10-A local codebook repair for test-time adaptation.

The adapter in this module never mutates the checkpoint. It keeps a local
``codebook_view`` buffer, repairs that view online, and forwards with the
repaired quantized token. This lets us test whether DeMemte's codebook can
become a useful memory under shift without returning to training-time E6.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import effective_codebook
from .tta import TTAStats


REPAIR_MODES = {"disabled", "ema", "reseed", "ema_reseed"}
BLEND_MODES = {"source", "replace", "gated"}


@dataclass
class CodebookRepairConfig:
    """Configuration for local test-time codebook repair."""

    repair_mode: str = "ema_reseed"
    blend_mode: str = "gated"
    ema_lr: float = 0.05
    usage_decay: float = 0.99
    dead_threshold: float = 1.0
    repair_every: int = 25
    max_reseeds: int = 16
    anchor_strength: float = 0.01
    max_code_drift: float = 0.5
    lambda_max: float = 0.5
    margin_min: float = 0.0
    dq_max: float = math.inf


def _squared_distances(flat: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    codebook = codebook.to(device=flat.device, dtype=flat.dtype)
    return flat.pow(2).sum(1, keepdim=True) - 2.0 * flat @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t()


def _validate_cfg(cfg: CodebookRepairConfig) -> None:
    if cfg.repair_mode not in REPAIR_MODES:
        raise ValueError(f"repair_mode must be one of {sorted(REPAIR_MODES)}, got {cfg.repair_mode!r}")
    if cfg.blend_mode not in BLEND_MODES:
        raise ValueError(f"blend_mode must be one of {sorted(BLEND_MODES)}, got {cfg.blend_mode!r}")
    if cfg.repair_every < 1:
        raise ValueError("repair_every must be >= 1")
    if cfg.max_reseeds < 0:
        raise ValueError("max_reseeds must be >= 0")


def quantize_with_codebook_view(z: torch.Tensor, codebook: torch.Tensor, temperature: float = 1.0):
    """Quantize ``z`` with an adapter-local codebook view.

    Returns ``(zq, dq_map, soft_assign, encoding_indices)``. The operation is
    no-grad friendly and mirrors the geometry of the project quantizers.
    """
    b, c, h, w = z.shape
    flat = z.permute(0, 2, 3, 1).contiguous().view(-1, c)
    distances = _squared_distances(flat, codebook)
    idx = torch.argmin(distances, dim=1)
    q_flat = codebook.to(device=z.device, dtype=z.dtype)[idx]
    zq = q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    dq_map = (z - zq).pow(2).mean(dim=1, keepdim=True)
    temp = max(1e-6, float(temperature))
    soft_assign = F.softmax(-distances / temp, dim=1).view(b, h, w, codebook.size(0))
    return zq, dq_map, soft_assign, idx.view(b, h, w)


def _assignment_margin_and_dq(z: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, c, h, w = z.shape
    flat = z.permute(0, 2, 3, 1).contiguous().view(-1, c)
    distances = _squared_distances(flat, codebook)
    top2 = distances.topk(k=min(2, distances.size(1)), largest=False, dim=1).values
    if top2.size(1) == 1:
        margin = torch.full_like(top2[:, 0], float("inf"))
    else:
        margin = (top2[:, 1] - top2[:, 0]).clamp_min(0.0)
    dq = top2[:, 0].clamp_min(0.0) / max(1, c)
    return margin.view(b, h, w), dq.view(b, h, w)


@torch.no_grad()
def calibrate_repair_thresholds(
    model: nn.Module,
    loader,
    device: str = "cuda",
    max_batches: int = 8,
    margin_quantile: float = 0.25,
    dq_quantile: float = 0.75,
) -> dict:
    """Calibrate reliability thresholds on clean/source batches, no labels used."""
    model.eval()
    codebook = effective_codebook(model.vqsa.vq)
    if codebook is None:
        raise ValueError("calibrate_repair_thresholds requires a lookup-based codebook.")
    margins, dqs = [], []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        _, _, dbg = model(x, return_debug=True)
        margin, dq = _assignment_margin_and_dq(dbg["z"], codebook)
        margins.append(margin.detach().flatten().cpu())
        dqs.append(dq.detach().flatten().cpu())
    if not margins:
        return {"margin_min": 0.0, "dq_max": math.inf}
    margin_flat = torch.cat(margins).float()
    dq_flat = torch.cat(dqs).float()
    return {
        "margin_min": float(torch.quantile(margin_flat, margin_quantile).item()),
        "dq_max": float(torch.quantile(dq_flat, dq_quantile).item()),
    }


def _hard_metrics(indices: torch.Tensor, num_codes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = indices.size(0)
    flat = indices.reshape(b, -1).long()
    usage_rows, dead_rows, perplexity_rows = [], [], []
    for row in flat:
        counts = torch.bincount(row, minlength=num_codes).float().to(indices.device)
        used = counts > 0
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs[used] * probs[used].clamp_min(1e-8).log()).sum()
        usage_rows.append(used.float().mean())
        dead_rows.append((~used).float().mean())
        perplexity_rows.append(entropy.exp())
    return torch.stack(usage_rows), torch.stack(dead_rows), torch.stack(perplexity_rows)


class LocalCodebookRepairAdapter(nn.Module):
    """Frozen DeMemte wrapper with an online-repaired local codebook view."""

    method_name = "local_codebook_repair"

    def __init__(self, model: nn.Module, cfg: Optional[CodebookRepairConfig] = None):
        super().__init__()
        cfg = cfg or CodebookRepairConfig()
        _validate_cfg(cfg)
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.cfg = cfg
        self.stats = TTAStats()
        self._batch_idx = 0

        vqsa = getattr(model, "vqsa", None)
        if vqsa is None or not getattr(vqsa, "use_codebook", False):
            raise ValueError("LocalCodebookRepairAdapter requires a DeMemte VQSA model with a codebook.")
        cb = effective_codebook(vqsa.vq)
        if cb is None:
            raise ValueError("LocalCodebookRepairAdapter requires a lookup-based codebook; FSQ has no codebook.")
        device = cb.device
        self.register_buffer("codebook_source", cb.detach().clone().to(device=device))
        self.register_buffer("codebook_view", cb.detach().clone().to(device=device))
        self.register_buffer("usage_ema", torch.zeros(cb.size(0), device=device))

    @torch.no_grad()
    def reset(self) -> None:
        self.codebook_view.copy_(self.codebook_source)
        self.usage_ema.zero_()
        self.stats = TTAStats()
        self._batch_idx = 0

    @torch.no_grad()
    def _apply_anchor_and_cap(self) -> None:
        cfg = self.cfg
        if cfg.anchor_strength > 0.0:
            self.codebook_view.lerp_(self.codebook_source, float(cfg.anchor_strength))
        if cfg.max_code_drift > 0.0 and math.isfinite(cfg.max_code_drift):
            delta = self.codebook_view - self.codebook_source
            norm = delta.norm(dim=1, keepdim=True)
            scale = (float(cfg.max_code_drift) / norm.clamp_min(1e-8)).clamp_max(1.0)
            self.codebook_view.copy_(self.codebook_source + delta * scale)

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor, reliable: torch.Tensor) -> int:
        cfg = self.cfg
        if cfg.ema_lr <= 0.0:
            return 0
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, z.size(1))
        idx = indices.reshape(-1).long()
        mask = reliable.reshape(-1).bool()
        if not mask.any():
            return 0
        flat = flat[mask].detach()
        idx = idx[mask]
        counts = torch.bincount(idx, minlength=self.codebook_view.size(0)).to(flat.dtype).to(flat.device)
        sums = torch.zeros_like(self.codebook_view)
        sums.index_add_(0, idx, flat)
        used = counts > 0
        means = sums[used] / counts[used].unsqueeze(1).clamp_min(1.0)
        self.codebook_view[used] = (1.0 - cfg.ema_lr) * self.codebook_view[used] + cfg.ema_lr * means
        return int(used.sum().item())

    @torch.no_grad()
    def _update_usage(self, indices: torch.Tensor) -> None:
        counts = torch.bincount(indices.reshape(-1).long(), minlength=self.codebook_view.size(0)).float().to(self.usage_ema.device)
        self.usage_ema.mul_(float(self.cfg.usage_decay)).add_(counts, alpha=1.0 - float(self.cfg.usage_decay))

    @torch.no_grad()
    def _reseed_dead(self, z: torch.Tensor, dq_map: torch.Tensor) -> int:
        cfg = self.cfg
        if cfg.max_reseeds <= 0:
            return 0
        if (self._batch_idx % cfg.repair_every) != 0:
            return 0
        dead = self.usage_ema < float(cfg.dead_threshold)
        dead_idx = dead.nonzero(as_tuple=False).flatten()
        if dead_idx.numel() == 0:
            return 0
        num = min(int(cfg.max_reseeds), int(dead_idx.numel()), int(z.numel() // z.size(1)))
        if num <= 0:
            return 0
        flat_z = z.permute(0, 2, 3, 1).contiguous().view(-1, z.size(1))
        flat_dq = dq_map.permute(0, 2, 3, 1).reshape(-1)
        top = flat_dq.topk(k=min(num, flat_dq.numel()), largest=True).indices
        target_idx = dead_idx[: top.numel()]
        self.codebook_view[target_idx] = flat_z[top].detach()
        self.usage_ema[target_idx] = 1.0
        return int(target_idx.numel())

    @torch.no_grad()
    def _repair(self, z: torch.Tensor, indices: torch.Tensor, reliable: torch.Tensor, dq_map: torch.Tensor) -> tuple[int, int]:
        cfg = self.cfg
        updated = 0
        reseeded = 0
        self._update_usage(indices)
        if cfg.repair_mode in {"ema", "ema_reseed"}:
            updated = self._ema_update(z, indices, reliable)
        if cfg.repair_mode in {"reseed", "ema_reseed"}:
            reseeded = self._reseed_dead(z, dq_map)
        if cfg.repair_mode != "disabled":
            self._apply_anchor_and_cap()
        return updated, reseeded

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        cfg = self.cfg
        if cfg.repair_mode == "disabled" or cfg.blend_mode == "source" or cfg.lambda_max <= 0.0:
            logits, _, dbg = self.model(x, return_debug=True)
            debug = self._attach_repair_debug(dbg, dbg["zq_pool"], dbg["zq_pool"], 0, 0, None, None)
            self.stats.seen += x.size(0)
            if return_debug:
                return logits, debug
            return logits

        vqsa = self.model.vqsa
        feats = self.model.backbone(x)
        z = vqsa.projector(feats)
        zq_source, _, _, _, dq_source, _, _ = vqsa.vq(z)
        z_pool = vqsa.pool(z).flatten(1)
        zq_pool_source = vqsa.pool(zq_source).flatten(1)

        zq_pre, dq_pre, _, idx_pre = quantize_with_codebook_view(z, self.codebook_view, getattr(vqsa.vq, "temperature", 1.0))
        margin, dq_token = _assignment_margin_and_dq(z, self.codebook_view)
        reliable = (margin >= float(cfg.margin_min)) & (dq_token <= float(cfg.dq_max))
        updated, reseeded = self._repair(z, idx_pre, reliable, dq_pre)

        zq_repaired, dq_repaired, soft_repaired, idx_repaired = quantize_with_codebook_view(
            z, self.codebook_view, getattr(vqsa.vq, "temperature", 1.0)
        )
        zq_pool_repaired = vqsa.pool(zq_repaired).flatten(1)
        reliability = reliable.float().flatten(1).mean(dim=1)
        if cfg.blend_mode == "replace":
            zq_pool_final = zq_pool_repaired
        else:
            lambda_eff = (float(cfg.lambda_max) * reliability).clamp(0.0, 1.0)
            zq_pool_final = (1.0 - lambda_eff).unsqueeze(1) * zq_pool_source + lambda_eff.unsqueeze(1) * zq_pool_repaired

        tokens = vqsa._tokens_from_pooled(z_pool, zq_pool_final)
        attn_weights = []
        if vqsa.use_self_attention:
            for block in vqsa.attention:
                tokens, weights = block(tokens)
                attn_weights.append(weights)
        fused = tokens.flatten(1)
        logits = self.model.classifier(fused)

        commitment = F.mse_loss(z, zq_repaired.detach())
        codebook_loss = F.mse_loss(zq_repaired, z.detach())
        vq_loss = codebook_loss + float(getattr(vqsa.vq, "commitment_cost", 0.25)) * commitment
        debug = {
            "z": z,
            "zq": zq_repaired,
            "z_pool": z_pool,
            "zq_pool": zq_pool_final,
            "tokens": tokens,
            "attention_weights": torch.stack(attn_weights, dim=1) if attn_weights else None,
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment,
            "dq_map": dq_repaired,
            "soft_assign": soft_repaired,
            "encoding_indices": idx_repaired,
            "quantizer_type": vqsa.quantizer_type,
            "num_embeddings": int(self.codebook_view.size(0)),
            "features": feats,
            "fused": fused,
        }
        debug = self._attach_repair_debug(debug, zq_pool_source, zq_pool_repaired, updated, reseeded, dq_source, reliability)
        self._batch_idx += 1
        self.stats.seen += x.size(0)
        self.stats.selected += int(reliable.sum().item())
        self.stats.reliable += int(reliable.sum().item())
        if updated > 0 or reseeded > 0:
            self.stats.updates += 1
        if return_debug:
            return logits, debug
        return logits

    @torch.no_grad()
    def _attach_repair_debug(
        self,
        dbg: dict,
        zq_pool_source: torch.Tensor,
        zq_pool_repaired: torch.Tensor,
        updated_codes: int,
        reseeded_codes: int,
        dq_source: Optional[torch.Tensor],
        reliability: Optional[torch.Tensor],
    ) -> dict:
        b = zq_pool_source.size(0)
        device = zq_pool_source.device
        dtype = zq_pool_source.dtype
        indices = dbg.get("encoding_indices")
        if indices is not None:
            usage, dead, hard_ppl = _hard_metrics(indices.detach(), self.codebook_view.size(0))
        else:
            usage = dead = hard_ppl = torch.zeros(b, device=device, dtype=dtype)
        delta = self.codebook_view - self.codebook_source
        drift = delta.norm(dim=1).mean()
        churn = (delta.norm(dim=1) > 1e-4).float().mean()
        if dq_source is None:
            dq_delta = torch.zeros(b, device=device, dtype=dtype)
        else:
            dq_delta = dbg["dq_map"].flatten(1).mean(dim=1) - dq_source.flatten(1).mean(dim=1)
        if reliability is None:
            reliability = torch.zeros(b, device=device, dtype=dtype)
        out = {
            **dbg,
            "zq_pool_source": zq_pool_source.detach(),
            "zq_pool_repaired": zq_pool_repaired.detach(),
            "repair_hard_usage": usage.to(device=device, dtype=dtype),
            "repair_dead_code_fraction": dead.to(device=device, dtype=dtype),
            "repair_hard_perplexity": hard_ppl.to(device=device, dtype=dtype),
            "repair_codebook_drift": torch.full((b,), float(drift.item()), device=device, dtype=dtype),
            "repair_code_churn": torch.full((b,), float(churn.item()), device=device, dtype=dtype),
            "repair_update_rate": torch.full((b,), updated_codes / max(1, self.codebook_view.size(0)), device=device, dtype=dtype),
            "repair_reseed_rate": torch.full((b,), reseeded_codes / max(1, self.codebook_view.size(0)), device=device, dtype=dtype),
            "repair_dq_delta": dq_delta.detach().to(device=device, dtype=dtype),
            "repair_reliability": reliability.detach().to(device=device, dtype=dtype),
        }
        return out
