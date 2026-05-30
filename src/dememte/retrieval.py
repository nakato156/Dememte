"""E11 retrieval memory: cache/kNN logits that vote next to the classifier.

This module keeps DeMemte fully frozen and adds a non-parametric memory head:

``logits_final = logits_base + alpha_eff(x) * logits_cache``.

Unlike E10's soft blend in ``zq_pool``, E11 gives memory a direct route to the
decision surface while still exposing diagnostics through the standard TTA
adapter interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tta import TTAStats


VALID_KEY_SPACES = {"z_pool", "zq_pool", "fused"}
VALID_ALPHA_MODES = {"fixed", "familiarity", "unfamiliarity"}


@dataclass
class RetrievalConfig:
    """Configuration for E11 retrieval-logit memory."""

    key_space: str = "zq_pool"
    top_k: int = 16
    beta: float = 5.0
    alpha_max: float = 1.0
    alpha_mode: str = "unfamiliarity"
    episodic_size: int = 256
    write_confidence: float = 0.8
    cache_source: bool = True


def _validate_key_space(key_space: str) -> None:
    if key_space not in VALID_KEY_SPACES:
        raise ValueError(f"key_space must be one of {sorted(VALID_KEY_SPACES)}, got {key_space!r}")


def _validate_alpha_mode(alpha_mode: str) -> None:
    if alpha_mode not in VALID_ALPHA_MODES:
        raise ValueError(f"alpha_mode must be one of {sorted(VALID_ALPHA_MODES)}, got {alpha_mode!r}")


def extract_retrieval_key(dbg: dict, key_space: str) -> torch.Tensor:
    """Extract the requested per-sample key from a DeMemte debug dict."""
    _validate_key_space(key_space)
    key = dbg[key_space]
    if key.dim() != 2:
        raise ValueError(f"retrieval key {key_space!r} must be 2D (B,D), got {tuple(key.shape)}")
    return key


class RetrievalCache(nn.Module):
    """Normalized key-value memory for retrieval logits.

    Keys are stored L2-normalized. Values are one-hot label rows. Querying uses
    Tip-Adapter-style affinities ``exp(-beta * (1 - cosine_sim))`` over top-k
    neighbors and returns class logits plus retrieval diagnostics.
    """

    def __init__(
        self,
        keys: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        super().__init__()
        if labels is not None and labels.dim() == 1:
            if num_classes is None:
                num_classes = int(labels.max().item()) + 1 if labels.numel() else 0
            values = F.one_hot(labels.long(), num_classes=num_classes).float()
        elif labels is not None:
            values = labels.float()
            if num_classes is None:
                num_classes = values.size(1)
        else:
            values = None

        if keys is None:
            if num_classes is None:
                raise ValueError("RetrievalCache requires num_classes when initialized empty.")
            keys = torch.empty(0, 0)
            values = torch.empty(0, int(num_classes))
        if values is None:
            raise ValueError("RetrievalCache requires labels/values when keys are provided.")
        if keys.dim() != 2 or values.dim() != 2 or keys.size(0) != values.size(0):
            raise ValueError(
                "RetrievalCache expects keys (N,D) and labels/values (N,C) with matching N; "
                f"got {tuple(keys.shape)} and {tuple(values.shape)}"
            )
        self.max_size = int(max_size) if max_size is not None else None
        self.num_classes = int(values.size(1))
        self.register_buffer("keys", F.normalize(keys.float(), dim=1) if keys.numel() else keys.float())
        self.register_buffer("values", values.float())

    @property
    def size(self) -> int:
        return int(self.values.size(0))

    @torch.no_grad()
    def append(self, keys: torch.Tensor, labels: torch.Tensor) -> None:
        """Append rows and enforce ``max_size`` by keeping the newest rows."""
        if keys.numel() == 0:
            return
        if labels.dim() == 1:
            values = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        else:
            values = labels.float()
        if values.size(1) != self.num_classes:
            raise ValueError(f"labels must have {self.num_classes} classes, got {values.size(1)}")
        keys = F.normalize(keys.detach().float().to(self.keys.device), dim=1)
        values = values.detach().float().to(self.values.device)
        if self.keys.numel() == 0:
            self.keys = keys
        else:
            self.keys = torch.cat([self.keys, keys], dim=0)
        self.values = torch.cat([self.values, values], dim=0)
        if self.max_size is not None and self.size > self.max_size:
            self.keys = self.keys[-self.max_size :]
            self.values = self.values[-self.max_size :]

    @torch.no_grad()
    def query(self, query_keys: torch.Tensor, top_k: int = 16, beta: float = 5.0) -> dict:
        """Return retrieval logits and per-sample diagnostics for ``query_keys``."""
        b = query_keys.size(0)
        device = query_keys.device
        if self.size == 0:
            zeros = torch.zeros(b, self.num_classes, device=device, dtype=query_keys.dtype)
            return {
                "logits": zeros,
                "top_affinity": torch.zeros(b, device=device, dtype=query_keys.dtype),
                "margin": torch.zeros(b, device=device, dtype=query_keys.dtype),
                "entropy": torch.zeros(b, device=device, dtype=query_keys.dtype),
            }
        q = F.normalize(query_keys.float(), dim=1)
        keys = self.keys.to(device=device, dtype=q.dtype)
        values = self.values.to(device=device, dtype=q.dtype)
        sims = q @ keys.t()
        k = max(1, min(int(top_k), self.size))
        top_sim, top_idx = sims.topk(k, dim=1)
        affinity = torch.exp(-float(beta) * (1.0 - top_sim))
        top_values = values[top_idx]
        logits = (affinity.unsqueeze(-1) * top_values).sum(dim=1)
        probs = logits / logits.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
        if logits.size(1) > 1:
            top2 = logits.topk(2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = logits[:, 0]
        return {
            "logits": logits.to(dtype=query_keys.dtype),
            "top_affinity": affinity.max(dim=1).values.to(dtype=query_keys.dtype),
            "margin": margin.to(dtype=query_keys.dtype),
            "entropy": entropy.to(dtype=query_keys.dtype),
        }


@torch.no_grad()
def build_labeled_cache(
    model: nn.Module,
    loader,
    device: str = "cuda",
    key_space: str = "zq_pool",
    num_classes: Optional[int] = None,
    max_items: Optional[int] = None,
) -> RetrievalCache:
    """Build a labeled retrieval cache from a loader with ground-truth labels."""
    _validate_key_space(key_space)
    model.eval()
    keys, labels = [], []
    seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, _, dbg = model(x, return_debug=True)
        del logits
        batch_keys = extract_retrieval_key(dbg, key_space).detach().cpu()
        keys.append(batch_keys)
        labels.append(y.detach().cpu().long())
        seen += y.size(0)
        if max_items is not None and seen >= max_items:
            break
    key_tensor = torch.cat(keys, dim=0) if keys else torch.empty(0, 0)
    label_tensor = torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long)
    if max_items is not None:
        key_tensor = key_tensor[:max_items]
        label_tensor = label_tensor[:max_items]
    if num_classes is None:
        num_classes = int(getattr(model, "classifier")[-1].out_features)
    return RetrievalCache(key_tensor, label_tensor, num_classes=num_classes)


class RetrievalLogitAdapter(nn.Module):
    """Frozen DeMemte wrapper that adds retrieval-cache logits to base logits."""

    method_name = "retrieval_logit"

    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[RetrievalConfig] = None,
        source_cache: Optional[RetrievalCache] = None,
        teacher_model: Optional[nn.Module] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        cfg = cfg or RetrievalConfig()
        _validate_key_space(cfg.key_space)
        _validate_alpha_mode(cfg.alpha_mode)
        if cfg.top_k < 1:
            raise ValueError("RetrievalConfig.top_k must be >= 1")
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.teacher_model = teacher_model if teacher_model is not None else model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.cfg = cfg
        self.stats = TTAStats()
        self.source_cache = source_cache if cfg.cache_source else None
        if num_classes is None:
            num_classes = int(getattr(model, "classifier")[-1].out_features)
        self.num_classes = int(num_classes)
        self.episodic_cache = RetrievalCache(num_classes=self.num_classes, max_size=cfg.episodic_size)

    def reset(self) -> None:
        self.stats = TTAStats()
        self.episodic_cache = RetrievalCache(num_classes=self.num_classes, max_size=self.cfg.episodic_size)

    def _alpha(self, base_probs: torch.Tensor, cache_info: dict) -> torch.Tensor:
        cfg = self.cfg
        if cfg.alpha_mode == "fixed":
            return torch.full((base_probs.size(0),), float(cfg.alpha_max), device=base_probs.device)
        if cfg.alpha_mode == "familiarity":
            return float(cfg.alpha_max) * cache_info["top_affinity"].clamp(0.0, 1.0)
        base_conf = base_probs.max(dim=1).values
        return float(cfg.alpha_max) * (1.0 - base_conf).clamp(0.0, 1.0)

    @torch.no_grad()
    def _write_episodic(self, keys: torch.Tensor, teacher_logits: torch.Tensor) -> int:
        probs = teacher_logits.softmax(dim=1)
        conf, pseudo = probs.max(dim=1)
        mask = conf >= float(self.cfg.write_confidence)
        if mask.any():
            self.episodic_cache.append(keys[mask].detach().cpu(), pseudo[mask].detach().cpu())
        return int(mask.sum().item())

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        base_logits, _, dbg = self.model(x, return_debug=True)
        query_key = extract_retrieval_key(dbg, self.cfg.key_space)
        cache_logits = torch.zeros_like(base_logits)
        margin = torch.zeros(x.size(0), device=x.device, dtype=base_logits.dtype)
        entropy = torch.zeros(x.size(0), device=x.device, dtype=base_logits.dtype)
        top_affinity = torch.zeros(x.size(0), device=x.device, dtype=base_logits.dtype)

        queried = 0
        for cache in (self.source_cache, self.episodic_cache):
            if cache is None or cache.size == 0:
                continue
            info = cache.query(query_key, top_k=self.cfg.top_k, beta=self.cfg.beta)
            cache_logits = cache_logits + info["logits"].to(device=x.device, dtype=base_logits.dtype)
            margin = torch.maximum(margin, info["margin"].to(device=x.device, dtype=base_logits.dtype))
            entropy = entropy + info["entropy"].to(device=x.device, dtype=base_logits.dtype)
            top_affinity = torch.maximum(
                top_affinity, info["top_affinity"].to(device=x.device, dtype=base_logits.dtype)
            )
            queried += 1
        if queried > 0:
            entropy = entropy / queried
        cache_info = {"top_affinity": top_affinity}
        base_probs = base_logits.softmax(dim=1)
        alpha_eff = self._alpha(base_probs, cache_info)
        final_logits = base_logits + alpha_eff.unsqueeze(1) * cache_logits

        if self.cfg.episodic_size > 0:
            teacher_logits, _, _ = self.teacher_model(x, return_debug=True)
            written = self._write_episodic(query_key, teacher_logits)
        else:
            written = 0
        self.stats.seen += x.size(0)
        self.stats.selected += written
        self.stats.reliable += written
        self.stats.updates += 0

        base_pred = base_logits.argmax(dim=1)
        cache_pred = cache_logits.argmax(dim=1)
        final_pred = final_logits.argmax(dim=1)
        retrieval_agreement = (base_pred == cache_pred).float()

        dbg = {
            **dbg,
            "base_logits": base_logits.detach(),
            "cache_logits": cache_logits.detach(),
            "retrieval_alpha": alpha_eff.detach(),
            "retrieval_margin": margin.detach(),
            "retrieval_entropy": entropy.detach(),
            "retrieval_agreement": retrieval_agreement.detach(),
            "base_pred": base_pred.detach(),
            "cache_pred": cache_pred.detach(),
            "final_pred": final_pred.detach(),
            "alpha_eff": alpha_eff.detach(),
        }
        if return_debug:
            return final_logits, dbg
        return final_logits
