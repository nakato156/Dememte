"""Vector quantizer + latent projector / unprojector for the DeMemte memory block."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _squared_distances(flat: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    return flat.pow(2).sum(1, keepdim=True) - 2 * flat @ emb.t() + emb.pow(2).sum(1, keepdim=True).t()


def _usage_from_indices(indices: torch.Tensor, num_codes: int, dtype: torch.dtype) -> torch.Tensor:
    flat_idx = indices.reshape(-1).long()
    return torch.bincount(flat_idx, minlength=num_codes).to(device=indices.device, dtype=dtype)


class VectorQuantizer2D(nn.Module):
    """Spatial VQ with straight-through estimator and diagnostic losses."""

    quantizer_type = "vq"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        temperature: float = 1.0,
        return_indices: bool = False,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        self.return_indices = return_indices
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def _pack(self, values, indices):
        if self.return_indices:
            return (*values, indices)
        return values

    def forward(self, z_e: torch.Tensor):
        b, c, h, w = z_e.shape
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e_perm.view(-1, c)
        emb = self.embedding.weight
        distances = _squared_distances(flat, emb)
        idx = torch.argmin(distances, dim=1)
        one_hot = F.one_hot(idx, self.num_embeddings).type(flat.dtype)
        q_flat = one_hot @ emb
        q = q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        commitment_loss = F.mse_loss(z_e, q.detach())
        codebook_loss = F.mse_loss(q, z_e.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        q_st = z_e + (q - z_e).detach()
        dq_map = ((z_e - q.detach()) ** 2).mean(dim=1, keepdim=True)
        temp = max(1e-6, float(self.temperature))
        soft_assign = F.softmax(-distances / temp, dim=1).view(b, h, w, self.num_embeddings)
        idx_map = idx.view(b, h, w)
        return self._pack((q_st, vq_loss, codebook_loss, commitment_loss, dq_map, soft_assign), idx_map)


class EMAVectorQuantizer2D(VectorQuantizer2D):
    """Spatial VQ whose codebook is updated by exponential moving averages."""

    quantizer_type = "ema_vq"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        temperature: float = 1.0,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
        restart_jitter: float = 0.01,
        return_indices: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim, commitment_cost, temperature, return_indices=return_indices)
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        self.restart_jitter = restart_jitter
        self.embedding.weight.requires_grad_(False)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.detach().clone())
        self.register_buffer("usage_ema", torch.zeros(num_embeddings))

    @torch.no_grad()
    def initialize_from_data(self, z_e: torch.Tensor, steps: int = 10) -> None:
        """Initialize the codebook with a small k-means run over latent tokens."""
        b, c, h, w = z_e.shape
        flat = z_e.detach().permute(0, 2, 3, 1).reshape(-1, c)
        if flat.numel() == 0:
            return
        if flat.size(0) < self.num_embeddings:
            repeats = (self.num_embeddings + flat.size(0) - 1) // flat.size(0)
            centers = flat.repeat(repeats, 1)[: self.num_embeddings].clone()
        else:
            perm = torch.randperm(flat.size(0), device=flat.device)[: self.num_embeddings]
            centers = flat[perm].clone()
        for _ in range(max(1, steps)):
            distances = _squared_distances(flat, centers)
            idx = torch.argmin(distances, dim=1)
            counts = _usage_from_indices(idx, self.num_embeddings, flat.dtype).clamp_min(1.0)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, idx, flat)
            updated = sums / counts.unsqueeze(1)
            centers = torch.where(counts.unsqueeze(1) > 1.0, updated, centers)
        self.embedding.weight.data.copy_(centers)
        self.ema_w.copy_(centers)
        self.ema_cluster_size.fill_(1.0)
        self.usage_ema.fill_(1.0)

    @torch.no_grad()
    def restart_dead_codes(self, flat: torch.Tensor) -> None:
        dead = self.usage_ema < self.dead_code_threshold
        num_dead = int(dead.sum().item())
        if num_dead == 0 or flat.numel() == 0:
            return
        sample_idx = torch.randint(0, flat.size(0), (num_dead,), device=flat.device)
        samples = flat[sample_idx]
        if self.restart_jitter > 0:
            samples = samples + torch.randn_like(samples) * self.restart_jitter
        self.embedding.weight.data[dead] = samples
        self.ema_w[dead] = samples
        self.ema_cluster_size[dead] = 1.0
        self.usage_ema[dead] = 1.0

    def forward(self, z_e: torch.Tensor):
        b, c, h, w = z_e.shape
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e_perm.view(-1, c)
        emb = self.embedding.weight
        distances = _squared_distances(flat, emb)
        idx = torch.argmin(distances, dim=1)
        one_hot = F.one_hot(idx, self.num_embeddings).type(flat.dtype)
        q_flat = one_hot @ emb
        q = q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        commitment_loss = F.mse_loss(z_e, q.detach())
        codebook_loss = F.mse_loss(q.detach(), z_e.detach())
        vq_loss = self.commitment_cost * commitment_loss
        q_st = z_e + (q - z_e).detach()
        dq_map = ((z_e - q.detach()) ** 2).mean(dim=1, keepdim=True)
        temp = max(1e-6, float(self.temperature))
        soft_assign = F.softmax(-distances / temp, dim=1).view(b, h, w, self.num_embeddings)
        if self.training:
            with torch.no_grad():
                counts = one_hot.sum(dim=0)
                sums = one_hot.t() @ flat.detach()
                self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
                self.ema_w.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
                self.embedding.weight.data.copy_(self.ema_w / smoothed.unsqueeze(1).clamp_min(self.eps))
                self.usage_ema.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        idx_map = idx.view(b, h, w)
        return self._pack((q_st, vq_loss, codebook_loss, commitment_loss, dq_map, soft_assign), idx_map)


class SimVQLinearQuantizer2D(VectorQuantizer2D):
    """SimVQ-style codebook generated from a shared linear transform."""

    quantizer_type = "simvq_linear"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        temperature: float = 1.0,
        return_indices: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim, commitment_cost, temperature, return_indices=return_indices)
        weight = self.embedding.weight.detach().clone()
        del self.embedding
        self.codebook_base = nn.Parameter(weight)
        self.codebook_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.eye_(self.codebook_transform.weight)

    @property
    def codebook(self) -> torch.Tensor:
        return self.codebook_transform(self.codebook_base)

    def forward(self, z_e: torch.Tensor):
        b, c, h, w = z_e.shape
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e_perm.view(-1, c)
        emb = self.codebook
        distances = _squared_distances(flat, emb)
        idx = torch.argmin(distances, dim=1)
        one_hot = F.one_hot(idx, self.num_embeddings).type(flat.dtype)
        q_flat = one_hot @ emb
        q = q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        commitment_loss = F.mse_loss(z_e, q.detach())
        codebook_loss = F.mse_loss(q, z_e.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        q_st = z_e + (q - z_e).detach()
        dq_map = ((z_e - q.detach()) ** 2).mean(dim=1, keepdim=True)
        temp = max(1e-6, float(self.temperature))
        soft_assign = F.softmax(-distances / temp, dim=1).view(b, h, w, self.num_embeddings)
        idx_map = idx.view(b, h, w)
        return self._pack((q_st, vq_loss, codebook_loss, commitment_loss, dq_map, soft_assign), idx_map)


class FSQQuantizer2D(nn.Module):
    """Lookup-free finite scalar quantization with straight-through rounding."""

    quantizer_type = "fsq"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        temperature: float = 1.0,
        levels: int | Iterable[int] = 8,
        return_indices: bool = False,
    ):
        super().__init__()
        self.num_embeddings = int(levels[0] if isinstance(levels, (tuple, list)) else levels)
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        self.return_indices = return_indices

    def _pack(self, values, indices):
        if self.return_indices:
            return (*values, indices)
        return values

    def forward(self, z_e: torch.Tensor):
        levels = max(2, int(self.num_embeddings))
        bounded = torch.tanh(z_e)
        scaled = (bounded + 1.0) * 0.5 * (levels - 1)
        rounded = torch.round(scaled).clamp(0, levels - 1)
        q = (rounded / (levels - 1)) * 2.0 - 1.0
        q_st = z_e + (q - z_e).detach()
        commitment_loss = F.mse_loss(z_e, q.detach())
        codebook_loss = z_e.sum() * 0.0
        vq_loss = self.commitment_cost * commitment_loss
        dq_map = ((z_e - q.detach()) ** 2).mean(dim=1, keepdim=True)
        idx = rounded.long().permute(0, 2, 3, 1).contiguous()
        return self._pack((q_st, vq_loss, codebook_loss, commitment_loss, dq_map, None), idx)


def make_quantizer2d(
    quantizer_type: str,
    num_embeddings: int,
    embedding_dim: int,
    commitment_cost: float = 0.25,
    temperature: float = 1.0,
    ema_decay: float = 0.99,
    dead_code_threshold: float = 1.0,
    restart_jitter: float = 0.01,
    fsq_levels: int = 8,
    return_indices: bool = False,
) -> nn.Module:
    quantizer_type = str(quantizer_type)
    if quantizer_type == "vq":
        return VectorQuantizer2D(num_embeddings, embedding_dim, commitment_cost, temperature, return_indices=return_indices)
    if quantizer_type == "ema_vq":
        return EMAVectorQuantizer2D(
            num_embeddings,
            embedding_dim,
            commitment_cost,
            temperature,
            decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
            restart_jitter=restart_jitter,
            return_indices=return_indices,
        )
    if quantizer_type == "simvq_linear":
        return SimVQLinearQuantizer2D(num_embeddings, embedding_dim, commitment_cost, temperature, return_indices=return_indices)
    if quantizer_type == "fsq":
        return FSQQuantizer2D(num_embeddings, embedding_dim, commitment_cost, temperature, levels=fsq_levels, return_indices=return_indices)
    raise ValueError("Unknown quantizer_type: {!r}. Choices: ['vq', 'ema_vq', 'simvq_linear', 'fsq']".format(quantizer_type))


class LatentProjector(nn.Module):
    def __init__(self, in_channels: int = 512, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentUnprojector(nn.Module):
    def __init__(self, latent_dim: int = 128, out_channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def sigreg_weak_loss(x: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """Frobenius distance between sketched feature covariance and identity."""
    if x.dim() != 2:
        x = x.view(-1, x.size(-1))
    n, c = x.size()
    if n <= 1:
        return x.new_zeros(())
    if c > sketch_dim:
        sketch = torch.randn(sketch_dim, c, device=x.device, dtype=x.dtype) / (c ** 0.5)
        x = x @ sketch.t()
        c = sketch_dim
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.t() @ x) / (n - 1 + 1e-6)
    target = torch.eye(c, device=x.device, dtype=x.dtype)
    return torch.norm(cov - target, p="fro")


def sigreg_latent_loss(z: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    tokens = z.flatten(2).transpose(1, 2).reshape(-1, z.size(1))
    return sigreg_weak_loss(tokens, sketch_dim=sketch_dim)
