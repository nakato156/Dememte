"""Vector quantizer + latent projector / unprojector for the DeMemte memory block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer2D(nn.Module):
    """Spatial VQ with straight-through estimator and diagnostic losses."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, temperature: float = 1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        b, c, h, w = z_e.shape
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e_perm.view(-1, c)
        emb = self.embedding.weight
        distances = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ emb.t() + emb.pow(2).sum(1, keepdim=True).t()
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
        return q_st, vq_loss, codebook_loss, commitment_loss, dq_map, soft_assign


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
