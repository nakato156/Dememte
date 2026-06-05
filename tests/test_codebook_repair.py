"""CPU tests for E10-A local codebook repair."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dememte.codebook_repair import (
    CodebookRepairConfig,
    LocalCodebookRepairAdapter,
    quantize_with_codebook_view,
)
from dememte.config import E5Config
from dememte.evaluation import evaluate_dememte_tta_suite
from dememte.models import DeMemteVQSA


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(3, 512, kernel_size=3, padding=1)

    def forward(self, x):
        return self.net(x)


def tiny_config(**overrides):
    cfg = E5Config(
        num_classes=3,
        batch_size=2,
        num_workers=0,
        latent_dim=16,
        num_embeddings=32,
        vqsa_heads=4,
        vqsa_layers=1,
        vqsa_dropout=0.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_tiny_model(cfg=None):
    cfg = cfg or tiny_config()
    return DeMemteVQSA(
        backbone=TinyBackbone(),
        num_classes=cfg.num_classes,
        latent_dim=cfg.latent_dim,
        num_embeddings=cfg.num_embeddings,
        commitment_cost=cfg.commitment_cost,
        vq_temperature=cfg.vq_temperature,
        vqsa_heads=cfg.vqsa_heads,
        vqsa_layers=cfg.vqsa_layers,
        vqsa_dropout=cfg.vqsa_dropout,
        vqsa_fusion_mode=cfg.vqsa_fusion_mode,
        vqsa_use_codebook=cfg.vqsa_use_codebook,
        vqsa_use_self_attention=cfg.vqsa_use_self_attention,
        quantizer_type=getattr(cfg, "quantizer_type", "vq"),
        vq_ema_decay=getattr(cfg, "vq_ema_decay", 0.99),
        dead_code_threshold=getattr(cfg, "dead_code_threshold", 1.0),
        dead_code_restart_jitter=getattr(cfg, "dead_code_restart_jitter", 0.01),
        fsq_levels=getattr(cfg, "fsq_levels", 8),
    )


def make_loader(n=6):
    x = torch.randn(n, 3, 7, 7)
    y = torch.arange(n) % 3
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def test_quantize_with_codebook_view_shapes():
    z = torch.randn(2, 4, 3, 3)
    codebook = torch.randn(8, 4)
    zq, dq, soft, idx = quantize_with_codebook_view(z, codebook)
    assert zq.shape == z.shape
    assert dq.shape == (2, 1, 3, 3)
    assert soft.shape == (2, 3, 3, 8)
    assert idx.shape == (2, 3, 3)


def test_repair_disabled_identity_check_matches_source_logits():
    torch.manual_seed(0)
    model = make_tiny_model()
    x = torch.randn(2, 3, 7, 7)
    adapter = LocalCodebookRepairAdapter(model, CodebookRepairConfig(repair_mode="disabled"))
    with torch.no_grad():
        source_logits = model(x)
        adapter_logits, dbg = adapter(x, return_debug=True)
    assert torch.allclose(adapter_logits, source_logits, atol=1e-6)
    assert "repair_hard_usage" in dbg


def test_repair_changes_only_local_codebook_view_not_model_codebook():
    torch.manual_seed(1)
    model = make_tiny_model()
    original = model.vq.embedding.weight.detach().clone()
    cfg = CodebookRepairConfig(
        repair_mode="ema_reseed",
        blend_mode="replace",
        ema_lr=0.2,
        repair_every=1,
        max_reseeds=2,
        anchor_strength=0.0,
        max_code_drift=10.0,
    )
    adapter = LocalCodebookRepairAdapter(model, cfg)
    adapter(torch.randn(2, 3, 7, 7))
    assert torch.allclose(model.vq.embedding.weight, original)
    assert not torch.allclose(adapter.codebook_view, adapter.codebook_source)


def test_ema_update_moves_used_code_toward_assigned_token():
    torch.manual_seed(2)
    model = make_tiny_model(tiny_config(latent_dim=4, num_embeddings=8))
    adapter = LocalCodebookRepairAdapter(
        model,
        CodebookRepairConfig(repair_mode="ema", ema_lr=1.0, anchor_strength=0.0, max_code_drift=10.0),
    )
    token = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])
    indices = torch.zeros(1, 1, 1, dtype=torch.long)
    reliable = torch.ones(1, 1, 1, dtype=torch.bool)
    adapter._ema_update(token, indices, reliable)
    assert torch.allclose(adapter.codebook_view[0], token.flatten(), atol=1e-6)


def test_reseed_replaces_dead_code_with_high_error_token():
    torch.manual_seed(3)
    model = make_tiny_model(tiny_config(latent_dim=4, num_embeddings=8))
    adapter = LocalCodebookRepairAdapter(
        model,
        CodebookRepairConfig(repair_mode="reseed", repair_every=1, max_reseeds=1, anchor_strength=0.0, max_code_drift=10.0),
    )
    before = adapter.codebook_view.clone()
    z = torch.randn(1, 4, 1, 2)
    dq = torch.tensor([[[[0.1, 9.0]]]])
    n = adapter._reseed_dead(z, dq)
    assert n == 1
    assert not torch.allclose(adapter.codebook_view, before)


def test_anchor_and_drift_cap_limit_codebook_view():
    torch.manual_seed(4)
    model = make_tiny_model(tiny_config(latent_dim=4, num_embeddings=8))
    adapter = LocalCodebookRepairAdapter(
        model,
        CodebookRepairConfig(anchor_strength=0.0, max_code_drift=0.5),
    )
    adapter.codebook_view.copy_(adapter.codebook_source + 10.0)
    adapter._apply_anchor_and_cap()
    drift = (adapter.codebook_view - adapter.codebook_source).norm(dim=1)
    assert torch.all(drift <= 0.5001)


def test_evaluate_dememte_tta_suite_emits_repair_columns():
    torch.manual_seed(5)
    model = make_tiny_model()
    loader = make_loader()

    def factory():
        return LocalCodebookRepairAdapter(
            model,
            CodebookRepairConfig(
                repair_mode="ema_reseed",
                blend_mode="gated",
                repair_every=1,
                max_reseeds=1,
                anchor_strength=0.0,
                max_code_drift=1.0,
            ),
        )

    metrics = evaluate_dememte_tta_suite(
        factory,
        loader,
        device="cpu",
        suite={"gaussian_noise": [0.1]},
        tta_method="repair_test",
        tta_base_variant="tiny",
    )
    for key in (
        "repair_hard_usage_clean",
        "repair_dead_code_fraction_corrupt_avg",
        "repair_codebook_drift_corrupt_avg",
        "repair_dq_delta_corrupt_avg",
        "repair_reliability_clean",
    ):
        assert key in metrics
