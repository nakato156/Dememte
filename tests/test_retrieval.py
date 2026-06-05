"""CPU tests for E11 retrieval-logit memory."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dememte.config import E5Config
from dememte.evaluation import evaluate_dememte_tta_suite
from dememte.models import DeMemteVQSA
from dememte.retrieval import (
    RetrievalCache,
    RetrievalConfig,
    RetrievalLogitAdapter,
    build_labeled_cache,
    extract_retrieval_key,
)


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


def test_retrieval_alpha_zero_matches_source_logits():
    torch.manual_seed(0)
    model = make_tiny_model()
    loader = make_loader()
    cache = build_labeled_cache(model, loader, device="cpu", key_space="zq_pool", num_classes=3)
    cfg = RetrievalConfig(alpha_max=0.0, alpha_mode="fixed", key_space="zq_pool")
    adapter = RetrievalLogitAdapter(model, cfg, source_cache=cache, num_classes=3)
    x, _ = next(iter(loader))
    with torch.no_grad():
        source_logits = model(x)
        adapter_logits = adapter(x)
    assert torch.allclose(adapter_logits, source_logits, atol=1e-6)


def test_retrieval_cache_perfect_neighbor_increases_matching_class_logit():
    keys = torch.eye(3)
    labels = torch.tensor([0, 1, 2])
    cache = RetrievalCache(keys, labels, num_classes=3)
    out = cache.query(keys[1:2], top_k=1, beta=5.0)
    logits = out["logits"]
    assert logits[0, 1] > logits[0, 0]
    assert logits[0, 1] > logits[0, 2]


def test_retrieval_adapter_does_not_mutate_model_parameters():
    torch.manual_seed(1)
    model = make_tiny_model()
    loader = make_loader()
    cache = build_labeled_cache(model, loader, device="cpu", key_space="zq_pool", num_classes=3)
    adapter = RetrievalLogitAdapter(
        model,
        RetrievalConfig(alpha_max=1.0, alpha_mode="fixed", key_space="zq_pool"),
        source_cache=cache,
        num_classes=3,
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    x, _ = next(iter(loader))
    adapter(x)
    after = dict(model.named_parameters())
    for name, param in before.items():
        assert torch.allclose(after[name], param)


def test_retrieval_episodic_cache_writes_only_when_confident():
    torch.manual_seed(2)
    model = make_tiny_model()
    x, _ = next(iter(make_loader()))

    writer = RetrievalLogitAdapter(
        model,
        RetrievalConfig(cache_source=False, write_confidence=0.0, episodic_size=8),
        source_cache=None,
        num_classes=3,
    )
    writer(x)
    assert writer.episodic_cache.size == x.size(0)
    assert writer.stats.selected == x.size(0)

    non_writer = RetrievalLogitAdapter(
        model,
        RetrievalConfig(cache_source=False, write_confidence=1.1, episodic_size=8),
        source_cache=None,
        num_classes=3,
    )
    non_writer(x)
    assert non_writer.episodic_cache.size == 0
    assert non_writer.stats.selected == 0


def test_evaluate_dememte_tta_suite_emits_retrieval_columns():
    torch.manual_seed(3)
    model = make_tiny_model()
    loader = make_loader()
    cache = build_labeled_cache(model, loader, device="cpu", key_space="zq_pool", num_classes=3)

    def factory():
        return RetrievalLogitAdapter(
            model,
            RetrievalConfig(alpha_max=0.5, alpha_mode="fixed", key_space="zq_pool"),
            source_cache=cache,
            num_classes=3,
        )

    metrics = evaluate_dememte_tta_suite(
        factory,
        loader,
        device="cpu",
        suite={"gaussian_noise": [0.1]},
        tta_method="retrieval_test",
        tta_base_variant="tiny",
    )
    for key in (
        "retrieval_alpha_clean",
        "retrieval_margin_corrupt_avg",
        "flip_rate_corrupt_avg",
        "corrected_by_retrieval_corrupt_avg",
        "broken_by_retrieval_corrupt_avg",
    ):
        assert key in metrics


def test_retrieval_key_spaces_have_expected_dimensions():
    torch.manual_seed(4)
    cfg = tiny_config(latent_dim=16)
    model = make_tiny_model(cfg)
    x = torch.randn(2, 3, 7, 7)
    with torch.no_grad():
        _, _, dbg = model(x, return_debug=True)
    assert extract_retrieval_key(dbg, "z_pool").shape == (2, 16)
    assert extract_retrieval_key(dbg, "zq_pool").shape == (2, 16)
    assert extract_retrieval_key(dbg, "fused").shape == (2, 32)
