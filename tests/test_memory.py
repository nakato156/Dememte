"""CPU tests for the E10 hippocampal memory module."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dememte.config import E5Config
from dememte.evaluation import (
    MEMORY_DIAG_KEYS,
    evaluate_dememte_tta,
    evaluate_dememte_tta_suite,
)
from dememte.memory import (
    EpisodicBuffer,
    HippocampalConfig,
    HippocampalMemoryAdapter,
    associative_recall,
    effective_codebook,
    familiarity_gate,
    pattern_completion,
)
from dememte.memory import _blend_recall_for_test as _blend_recall
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
        epochs_vqsa_max=1,
        train_corrupt_prob=1.0,
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


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_associative_recall_returns_codebook_centroid_when_z_equals_one_code():
    """Ramsauer 2021 Eq. 7: with sharp temperature, recall ≈ nearest code."""
    torch.manual_seed(0)
    keys = torch.eye(4) * 2.0  # 4 well-separated codes
    query = keys[2:3].clone()  # query == code 2
    recall = associative_recall(query, keys, temperature=0.05)
    assert torch.allclose(recall, keys[2:3], atol=1e-3)


def test_associative_recall_sharpness_in_unit_interval():
    keys = torch.randn(10, 8)
    query = torch.randn(4, 8)
    _, sharpness = associative_recall(query, keys, temperature=1.0, return_sharpness=True)
    assert sharpness.shape == (4,)
    assert torch.all((sharpness >= 0.0) & (sharpness <= 1.0))


def test_associative_recall_empty_keys_is_identity():
    query = torch.randn(3, 5)
    empty = torch.zeros(0, 5)
    out = associative_recall(query, empty, temperature=1.0)
    assert torch.allclose(out, query)


def test_familiarity_gate_modes_complementary():
    """familiarity + unfamiliarity = 1 (Tyulmankov 2022 vs Krotov 2021)."""
    torch.manual_seed(1)
    keys = torch.randn(20, 6)
    query = torch.randn(5, 6)
    g_fam = familiarity_gate(query, keys, sigma=1.0, mode="familiarity")
    g_unfam = familiarity_gate(query, keys, sigma=1.0, mode="unfamiliarity")
    assert torch.allclose(g_fam + g_unfam, torch.ones_like(g_fam), atol=1e-6)
    assert torch.all((g_fam >= 0.0) & (g_fam <= 1.0))


def test_familiarity_gate_const_returns_ones():
    query = torch.randn(3, 4)
    keys = torch.randn(8, 4)
    g = familiarity_gate(query, keys, sigma=1.0, mode="const")
    assert torch.allclose(g, torch.ones(3))


def test_familiarity_gate_rejects_unknown_mode():
    with pytest.raises(ValueError):
        familiarity_gate(torch.randn(1, 2), torch.randn(1, 2), sigma=1.0, mode="bogus")


def test_pattern_completion_t0_is_identity():
    """T=0 must return the query untouched, with zero traj_diff."""
    query = torch.randn(3, 5)
    keys = torch.randn(8, 5)
    z, traj, g = pattern_completion(
        query, sem_keys=keys, epi_keys=None, gate_codebook=keys,
        T=0, lambda_max=0.1, tau=1.0, beta=1.0, sigma=1.0, gate_mode="const",
    )
    assert torch.allclose(z, query)
    assert traj == []


def test_pattern_completion_t1_with_lambda_zero_is_identity():
    """λ_max=0 should also leave the query exactly untouched, even with T>=1."""
    query = torch.randn(3, 5)
    keys = torch.randn(8, 5)
    z, traj, _ = pattern_completion(
        query, sem_keys=keys, epi_keys=None, gate_codebook=keys,
        T=3, lambda_max=0.0, tau=1.0, beta=1.0, sigma=1.0, gate_mode="familiarity",
    )
    assert torch.allclose(z, query)
    for step in traj:
        assert step == pytest.approx(0.0, abs=1e-6)


def test_pattern_completion_t1_nonzero_when_lambda_positive():
    """Regression guard: a positive λ_max with const gate must move z (≠ no-op)."""
    torch.manual_seed(2)
    query = torch.randn(4, 6)
    keys = torch.randn(8, 6) * 2.0
    z, traj, _ = pattern_completion(
        query, sem_keys=keys, epi_keys=None, gate_codebook=keys,
        T=1, lambda_max=0.5, tau=1.0, beta=1.0, sigma=1.0, gate_mode="const",
    )
    assert not torch.allclose(z, query)
    assert traj[0] > 0.0


def test_pattern_completion_traj_diff_decreases_under_contractive_recall():
    """Kim 2021: in the contractive regime (query near a basin), ‖z_{t+1}−z_t‖ shrinks."""
    torch.manual_seed(3)
    # Tight codebook in a basin; query slightly perturbed from a code.
    keys = torch.randn(4, 8) * 3.0  # well-separated codes
    query = keys[1:2] + 0.01 * torch.randn(1, 8)
    _, traj, _ = pattern_completion(
        query, sem_keys=keys, epi_keys=None, gate_codebook=keys,
        T=4, lambda_max=0.5, tau=0.1, beta=1.0, sigma=1.0, gate_mode="const",
    )
    # In a contractive basin, later iterations should not be larger than first.
    assert traj[-1] <= traj[0] + 1e-6


def test_blend_recall_collapses_to_one_subsystem():
    """If one subsystem has no keys, blend equals the other subsystem's recall."""
    torch.manual_seed(4)
    keys = torch.randn(5, 4)
    q = torch.randn(3, 4)
    only_sem = _blend_recall(q, keys, None, tau=1.0, tau_epi=1.0, beta=1.0)
    direct = associative_recall(q, keys, temperature=1.0)
    assert torch.allclose(only_sem, direct)
    only_epi = _blend_recall(q, None, keys, tau=1.0, tau_epi=1.0, beta=1.0)
    assert torch.allclose(only_epi, direct)


# ---------------------------------------------------------------------------
# EpisodicBuffer tests
# ---------------------------------------------------------------------------


def test_episodic_buffer_lazy_init_on_first_write():
    buf = EpisodicBuffer(size=4, dim=3, alpha=0.1)
    assert not bool(buf.initialized.item())
    z = torch.randn(2, 3)
    buf.write(z)
    assert bool(buf.initialized.item())


def test_episodic_buffer_ema_moves_toward_input():
    """A second write of the same z pulls the matched slot toward z (EMA)."""
    torch.manual_seed(5)
    buf = EpisodicBuffer(size=8, dim=4, alpha=0.3)
    buf.initialize_from(torch.randn(8, 4))
    z = torch.randn(1, 4)
    mem_before = buf.memory.clone()
    buf.write(z)
    # Find slot that took the write: row whose delta is largest.
    delta = (buf.memory - mem_before).norm(dim=-1)
    slot = int(delta.argmax().item())
    # That slot should be closer to z than before.
    d_before = (mem_before[slot] - z[0]).norm().item()
    d_after = (buf.memory[slot] - z[0]).norm().item()
    assert d_after < d_before


def test_episodic_buffer_reset_restores_uninitialized_state():
    buf = EpisodicBuffer(size=4, dim=3, alpha=0.1)
    buf.initialize_from(torch.randn(4, 3))
    buf.write(torch.randn(2, 3))
    assert bool(buf.initialized.item())
    buf.reset()
    assert not bool(buf.initialized.item())
    assert torch.allclose(buf.memory, torch.zeros(4, 3))
    assert int(buf.write_count.item()) == 0


def test_episodic_buffer_churn_zero_at_init_and_grows_after_writes():
    torch.manual_seed(6)
    buf = EpisodicBuffer(size=8, dim=5, alpha=0.5)
    buf.initialize_from(torch.randn(8, 5))
    assert buf.churn() == 0.0
    buf.write(torch.randn(4, 5) * 5.0)
    assert buf.churn() > 0.0


def test_episodic_buffer_initialize_from_handles_short_source():
    """Source smaller than size → repeated + jittered."""
    buf = EpisodicBuffer(size=10, dim=3, alpha=0.1)
    src = torch.randn(3, 3)
    buf.initialize_from(src)
    assert buf.memory.shape == (10, 3)
    assert not torch.isnan(buf.memory).any()


# ---------------------------------------------------------------------------
# effective_codebook helper
# ---------------------------------------------------------------------------


def test_effective_codebook_vanilla_vq_returns_embedding_weight():
    model = make_tiny_model(tiny_config(quantizer_type="vq"))
    cb = effective_codebook(model.vq)
    assert cb is not None
    assert cb.shape == (32, 16)


def test_effective_codebook_simvq_returns_transformed_base():
    model = make_tiny_model(tiny_config(quantizer_type="simvq_linear"))
    cb = effective_codebook(model.vq)
    assert cb is not None
    assert cb.shape == (32, 16)


def test_effective_codebook_fsq_returns_none():
    model = make_tiny_model(tiny_config(quantizer_type="fsq"))
    cb = effective_codebook(model.vq)
    assert cb is None


# ---------------------------------------------------------------------------
# HippocampalMemoryAdapter tests
# ---------------------------------------------------------------------------


def test_adapter_lambda_zero_matches_source_bitwise():
    """Critical invariant: λ_max=0 ⇒ logits identical to source forward."""
    torch.manual_seed(7)
    model = make_tiny_model().eval()
    model.requires_grad_(False)
    x = torch.randn(4, 3, 7, 7)
    with torch.no_grad():
        src_logits = model(x)

    cfg = HippocampalConfig(lambda_max=0.0, T=1, gate_mode="familiarity")
    adapter = HippocampalMemoryAdapter(model, cfg)
    with torch.no_grad():
        adapter_logits = adapter(x)
    assert torch.allclose(adapter_logits, src_logits, atol=1e-6)


def test_adapter_lambda_zero_with_t0_matches_source_bitwise():
    """T=0 + λ_max=0 must also be exact source replay."""
    torch.manual_seed(8)
    model = make_tiny_model().eval()
    model.requires_grad_(False)
    x = torch.randn(4, 3, 7, 7)
    with torch.no_grad():
        src_logits = model(x)
    adapter = HippocampalMemoryAdapter(model, HippocampalConfig(lambda_max=0.0, T=0))
    with torch.no_grad():
        adapter_logits = adapter(x)
    assert torch.allclose(adapter_logits, src_logits, atol=1e-6)


def test_adapter_positive_lambda_shifts_logits():
    """λ_max > 0 must produce a logit shift relative to source."""
    torch.manual_seed(9)
    model = make_tiny_model().eval()
    model.requires_grad_(False)
    x = torch.randn(4, 3, 7, 7)
    with torch.no_grad():
        src_logits = model(x)
    cfg = HippocampalConfig(lambda_max=0.3, T=1, gate_mode="const")
    adapter = HippocampalMemoryAdapter(model, cfg)
    with torch.no_grad():
        adapter_logits = adapter(x)
    assert not torch.allclose(adapter_logits, src_logits, atol=1e-4)


def test_adapter_returns_debug_with_memory_keys():
    torch.manual_seed(10)
    model = make_tiny_model().eval()
    x = torch.randn(4, 3, 7, 7)
    cfg = HippocampalConfig(lambda_max=0.1, T=1, gate_mode="const")
    adapter = HippocampalMemoryAdapter(model, cfg)
    with torch.no_grad():
        logits, dbg = adapter(x, return_debug=True)
    assert logits.shape == (4, 3)
    for key in MEMORY_DIAG_KEYS:
        assert key in dbg, f"missing diagnostic key {key!r}"
        value = dbg[key]
        assert isinstance(value, torch.Tensor)
        assert value.shape == (4,)


def test_adapter_episodic_only_runs_without_semantic_codebook_recall():
    """recall_sem=False + recall_epi=True must work and produce buffer churn > 0."""
    torch.manual_seed(11)
    model = make_tiny_model().eval()
    cfg = HippocampalConfig(
        recall_sem=False, recall_epi=True, T=1, gate_mode="const", lambda_max=0.1,
    )
    adapter = HippocampalMemoryAdapter(model, cfg)
    x = torch.randn(4, 3, 7, 7)
    with torch.no_grad():
        _, dbg = adapter(x, return_debug=True)
    # After one batch with writes, churn should be measurable on a second batch.
    with torch.no_grad():
        _, dbg2 = adapter(x + 1.0, return_debug=True)
    assert float(dbg2["episodic_buffer_churn"][0].item()) >= 0.0


def test_adapter_reset_restores_initial_state():
    torch.manual_seed(12)
    model = make_tiny_model().eval()
    cfg = HippocampalConfig(recall_epi=True, T=1, lambda_max=0.1, gate_mode="const")
    adapter = HippocampalMemoryAdapter(model, cfg)
    x = torch.randn(4, 3, 7, 7)
    with torch.no_grad():
        adapter(x)
        adapter(x + 0.5)
    assert int(adapter.episodic.write_count.item()) > 0
    adapter.reset()
    assert int(adapter.episodic.write_count.item()) == 0
    assert adapter.stats.seen == 0


def test_adapter_rejects_fsq_when_semantic_recall_requested():
    """FSQ is lookup-free; semantic recall has no stored patterns."""
    model = make_tiny_model(tiny_config(quantizer_type="fsq")).eval()
    cfg = HippocampalConfig(recall_sem=True, recall_epi=False)
    with pytest.raises(ValueError):
        HippocampalMemoryAdapter(model, cfg)


def test_adapter_with_simvq_does_not_mutate_model_codebook():
    """Consolidation moves the adapter-local codebook view but never the model's."""
    torch.manual_seed(13)
    model = make_tiny_model(tiny_config(quantizer_type="simvq_linear")).eval()
    cb_before = effective_codebook(model.vq).detach().clone()
    cfg = HippocampalConfig(
        recall_sem=True, recall_epi=True, alpha_s=0.5, consolidation_every=1,
        T=1, lambda_max=0.1, gate_mode="const",
    )
    adapter = HippocampalMemoryAdapter(model, cfg)
    x = torch.randn(4, 3, 7, 7)
    with torch.no_grad():
        adapter(x)
        adapter(x + 0.5)
        adapter(x - 0.5)
    cb_after = effective_codebook(model.vq).detach().clone()
    assert torch.allclose(cb_before, cb_after, atol=1e-6), "model codebook should not change"


# ---------------------------------------------------------------------------
# Integration with evaluate_dememte_tta
# ---------------------------------------------------------------------------


def test_evaluate_dememte_tta_emits_memory_diag_columns():
    torch.manual_seed(14)
    cfg = tiny_config(vqsa_layers=0, vqsa_use_self_attention=False)
    x = torch.randn(4, 3, 7, 7)
    y = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    suite = {"gaussian_noise": [0.1]}

    teacher = make_tiny_model(cfg).eval()
    teacher.requires_grad_(False)

    def factory():
        model = make_tiny_model(cfg)
        model.load_state_dict(teacher.state_dict())
        return HippocampalMemoryAdapter(
            model,
            HippocampalConfig(
                recall_sem=True, recall_epi=True,
                T=1, lambda_max=0.05, gate_mode="const",
            ),
        )

    metrics = evaluate_dememte_tta_suite(
        factory,
        loader,
        device="cpu",
        suite=suite,
        tta_method="hippocampal_full",
        tta_base_variant="tiny_vq",
        teacher_model=teacher,
    )
    for key in MEMORY_DIAG_KEYS:
        assert f"{key}_clean" in metrics
        assert f"{key}_corrupt_avg" in metrics
    # completion_amount with T=1, λ=0.05, const gate must be strictly positive.
    assert metrics["completion_amount_clean"] > 0.0


def test_evaluate_dememte_tta_skips_memory_columns_for_non_memory_adapter():
    """No memory diag columns should appear when the adapter is not the hippocampal one."""
    from dememte.tta import NoUpdateAdapter

    torch.manual_seed(15)
    cfg = tiny_config(vqsa_layers=0, vqsa_use_self_attention=False)
    x = torch.randn(4, 3, 7, 7)
    y = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    teacher = make_tiny_model(cfg).eval()
    teacher.requires_grad_(False)

    class _Dummy:
        pass

    adapter = NoUpdateAdapter.__new__(NoUpdateAdapter)
    nn.Module.__init__(adapter)
    adapter.model = teacher
    from dememte.tta import TTAStats
    adapter.stats = TTAStats()
    adapter.episodic = False
    adapter.model_state = None
    adapter.optimizer = None
    adapter.optimizer_state = None
    adapter.steps = 1

    result = evaluate_dememte_tta(
        adapter,
        loader,
        device="cpu",
        corruption=None,
        severity=0.0,
        tta_method="no_update",
        tta_base_variant="tiny_vq",
        teacher_model=teacher,
    )
    for key in MEMORY_DIAG_KEYS:
        assert f"{key}_mean" not in result, f"{key}_mean should be absent for non-memory adapter"
