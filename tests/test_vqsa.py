import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dememte.config import E5Config, e6_config
from dememte.evaluation import evaluate_dememte_suite, evaluate_dememte_tta_suite
from dememte.io import load_checkpoint, save_checkpoint
from dememte.models import DeMemteVQSA, EMAVectorQuantizer2D, FSQQuantizer2D, SimVQLinearQuantizer2D, VectorQuantizer2D
from dememte.tta import (
    AlphaBNStatsAdapter,
    CodebookLossAdapter,
    EATALiteAdapter,
    MemoryTentAdapter,
    NoUpdateAdapter,
    SoftAssignTentAdapter,
    SourceFilterEATAAdapter,
    TentAdapter,
    collect_tta_bn_params,
    collect_tta_codebook_params,
    collect_tta_ln_params,
    configure_tta_codebook,
    configure_tta_layernorm,
    configure_tta_model,
    latent_memory_loss,
    make_tta_optimizer,
    soft_assign_entropy,
)
from dememte.training import make_optimizer_vqsa, run_epoch_vqsa


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


def test_vector_quantizer_outputs_and_soft_assign():
    quantizer = VectorQuantizer2D(num_embeddings=32, embedding_dim=16)
    z = torch.randn(2, 16, 7, 7)
    zq, vq_loss, codebook_loss, commitment_loss, dq_map, soft_assign = quantizer(z)

    assert zq.shape == z.shape
    assert dq_map.shape == (2, 1, 7, 7)
    assert soft_assign.shape == (2, 7, 7, 32)
    assert torch.allclose(soft_assign.sum(dim=-1), torch.ones(2, 7, 7), atol=1e-5)
    for loss in (vq_loss, codebook_loss, commitment_loss):
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_vector_quantizer_can_return_hard_indices():
    quantizer = VectorQuantizer2D(num_embeddings=32, embedding_dim=16, return_indices=True)
    z = torch.randn(2, 16, 7, 7)
    *_, indices = quantizer(z)

    assert indices.shape == (2, 7, 7)
    assert indices.min() >= 0
    assert indices.max() < 32


def test_ema_quantizer_kmeans_update_and_dead_restart():
    quantizer = EMAVectorQuantizer2D(
        num_embeddings=8,
        embedding_dim=4,
        decay=0.5,
        dead_code_threshold=0.5,
        return_indices=True,
    )
    z = torch.randn(2, 4, 3, 3)
    quantizer.initialize_from_data(z, steps=2)
    assert quantizer.usage_ema.sum().item() == pytest.approx(8.0)

    quantizer.train()
    zq, vq_loss, *_rest, indices = quantizer(z)

    assert zq.shape == z.shape
    assert torch.isfinite(vq_loss)
    assert indices.shape == (2, 3, 3)
    assert quantizer.usage_ema.sum().item() > 8.0

    quantizer.usage_ema.zero_()
    flat = z.permute(0, 2, 3, 1).reshape(-1, 4)
    quantizer.restart_dead_codes(flat)
    assert torch.all(quantizer.usage_ema >= 0.5)


def test_simvq_linear_backward_reaches_global_transform():
    quantizer = SimVQLinearQuantizer2D(num_embeddings=8, embedding_dim=4, return_indices=True)
    z = torch.randn(2, 4, 3, 3, requires_grad=True)
    zq, vq_loss, *_ = quantizer(z)
    loss = zq.mean() + vq_loss
    loss.backward()

    assert z.grad is not None
    assert quantizer.codebook_transform.weight.grad is not None


def test_fsq_quantizer_forward_backward_and_scalar_indices():
    quantizer = FSQQuantizer2D(num_embeddings=32, embedding_dim=4, levels=6, return_indices=True)
    z = torch.randn(2, 4, 3, 3, requires_grad=True)
    zq, vq_loss, codebook_loss, commitment_loss, dq_map, soft_assign, indices = quantizer(z)
    loss = zq.mean() + vq_loss
    loss.backward()

    assert zq.shape == z.shape
    assert dq_map.shape == (2, 1, 3, 3)
    assert soft_assign is None
    assert indices.shape == (2, 3, 3, 4)
    assert indices.min() >= 0
    assert indices.max() < 6
    assert codebook_loss.item() == 0.0
    assert torch.isfinite(commitment_loss)
    assert z.grad is not None


def test_dememte_vqsa_forward_debug_and_backward():
    cfg = tiny_config()
    model = make_tiny_model(cfg)
    x = torch.randn(2, 3, 7, 7)
    y = torch.tensor([0, 1])

    logits, vq_loss, dbg = model(x, return_debug=True)
    loss = nn.CrossEntropyLoss()(logits, y) + vq_loss
    loss.backward()

    assert logits.shape == (2, cfg.num_classes)
    assert dbg["z"].shape == (2, cfg.latent_dim, 7, 7)
    assert dbg["zq"].shape == dbg["z"].shape
    assert dbg["encoding_indices"].shape == (2, 7, 7)
    assert dbg["attention_weights"].shape[:3] == (2, cfg.vqsa_layers, cfg.vqsa_heads)
    assert model.projector.net[0].weight.grad is not None
    assert model.vq.embedding.weight.grad is not None
    assert model.vqsa.attention[0].attn.in_proj_weight.grad is not None
    assert model.classifier[-1].weight.grad is not None


def test_vqsa_training_smoke_and_checkpoint(tmp_path):
    cfg = tiny_config()
    model = make_tiny_model(cfg)
    x = torch.randn(4, 3, 7, 7)
    y = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    optimizer = make_optimizer_vqsa(model, cfg)

    totals = run_epoch_vqsa(model, loader, optimizer, True, cfg, "cpu", nn.CrossEntropyLoss())
    assert torch.isfinite(torch.tensor(totals["loss"]))
    assert "align" in totals
    assert "hard_usage" in totals
    assert "hard_perplexity" in totals
    assert "dead_code_fraction" in totals
    assert totals["align"] == 0.0

    ckpt = tmp_path / "vqsa.pt"
    save_checkpoint(model, ckpt, extra={"best_val": totals["acc"]})
    reloaded = make_tiny_model(cfg)
    payload = load_checkpoint(reloaded, ckpt, device="cpu", strict=True)
    assert "best_val" in payload


def test_e6_configs_select_paper_faithful_and_zq_alignment():
    paper = e6_config("e6_paper_faithful")
    aligned = e6_config("e6_zq_align_mse")
    ema = e6_config("e6_ema_kmeans_restart")
    winner = e6_config("e6_winner")
    simvq = e6_config("e6_simvq_linear")
    fsq = e6_config("e6_fsq")

    assert paper.variant_name == "e6_paper_faithful"
    assert paper.vqsa_align_mode == "none"
    assert paper.align_weight == 0.0
    assert paper.quantizer_type == "vq"

    assert aligned.variant_name == "e6_zq_align_mse"
    assert aligned.vqsa_align_mode == "zq_mse"
    assert aligned.align_weight == 0.1
    assert aligned.quantizer_type == "vq"

    assert ema.quantizer_type == "ema_vq"
    assert ema.vq_kmeans_init is True
    assert ema.dead_code_restart is True

    assert winner.variant_name == "e6_winner"
    assert winner.quantizer_type == ema.quantizer_type
    assert winner.vq_kmeans_init == ema.vq_kmeans_init
    assert winner.dead_code_restart == ema.dead_code_restart
    assert winner.vqsa_align_mode == ema.vqsa_align_mode

    assert simvq.quantizer_type == "simvq_linear"

    assert fsq.quantizer_type == "fsq"
    assert fsq.fsq_levels == 8


def test_vqsa_zq_alignment_training_smoke():
    cfg = tiny_config(vqsa_align_mode="zq_mse", align_weight=0.1)
    model = make_tiny_model(cfg)
    x = torch.randn(4, 3, 7, 7)
    y = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    optimizer = make_optimizer_vqsa(model, cfg)

    totals = run_epoch_vqsa(model, loader, optimizer, True, cfg, "cpu", nn.CrossEntropyLoss())

    assert torch.isfinite(torch.tensor(totals["loss"]))
    assert torch.isfinite(torch.tensor(totals["align"]))
    assert totals["align"] >= 0.0


def test_vqsa_training_smoke_with_new_quantizer_variants():
    for quantizer_type in ("ema_vq", "simvq_linear", "fsq"):
        cfg = tiny_config(
            quantizer_type=quantizer_type,
            vq_kmeans_init=(quantizer_type == "ema_vq"),
            dead_code_restart=(quantizer_type == "ema_vq"),
            dead_code_restart_after_epoch=0,
            fsq_levels=6,
        )
        model = make_tiny_model(cfg)
        x = torch.randn(4, 3, 7, 7)
        y = torch.tensor([0, 1, 2, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        optimizer = make_optimizer_vqsa(model, cfg)

        totals = run_epoch_vqsa(model, loader, optimizer, True, cfg, "cpu", nn.CrossEntropyLoss(), epoch=1)

        assert torch.isfinite(torch.tensor(totals["loss"]))
        assert 0.0 <= totals["hard_usage"] <= 1.0
        assert totals["hard_perplexity"] >= 0.0


def test_vqsa_zq_alignment_requires_codebook():
    cfg = tiny_config(vqsa_use_codebook=False, vqsa_align_mode="zq_mse", align_weight=0.1)
    model = make_tiny_model(cfg)
    x = torch.randn(2, 3, 7, 7)
    y = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    optimizer = make_optimizer_vqsa(model, cfg)

    with pytest.raises(ValueError, match="requires vqsa_use_codebook=True"):
        run_epoch_vqsa(model, loader, optimizer, True, cfg, "cpu", nn.CrossEntropyLoss())


def test_evaluate_dememte_suite_has_vqsa_metrics_without_gate_keys():
    cfg = tiny_config(vqsa_layers=0, vqsa_use_self_attention=False)
    model = make_tiny_model(cfg)
    x = torch.randn(4, 3, 7, 7)
    y = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    suite = {"gaussian_noise": [0.1]}

    metrics = evaluate_dememte_suite(model, loader, device="cpu", suite=suite)

    assert "clean_acc" in metrics
    assert "corrupt_acc_avg" in metrics
    assert "vq_loss_clean" in metrics
    assert "attention_entropy_clean" in metrics
    assert "hard_usage_clean" in metrics
    assert "hard_perplexity_clean" in metrics
    assert "dead_code_fraction_clean" in metrics
    assert all("gate" not in key for key in metrics)


def test_tta_collects_only_batchnorm_affine_params():
    cfg = tiny_config(quantizer_type="ema_vq")
    model = make_tiny_model(cfg)
    params, names = collect_tta_bn_params(model)

    assert params
    assert all(name.endswith((".weight", ".bias")) for name in names)
    assert all("projector.net.1" in name for name in names)
    assert all("classifier" not in name for name in names)
    assert all("attention" not in name for name in names)
    assert all("vq.embedding" not in name for name in names)


def test_configure_tta_model_keeps_vq_and_dropout_eval_but_bn_adaptable():
    cfg = tiny_config(quantizer_type="ema_vq", vqsa_dropout=0.2)
    model = make_tiny_model(cfg)
    configured = configure_tta_model(model)
    params, _ = collect_tta_bn_params(configured)

    assert configured.training is False
    assert configured.vq.training is False
    assert all(isinstance(m, nn.Dropout) and not m.training for m in configured.modules() if isinstance(m, nn.Dropout))
    assert all(p.requires_grad for p in params)
    assert any(not p.requires_grad for name, p in configured.named_parameters() if "classifier" in name)
    for module in configured.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.track_running_stats is False
            assert module.running_mean is None
            assert module.running_var is None


def test_tent_step_updates_bn_without_changing_ema_quantizer_state():
    cfg = tiny_config(quantizer_type="ema_vq", vqsa_dropout=0.0)
    model = configure_tta_model(make_tiny_model(cfg))
    params, _ = collect_tta_bn_params(model)
    optimizer = make_tta_optimizer(params, lr=1e-2, momentum=0.0)
    adapter = TentAdapter(model, optimizer)
    x = torch.randn(4, 3, 7, 7)
    bn_before = [p.detach().clone() for p in params]
    vq_before = {k: v.detach().clone() for k, v in model.vq.state_dict().items()}

    logits, dbg = adapter(x, return_debug=True)

    assert logits.shape == (4, cfg.num_classes)
    assert "zq" in dbg
    assert adapter.stats.updates == 1
    assert any(not torch.allclose(before, after.detach()) for before, after in zip(bn_before, params))
    for key, before in vq_before.items():
        assert torch.allclose(before, model.vq.state_dict()[key])


def test_eata_lite_with_impossible_margin_skips_update():
    cfg = tiny_config()
    model = configure_tta_model(make_tiny_model(cfg))
    params, _ = collect_tta_bn_params(model)
    optimizer = make_tta_optimizer(params, lr=1e-2, momentum=0.0)
    adapter = EATALiteAdapter(model, optimizer, num_classes=cfg.num_classes, e_margin=-1.0)
    x = torch.randn(4, 3, 7, 7)
    bn_before = [p.detach().clone() for p in params]

    adapter(x)

    assert adapter.stats.updates == 0
    assert adapter.stats.reliable == 0
    assert adapter.stats.selected == 0
    assert adapter.stats.seen == 4
    assert all(torch.allclose(before, after.detach()) for before, after in zip(bn_before, params))


def test_evaluate_dememte_tta_suite_reports_standard_and_tta_metrics():
    cfg = tiny_config(vqsa_layers=0, vqsa_use_self_attention=False)
    x = torch.randn(4, 3, 7, 7)
    y = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    suite = {"gaussian_noise": [0.1]}

    def factory():
        model = configure_tta_model(make_tiny_model(cfg))
        params, _ = collect_tta_bn_params(model)
        optimizer = make_tta_optimizer(params, lr=1e-3, momentum=0.0)
        return TentAdapter(model, optimizer)

    metrics = evaluate_dememte_tta_suite(
        factory,
        loader,
        device="cpu",
        suite=suite,
        tta_method="tent_bn",
        tta_base_variant="tiny",
    )

    assert "clean_acc" in metrics
    assert "corrupt_acc_avg" in metrics
    assert "vq_loss_clean" in metrics
    assert "tta_updates_clean" in metrics
    assert "tta_selection_rate_corrupt_avg" in metrics
    assert metrics["tta_method"] == "tent_bn"
    assert metrics["tta_base_variant"] == "tiny"


# --- E7b: LayerNorm adaptation + latent-memory preservation -----------------


def test_collect_tta_ln_params():
    cfg = tiny_config(quantizer_type="ema_vq")
    model = make_tiny_model(cfg)
    params, names = collect_tta_ln_params(model)

    assert params
    assert all(name.endswith((".weight", ".bias")) for name in names)
    assert all("attention" in name and ".norm" in name for name in names)
    assert all("classifier" not in name for name in names)
    assert all("projector" not in name for name in names)


def test_configure_tta_layernorm_preserves_bn_running_stats():
    cfg = tiny_config(quantizer_type="ema_vq", vqsa_dropout=0.2)
    model = make_tiny_model(cfg)
    configured = configure_tta_layernorm(model)
    ln_params, _ = collect_tta_ln_params(configured)

    assert configured.training is False
    assert configured.vq.training is False
    assert all(isinstance(m, nn.Dropout) and not m.training for m in configured.modules() if isinstance(m, nn.Dropout))
    # LayerNorm affine is trainable, classifier and BN are frozen.
    assert all(p.requires_grad for p in ln_params)
    assert any(not p.requires_grad for name, p in configured.named_parameters() if "classifier" in name)
    bn_params, _ = collect_tta_bn_params(configured)
    assert all(not p.requires_grad for p in bn_params)
    # BN keeps its source running statistics (the anti-collapse guarantee).
    for module in configured.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.track_running_stats is True
            assert module.running_mean is not None
            assert module.running_var is not None


def test_no_update_adapter_does_not_change_params():
    cfg = tiny_config(quantizer_type="ema_vq", vqsa_dropout=0.0)
    model = configure_tta_model(make_tiny_model(cfg))
    params, _ = collect_tta_bn_params(model)
    optimizer = make_tta_optimizer(params, lr=1e-2, momentum=0.0)
    adapter = NoUpdateAdapter(model, optimizer)
    x = torch.randn(4, 3, 7, 7)
    before = [p.detach().clone() for p in params]

    logits, dbg = adapter(x, return_debug=True)

    assert logits.shape == (4, cfg.num_classes)
    assert "zq" in dbg
    assert adapter.stats.updates == 0
    assert adapter.stats.seen == 4
    assert all(torch.allclose(b, a.detach()) for b, a in zip(before, params))


def test_latent_memory_loss_zero_when_identical():
    cfg = tiny_config(quantizer_type="ema_vq", vqsa_dropout=0.0)
    model = make_tiny_model(cfg).eval()
    x = torch.randn(4, 3, 7, 7)
    _, _, dbg = model(x, return_debug=True)

    loss = latent_memory_loss(dbg, dbg)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)

    # FSQ has soft_assign=None: the KL term must be skipped without error.
    fsq_model = make_tiny_model(tiny_config(quantizer_type="fsq")).eval()
    _, _, fsq_dbg = fsq_model(x, return_debug=True)
    assert fsq_dbg["soft_assign"] is None
    fsq_loss = latent_memory_loss(fsq_dbg, fsq_dbg)
    assert torch.allclose(fsq_loss, torch.zeros_like(fsq_loss), atol=1e-6)


def test_memory_tent_adapter_updates_only_layernorm():
    cfg = tiny_config(quantizer_type="ema_vq", vqsa_dropout=0.0)
    student = configure_tta_layernorm(make_tiny_model(cfg))
    teacher = make_tiny_model(cfg)
    ln_params, _ = collect_tta_ln_params(student)
    bn_params, _ = collect_tta_bn_params(student)
    optimizer = make_tta_optimizer(ln_params, lr=1e-1, momentum=0.0)
    adapter = MemoryTentAdapter(student, optimizer, source_model=teacher)
    x = torch.randn(4, 3, 7, 7)
    ln_before = [p.detach().clone() for p in ln_params]
    bn_before = [p.detach().clone() for p in bn_params]
    vq_before = {k: v.detach().clone() for k, v in student.vq.state_dict().items()}

    logits, dbg = adapter(x, return_debug=True)

    assert logits.shape == (4, cfg.num_classes)
    assert "z" in dbg and "zq" in dbg
    assert adapter.stats.updates == 1
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(ln_before, ln_params))
    assert all(torch.allclose(b, a.detach()) for b, a in zip(bn_before, bn_params))
    for key, before in vq_before.items():
        assert torch.allclose(before, student.vq.state_dict()[key])


def test_source_filter_eata_uses_teacher_logits():
    cfg = tiny_config(quantizer_type="ema_vq", vqsa_dropout=0.0)
    student = configure_tta_layernorm(make_tiny_model(cfg))
    teacher = make_tiny_model(cfg)
    ln_params, _ = collect_tta_ln_params(student)
    optimizer = make_tta_optimizer(ln_params, lr=1e-1, momentum=0.0)
    # Impossible entropy margin on the teacher => no sample passes the filter.
    adapter = SourceFilterEATAAdapter(
        student, optimizer, num_classes=cfg.num_classes, source_model=teacher, e_margin=-1.0
    )
    x = torch.randn(4, 3, 7, 7)
    ln_before = [p.detach().clone() for p in ln_params]

    adapter(x)

    assert adapter.stats.updates == 0
    assert adapter.stats.selected == 0
    assert adapter.stats.seen == 4
    assert all(torch.allclose(b, a.detach()) for b, a in zip(ln_before, ln_params))


# --- E7c-A: codebook plasticity (SimVQ) -------------------------------------


def test_collect_tta_codebook_params_returns_only_transform():
    cfg = tiny_config(quantizer_type="simvq_linear")
    model = make_tiny_model(cfg)
    params, names = collect_tta_codebook_params(model)

    assert len(params) == 1
    assert names == ["vqsa.vq.codebook_transform.weight"]
    assert params[0] is model.vq.codebook_transform.weight

    # Non-SimVQ quantizers have no learnable codebook_transform surface.
    for non_simvq in ("vq", "ema_vq", "fsq"):
        bad = make_tiny_model(tiny_config(quantizer_type=non_simvq))
        with pytest.raises(ValueError, match="simvq_linear"):
            collect_tta_codebook_params(bad)


def test_configure_tta_codebook_freezes_everything_else():
    cfg = tiny_config(quantizer_type="simvq_linear", vqsa_dropout=0.2)
    model = configure_tta_codebook(make_tiny_model(cfg))

    assert model.training is False
    assert all(isinstance(m, nn.Dropout) and not m.training for m in model.modules() if isinstance(m, nn.Dropout))
    # Only codebook_transform is trainable.
    cb_params, _ = collect_tta_codebook_params(model)
    assert all(p.requires_grad for p in cb_params)
    for name, p in model.named_parameters():
        if "codebook_transform" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad, f"expected {name} frozen"
    # BN keeps its source running statistics: nothing was touched.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.track_running_stats is True
            assert module.running_mean is not None
            assert module.running_var is not None


def test_tent_codebook_pure_is_structurally_inert():
    """Entropy on logits has *no graph at all* back to codebook_transform: the
    straight-through ``q_st = z + (q - z).detach()`` is the only path from the
    codebook to the classifier, and it severs the gradient. We confirm this by
    showing the entropy loss has no ``grad_fn`` and that the codebook weight is
    unchanged if a backward step is attempted (it would raise)."""
    cfg = tiny_config(quantizer_type="simvq_linear", vqsa_dropout=0.0)
    model = configure_tta_codebook(make_tiny_model(cfg))
    params, _ = collect_tta_codebook_params(model)
    weight_before = params[0].detach().clone()

    # Re-create the TENT loss path manually to inspect grad_fn:
    logits, _, _ = model(torch.randn(4, 3, 7, 7), return_debug=True)
    entropy = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1).mean()
    # codebook is the ONLY trainable param, so entropy must have requires_grad=False.
    assert not entropy.requires_grad, (
        "tent_codebook entropy must be detached from the codebook: straight-through "
        "blocks gradient and there are no other trainable params"
    )

    # Attempting an actual TentAdapter step raises because the loss has no grad_fn.
    optimizer = make_tta_optimizer(params, lr=1e-1, momentum=0.0)
    adapter = TentAdapter(model, optimizer)
    with pytest.raises(RuntimeError, match="does not require grad"):
        adapter(torch.randn(4, 3, 7, 7))

    # Weight is untouched.
    assert torch.allclose(weight_before, params[0].detach(), atol=1e-7)


def test_soft_assign_tent_adapter_moves_codebook_transform():
    """Entropy on soft_assign has live gradient to codebook_transform; weight must move."""
    cfg = tiny_config(quantizer_type="simvq_linear", vqsa_dropout=0.0)
    model = configure_tta_codebook(make_tiny_model(cfg))
    teacher = make_tiny_model(cfg)
    teacher.load_state_dict(model.state_dict())
    params, _ = collect_tta_codebook_params(model)
    optimizer = make_tta_optimizer(params, lr=1e-1, momentum=0.0)
    adapter = SoftAssignTentAdapter(model, optimizer)
    x = torch.randn(4, 3, 7, 7)
    weight_before = params[0].detach().clone()

    logits, dbg = adapter(x, return_debug=True)

    assert logits.shape == (4, cfg.num_classes)
    assert dbg["soft_assign"] is not None
    assert adapter.stats.updates == 1
    assert not torch.allclose(weight_before, params[0].detach(), atol=1e-6), (
        "soft-assign TENT should move codebook_transform.weight"
    )
    # zq drifts away from teacher because the codebook moved.
    with torch.no_grad():
        _, _, dbg_t = teacher(x, return_debug=True)
        zq_drift = (dbg["zq"].detach() - dbg_t["zq"]).pow(2).mean().sqrt().item()
    assert zq_drift > 1e-5


def test_codebook_loss_adapter_reduces_codebook_loss():
    cfg = tiny_config(quantizer_type="simvq_linear", vqsa_dropout=0.0)
    model = configure_tta_codebook(make_tiny_model(cfg))
    params, _ = collect_tta_codebook_params(model)
    optimizer = make_tta_optimizer(params, lr=1.0, momentum=0.0)
    adapter = CodebookLossAdapter(model, optimizer)
    x = torch.randn(4, 3, 7, 7)
    with torch.no_grad():
        _, _, dbg0 = model(x, return_debug=True)
        loss_before = float(dbg0["codebook_loss"].item())

    adapter(x)

    with torch.no_grad():
        _, _, dbg1 = model(x, return_debug=True)
        loss_after = float(dbg1["codebook_loss"].item())
    assert loss_after < loss_before, f"codebook_loss must drop: before={loss_before} after={loss_after}"
    assert adapter.stats.updates == 1


def test_alpha_bn_adapter_replaces_and_restores_projector_bn_forward():
    cfg = tiny_config(quantizer_type="simvq_linear", vqsa_dropout=0.0)
    model = make_tiny_model(cfg).eval()
    target = model.get_submodule("vqsa.projector.net.1")
    assert isinstance(target, nn.BatchNorm2d)
    # Capture a real BN input by hooking it once before the adapter is built.
    captured = {}

    def capture(_module, inputs, _output):
        captured["x"] = inputs[0].detach().clone()

    h = target.register_forward_hook(capture)
    with torch.no_grad():
        _ = model(torch.randn(2, 3, 7, 7))
    h.remove()
    bn_input = captured["x"]

    alpha = 0.25
    adapter = AlphaBNStatsAdapter(model, alpha=alpha)
    x_full = torch.randn(2, 3, 7, 7)
    with torch.no_grad():
        _ = adapter(x_full)
    # After running, the hook is still installed; compute expected alpha-mix manually.
    with torch.no_grad():
        running_mean, running_var = target.running_mean, target.running_var
        batch_mean = bn_input.mean(dim=(0, 2, 3))
        batch_var = bn_input.var(dim=(0, 2, 3), unbiased=False)
        mixed_mean = alpha * running_mean + (1.0 - alpha) * batch_mean
        mixed_var = alpha * running_var + (1.0 - alpha) * batch_var
        import torch.nn.functional as F
        expected = F.batch_norm(
            bn_input, mixed_mean, mixed_var, target.weight, target.bias,
            training=False, momentum=0.0, eps=target.eps,
        )
        actual = target(bn_input)
    assert torch.allclose(actual, expected, atol=1e-6)

    # close() removes the hook so the original BN forward is back.
    adapter.close()
    with torch.no_grad():
        restored = target(bn_input)
        original_expected = F.batch_norm(
            bn_input, target.running_mean, target.running_var, target.weight, target.bias,
            training=False, momentum=0.0, eps=target.eps,
        )
    assert torch.allclose(restored, original_expected, atol=1e-6)
    # The alpha-mixed output differs from the source BN (unless alpha happened to be 1.0).
    assert not torch.allclose(actual, restored, atol=1e-3)


def test_evaluate_dememte_tta_with_teacher_reports_drift_columns():
    cfg = tiny_config(quantizer_type="simvq_linear", vqsa_layers=0, vqsa_use_self_attention=False)
    x = torch.randn(4, 3, 7, 7)
    y = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    suite = {"gaussian_noise": [0.1]}

    teacher = make_tiny_model(cfg).eval()
    teacher.requires_grad_(False)

    def factory():
        model = configure_tta_codebook(make_tiny_model(cfg))
        model.load_state_dict(teacher.state_dict())
        params, _ = collect_tta_codebook_params(model)
        optimizer = make_tta_optimizer(params, lr=1e-3, momentum=0.0)
        return SoftAssignTentAdapter(model, optimizer)

    metrics = evaluate_dememte_tta_suite(
        factory,
        loader,
        device="cpu",
        suite=suite,
        tta_method="tent_codebook_softassign",
        tta_base_variant="tiny_simvq",
        teacher_model=teacher,
    )

    for key in ("z_drift", "zq_drift", "assignment_churn", "kl_assign_src"):
        assert f"{key}_clean" in metrics
        assert f"{key}_corrupt_avg" in metrics
    # Source-vs-source (clean, teacher loaded with same state) starts at zero drift
    # before any adapter step, but adaptation may shift soft-assign minutely. The
    # corruption drift should be non-negative and finite.
    assert metrics["zq_drift_corrupt_avg"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["kl_assign_src_corrupt_avg"]))
    assert "hard_usage_delta_vs_src_clean" in metrics
    assert "dead_code_fraction_delta_vs_src_corrupt_avg" in metrics
