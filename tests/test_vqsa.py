import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dememte.config import E5Config
from dememte.evaluation import evaluate_dememte_suite
from dememte.io import load_checkpoint, save_checkpoint
from dememte.models import DeMemteVQSA, VectorQuantizer2D
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

    ckpt = tmp_path / "vqsa.pt"
    save_checkpoint(model, ckpt, extra={"best_val": totals["acc"]})
    reloaded = make_tiny_model(cfg)
    payload = load_checkpoint(reloaded, ckpt, device="cpu", strict=True)
    assert "best_val" in payload


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
    assert all("gate" not in key for key in metrics)
