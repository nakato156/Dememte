"""Training loops for strict DeMemte VQSA and ResNet baselines."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .corruptions import apply_train_corruption
from .models.dememte import DeMemteVQSA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def _trainable_params(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]


def _vqsa_align_mode(config) -> str:
    return str(getattr(config, "vqsa_align_mode", "none"))


def _validate_vqsa_alignment_config(config) -> None:
    mode = _vqsa_align_mode(config)
    quantizer_type = str(getattr(config, "quantizer_type", "vq"))
    if quantizer_type not in {"vq", "ema_vq", "simvq_linear", "fsq"}:
        raise ValueError("Unknown quantizer_type: {!r}. Choices: ['vq', 'ema_vq', 'simvq_linear', 'fsq']".format(quantizer_type))
    if mode not in {"none", "zq_mse"}:
        raise ValueError(f"Unknown VQSA alignment mode: {mode!r}. Choices: ['none', 'zq_mse']")
    if mode != "none" and not bool(getattr(config, "vqsa_use_codebook", True)):
        raise ValueError("VQSA alignment requires vqsa_use_codebook=True because it operates on zq.")


def _vqsa_alignment_loss(dbg: dict, clean_batch_size: int, train: bool, config) -> torch.Tensor:
    mode = _vqsa_align_mode(config)
    zq = dbg["zq"]
    if mode == "none" or not train:
        return zq.sum() * 0.0
    if mode == "zq_mse":
        expected = clean_batch_size * 2
        if zq.size(0) != expected:
            raise ValueError(
                f"zq_mse alignment expects a clean+corrupt mixed batch of size {expected}, got {zq.size(0)}."
            )
        zq_clean = zq[:clean_batch_size].detach()
        zq_corrupt = zq[clean_batch_size:]
        return F.mse_loss(zq_corrupt, zq_clean)
    raise ValueError(f"Unknown VQSA alignment mode: {mode!r}. Choices: ['none', 'zq_mse']")


# ---------------------------------------------------------------------------
# Strict DeMemte VQSA training
# ---------------------------------------------------------------------------

def configure_vqsa_training(model: DeMemteVQSA, config, train: bool = True) -> None:
    train_backbone = bool(getattr(config, "vqsa_train_backbone", False))
    model.set_backbone_trainable(train_backbone)
    model.train(train)
    if not train_backbone:
        model.backbone.eval()


def make_optimizer_vqsa(model: DeMemteVQSA, config):
    groups = []
    vqsa_params = _trainable_params(model.vqsa)
    if vqsa_params:
        groups.append({"params": vqsa_params, "lr": config.lr_vq})
    cls_params = _trainable_params(model.classifier)
    if cls_params:
        groups.append({"params": cls_params, "lr": config.lr_cls})
    bb_params = _trainable_params(model.backbone)
    if bb_params:
        groups.append({"params": bb_params, "lr": min(config.lr_cls, 1e-5)})
    return optim.AdamW(groups, weight_decay=config.weight_decay)


def _add_hard_usage(counts: torch.Tensor | None, dbg: dict) -> torch.Tensor | None:
    indices = dbg.get("encoding_indices")
    num_embeddings = int(dbg.get("num_embeddings", 0) or 0)
    if indices is None or num_embeddings <= 0:
        return counts
    batch_counts = torch.bincount(indices.detach().reshape(-1).long().cpu(), minlength=num_embeddings).float()
    if counts is None:
        return batch_counts
    if counts.numel() < batch_counts.numel():
        padded = torch.zeros_like(batch_counts)
        padded[: counts.numel()] = counts
        counts = padded
    counts[: batch_counts.numel()] += batch_counts
    return counts


def _hard_usage_metrics(counts: torch.Tensor | None) -> dict:
    if counts is None or counts.numel() == 0 or counts.sum().item() <= 0:
        return {"hard_usage": 0.0, "hard_perplexity": 0.0, "dead_code_fraction": 0.0}
    probs = counts / counts.sum().clamp_min(1.0)
    used = counts > 0
    entropy = -(probs[used] * probs[used].log()).sum()
    return {
        "hard_usage": used.float().mean().item(),
        "hard_perplexity": entropy.exp().item(),
        "dead_code_fraction": (~used).float().mean().item(),
    }


@torch.no_grad()
def initialize_vqsa_codebook(model, loader, config, device) -> bool:
    if not bool(getattr(config, "vq_kmeans_init", False)):
        return False
    init_fn = getattr(model.vqsa, "initialize_codebook_from_latents", None)
    if init_fn is None:
        return False
    was_training = model.training
    model.eval()
    try:
        x, _ = next(iter(loader))
    except StopIteration:
        model.train(was_training)
        return False
    x = x.to(device, non_blocking=True)
    _, z = model.encode_z(x)
    initialized = init_fn(z, steps=int(getattr(config, "vq_kmeans_steps", 10)))
    model.train(was_training)
    return bool(initialized)


def run_epoch_vqsa(model, loader, optimizer, train, config, device, criterion, epoch: int | None = None):
    _validate_vqsa_alignment_config(config)
    configure_vqsa_training(model, config, train=train)
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "vq": 0.0,
        "codebook": 0.0,
        "commitment": 0.0,
        "align": 0.0,
        "acc": 0.0,
        "n": 0,
    }
    hard_counts = None
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            x_dirty = apply_train_corruption(x.clone(), prob=config.train_corrupt_prob)
            x_in = torch.cat([x, x_dirty], dim=0)
            y_in = torch.cat([y, y], dim=0)
            optimizer.zero_grad(set_to_none=True)
        else:
            x_in, y_in = x, y

        logits, vq_loss, dbg = model(x_in, return_debug=True)
        ce_loss = criterion(logits, y_in)
        align_loss = _vqsa_alignment_loss(dbg, x.size(0), train, config)
        loss = ce_loss + config.vq_weight * vq_loss + getattr(config, "align_weight", 0.0) * align_loss

        if train:
            loss.backward()
            optimizer.step()
            if bool(getattr(config, "dead_code_restart", False)) and epoch is not None:
                restart_after = int(getattr(config, "dead_code_restart_after_epoch", 1))
                if epoch > restart_after:
                    model.vqsa.restart_dead_codes_from_latents(dbg["z"].detach())

        bs = y_in.size(0)
        totals["loss"] += loss.item() * bs
        totals["ce"] += ce_loss.item() * bs
        totals["vq"] += vq_loss.item() * bs
        totals["codebook"] += dbg["codebook_loss"].item() * bs
        totals["commitment"] += dbg["commitment_loss"].item() * bs
        totals["align"] += align_loss.item() * bs
        totals["acc"] += (logits.argmax(1) == y_in).float().mean().item() * bs
        totals["n"] += bs
        hard_counts = _add_hard_usage(hard_counts, dbg)

    for k in ("loss", "ce", "vq", "codebook", "commitment", "align", "acc"):
        totals[k] /= max(1, totals["n"])
    totals.update(_hard_usage_metrics(hard_counts))
    return totals


def train_dememte_vqsa(model, train_loader, val_loader, config, device, criterion=None, verbose=True):
    """Single strict VQSA schedule: clean+corrupt mixed batches, early stop on clean val accuracy."""
    criterion = criterion or nn.CrossEntropyLoss()
    configure_vqsa_training(model, config, train=True)
    initialize_vqsa_codebook(model, train_loader, config, device)
    opt = make_optimizer_vqsa(model, config)
    sch = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    best_acc, best_state, no_imp = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(1, config.epochs_vqsa_max + 1):
        tr = run_epoch_vqsa(model, train_loader, opt, True, config, device, criterion, epoch=ep)
        va = run_epoch_vqsa(model, val_loader, opt, False, config, device, criterion, epoch=ep)
        sch.step(va["acc"])
        if verbose:
            print(
                f"[VQSA {ep:02d}] "
                f"tr_loss={tr['loss']:.4f} tr_acc={tr['acc']:.4f} "
                f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f} "
                f"val_vq={va['vq']:.4f} tr_align={tr['align']:.4f} "
                f"val_usage={va['hard_usage']:.4f} val_hard_ppl={va['hard_perplexity']:.2f}"
            )
        if va["acc"] > best_acc + config.early_stop_min_delta:
            best_acc, best_state, no_imp = va["acc"], copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= config.early_stop_patience:
                if verbose:
                    print(f"VQSA early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model, best_acc


def train_dememte_full(model, train_loader, val_loader, config, device, verbose=True):
    return train_dememte_vqsa(model, train_loader, val_loader, config, device, verbose=verbose)


# ---------------------------------------------------------------------------
# ResNet baseline phased training
# ---------------------------------------------------------------------------

def _baseline_epoch(model, loader, optimizer, train, device, criterion, corrupt_prob: float):
    model.train(train)
    totals = {"loss": 0.0, "acc": 0.0, "n": 0}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train and corrupt_prob > 0:
            x = apply_train_corruption(x.clone(), prob=corrupt_prob)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        totals["loss"] += loss.item() * bs
        totals["acc"] += (logits.argmax(1) == y).float().mean().item() * bs
        totals["n"] += bs
    for k in ("loss", "acc"):
        totals[k] /= max(1, totals["n"])
    return totals


def _run_baseline_stage(model, train_loader, val_loader, optimizer, scheduler, device, criterion, max_epochs, corrupt_prob, patience, min_delta, stage_label, verbose):
    best_acc, best_state, no_imp = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(1, max_epochs + 1):
        tr = _baseline_epoch(model, train_loader, optimizer, True, device, criterion, corrupt_prob)
        va = _baseline_epoch(model, val_loader, optimizer, False, device, criterion, 0.0)
        scheduler.step(va["acc"])
        if verbose:
            print(f"[{stage_label} {ep:02d}] tr_loss={tr['loss']:.4f} tr_acc={tr['acc']:.4f} val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}")
        if va["acc"] > best_acc + min_delta:
            best_acc, best_state, no_imp = va["acc"], copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                if verbose:
                    print(f"{stage_label} early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return best_acc


def train_baseline_phased(model, train_loader, val_loader, config, device, criterion=None, verbose=True):
    """3-stage schedule for ResNetBaseline: warmup clean, corruption augmentation, joint refine."""
    criterion = criterion or nn.CrossEntropyLoss()

    if config.freeze_backbone:
        params = list(model.classifier.parameters())
        opt = optim.AdamW(params, lr=config.lr_cls, weight_decay=config.weight_decay)
    else:
        opt = optim.AdamW([
            {"params": model.backbone.parameters(), "lr": config.backbone_lr},
            {"params": model.classifier.parameters(), "lr": config.lr_cls},
        ], weight_decay=config.weight_decay)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=config.scheduler_factor, patience=config.scheduler_patience)
    _run_baseline_stage(model, train_loader, val_loader, opt, sch, device, criterion, config.epochs_warmup, 0.0, config.early_stop_patience, config.early_stop_min_delta, "BASE-warmup", verbose)

    if config.freeze_backbone:
        opt = optim.AdamW(list(model.classifier.parameters()), lr=config.lr_cls, weight_decay=config.weight_decay)
    else:
        opt = optim.AdamW([
            {"params": model.backbone.parameters(), "lr": config.backbone_lr},
            {"params": model.classifier.parameters(), "lr": config.lr_cls},
        ], weight_decay=config.weight_decay)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=config.scheduler_factor, patience=config.scheduler_patience)
    _run_baseline_stage(model, train_loader, val_loader, opt, sch, device, criterion, config.epochs_corrupt, config.train_corrupt_prob, config.early_stop_patience, config.early_stop_min_delta, "BASE-corrupt", verbose)

    if config.freeze_backbone:
        opt = optim.AdamW(list(model.classifier.parameters()), lr=config.lr_cls * 0.5, weight_decay=config.weight_decay)
    else:
        opt = optim.AdamW([
            {"params": model.backbone.parameters(), "lr": config.backbone_lr * 0.5},
            {"params": model.classifier.parameters(), "lr": config.lr_cls * 0.5},
        ], weight_decay=config.weight_decay)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=config.scheduler_factor, patience=config.scheduler_patience)
    best_acc = _run_baseline_stage(model, train_loader, val_loader, opt, sch, device, criterion, config.epochs_joint, config.train_corrupt_prob, config.early_stop_patience, config.early_stop_min_delta, "BASE-joint", verbose)
    return model, best_acc
