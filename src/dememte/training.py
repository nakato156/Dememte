"""Training loops for DeMemte phases 1/2/3 and the ResNet baseline.

Mirrors the phase configuration + optimizer construction + epoch loops from
`Dememte_e5y.ipynb`. The baseline notebook reuses `train_baseline_phased` which
applies an analogous 3-stage schedule (warmup, corruption augmentation, joint
refine) to a plain `ResNetBaseline` so the comparison is fair.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .corruptions import apply_train_corruption
from .models.attractor import gate_entropy_regularizer
from .models.dememte import DeMemteAttractor, reset_gate_calibration_from_config
from .models.vq import sigreg_latent_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def _trainable_params(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]


def _temporarily_set_requires_grad(modules, requires_grad, eval_mode=False):
    previous = []
    for module in modules:
        module_prev = [p.requires_grad for p in module.parameters()]
        previous.append((module, module_prev, module.training))
        set_requires_grad(module, requires_grad)
        if eval_mode:
            module.eval()
    return previous


def _restore_requires_grad(previous):
    for module, module_prev, was_training in previous:
        for p, req in zip(module.parameters(), module_prev):
            p.requires_grad = req
        module.train(was_training)


# ---------------------------------------------------------------------------
# DeMemte phase configuration
# ---------------------------------------------------------------------------

def configure_phase1(model: DeMemteAttractor) -> None:
    model.set_backbone_trainable(False)
    set_requires_grad(model.projector, True)
    set_requires_grad(model.vq, True)
    set_requires_grad(model.unprojector, True)
    set_requires_grad(model.attractor, False)
    set_requires_grad(model.gate, False)
    set_requires_grad(model.aux_classifier, False)
    set_requires_grad(model.classifier, False)


def configure_phase2(model: DeMemteAttractor) -> None:
    model.set_backbone_trainable(False)
    set_requires_grad(model.projector, False)
    set_requires_grad(model.vq, False)
    set_requires_grad(model.unprojector, False)
    model.projector.eval()
    model.vq.eval()
    model.unprojector.eval()
    set_requires_grad(model.attractor, True)
    set_requires_grad(model.gate, True)
    set_requires_grad(model.aux_classifier, True)
    set_requires_grad(model.classifier, True)


def configure_phase3(model: DeMemteAttractor, config) -> None:
    model.set_backbone_trainable(False)
    if getattr(config, "phase3_backbone_train_mode", "frozen") == "partial_unfreeze":
        # layer4 of ResNet18 (Sequential[-1]) becomes trainable
        for p in model.backbone[-1].parameters():
            p.requires_grad = True
    memory_trainable = config.phase3_memory_grad_mode != "freeze_vq"
    set_requires_grad(model.projector, memory_trainable)
    set_requires_grad(model.vq, memory_trainable)
    set_requires_grad(model.unprojector, memory_trainable)
    if not memory_trainable:
        model.projector.eval()
        model.vq.eval()
        model.unprojector.eval()
    set_requires_grad(model.attractor, True)
    set_requires_grad(model.gate, True)
    if config.phase3_lock_familiarity:
        model.gate.midpoint.requires_grad = False
        model.gate.log_width.requires_grad = False
    set_requires_grad(model.aux_classifier, True)
    set_requires_grad(model.classifier, True)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def make_optimizer_phase1(model, config):
    params = list(model.projector.parameters()) + list(model.vq.parameters()) + list(model.unprojector.parameters())
    return optim.AdamW(params, lr=config.lr_vq, weight_decay=config.weight_decay)


def make_optimizer_phase2(model, config):
    return optim.AdamW([
        {"params": model.attractor.parameters(), "lr": config.lr_attractor},
        {"params": model.gate.parameters(), "lr": config.lr_gate},
        {"params": model.aux_classifier.parameters(), "lr": config.lr_cls},
        {"params": model.classifier.parameters(), "lr": config.lr_cls},
    ], weight_decay=config.weight_decay)


def make_optimizer_phase3(model, config):
    groups = []
    mem_params = _trainable_params(model.projector) + _trainable_params(model.vq) + _trainable_params(model.unprojector)
    if mem_params:
        groups.append({"params": mem_params, "lr": config.lr_vq})
    groups.append({"params": _trainable_params(model.attractor), "lr": config.lr_attractor})
    groups.append({"params": _trainable_params(model.gate), "lr": config.lr_gate})
    groups.append({"params": _trainable_params(model.aux_classifier) + _trainable_params(model.classifier), "lr": config.lr_cls})
    bb_params = _trainable_params(model.backbone)
    if bb_params:
        groups.append({"params": bb_params, "lr": min(config.lr_cls, 1e-5)})
    return optim.AdamW(groups, weight_decay=config.weight_decay)


# ---------------------------------------------------------------------------
# DeMemte phase loops
# ---------------------------------------------------------------------------

def run_epoch_phase1(model, loader, optimizer, train, config, device):
    model.train(train)
    configure_phase1(model)
    totals = {"loss": 0.0, "recon": 0.0, "vq": 0.0, "sigreg": 0.0, "n": 0}
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        recon_loss, vq_loss, sigreg_loss = model.pretrain_latent(
            x,
            feature_mask_ratio=config.masked_feature_ratio,
            sigreg_sketch_dim=config.weak_sigreg_sketch_dim,
        )
        loss = 0.5 * recon_loss + 0.25 * vq_loss + config.weak_sigreg_weight * sigreg_loss
        if train:
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        totals["loss"] += loss.item() * bs
        totals["recon"] += recon_loss.item() * bs
        totals["vq"] += vq_loss.item() * bs
        totals["sigreg"] += sigreg_loss.item() * bs
        totals["n"] += bs
    for k in ("loss", "recon", "vq", "sigreg"):
        totals[k] /= max(1, totals["n"])
    return totals


def train_phase1(model, train_loader, val_loader, config, device, verbose=True):
    opt = make_optimizer_phase1(model, config)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience)
    best_loss, best_state, no_imp = float("inf"), copy.deepcopy(model.state_dict()), 0
    for ep in range(1, config.epochs_phase1_max + 1):
        tr = run_epoch_phase1(model, train_loader, opt, True, config, device)
        va = run_epoch_phase1(model, val_loader, opt, False, config, device)
        sch.step(va["loss"])
        if verbose:
            print(f"[P1 {ep:02d}] tr_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_recon={va['recon']:.4f} val_vq={va['vq']:.4f}")
        if va["loss"] < best_loss - config.early_stop_min_delta:
            best_loss, best_state, no_imp = va["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= config.early_stop_patience:
                if verbose:
                    print(f"P1 early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model, best_loss


def run_epoch_phase2(model, loader, optimizer, train, config, device, criterion):
    model.train(train)
    configure_phase2(model)
    totals = {"loss": 0.0, "mse_dirty": 0.0, "mse_clean": 0.0, "sigreg": 0.0, "acc": 0.0, "gate": 0.0, "gate_raw": 0.0, "gate_prior": 0.0, "n": 0}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_dirty = apply_train_corruption(x.clone(), prob=config.train_corrupt_prob) if train else x

        with torch.no_grad():
            _, z_clean = model.encode_z(x)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits_clean, mse_clean, _, dbg_clean = model(x, target_z=z_clean, update_ema=True, return_debug=True)
        logits_dirty, mse_dirty, _, dbg_dirty = model(x_dirty, target_z=z_clean, update_ema=False, return_debug=True)
        ce_clean = criterion(logits_clean, y)
        entropy_reg = 0.5 * (gate_entropy_regularizer(dbg_clean["gate"]) + gate_entropy_regularizer(dbg_dirty["gate"]))
        raw_entropy_reg = 0.5 * (gate_entropy_regularizer(dbg_clean["gate_raw"]) + gate_entropy_regularizer(dbg_dirty["gate_raw"]))
        sigreg_loss = 0.5 * (
            sigreg_latent_loss(dbg_clean["z_completed"], config.weak_sigreg_sketch_dim)
            + sigreg_latent_loss(dbg_dirty["z_completed"], config.weak_sigreg_sketch_dim)
        )
        loss = mse_dirty + 0.3 * mse_clean + ce_clean
        loss = loss + config.gate_entropy_reg * entropy_reg + config.gate_raw_entropy_reg * raw_entropy_reg
        loss = loss + config.weak_sigreg_weight * sigreg_loss

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        totals["loss"] += loss.item() * bs
        totals["mse_dirty"] += mse_dirty.item() * bs
        totals["mse_clean"] += mse_clean.item() * bs
        totals["sigreg"] += sigreg_loss.item() * bs
        totals["acc"] += (logits_clean.argmax(1) == y).float().mean().item() * bs
        totals["gate"] += dbg_dirty["gate"].mean().item() * bs
        totals["gate_raw"] += dbg_dirty["gate_raw"].mean().item() * bs
        totals["gate_prior"] += dbg_dirty["gate_prior"].mean().item() * bs
        totals["n"] += bs
    for k in ("loss", "mse_dirty", "mse_clean", "sigreg", "acc", "gate", "gate_raw", "gate_prior"):
        totals[k] /= max(1, totals["n"])
    return totals


def train_phase2(model, train_loader, val_loader, config, device, criterion=None, verbose=True):
    criterion = criterion or nn.CrossEntropyLoss()
    opt = make_optimizer_phase2(model, config)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience)
    best_loss, best_state, no_imp = float("inf"), copy.deepcopy(model.state_dict()), 0
    for ep in range(1, config.epochs_phase2_max + 1):
        tr = run_epoch_phase2(model, train_loader, opt, True, config, device, criterion)
        va = run_epoch_phase2(model, val_loader, opt, False, config, device, criterion)
        sch.step(va["loss"])
        if verbose:
            print(f"[P2 {ep:02d}] tr_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_gate={va['gate']:.4f} val_raw={va['gate_raw']:.4f}")
        if va["loss"] < best_loss - config.early_stop_min_delta:
            best_loss, best_state, no_imp = va["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= config.early_stop_patience:
                if verbose:
                    print(f"P2 early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model, best_loss


def _antipareidolia_loss(dbg_dirty, logits_dirty, y):
    gate = dbg_dirty["gate"]
    familiarity = dbg_dirty["familiarity"]
    anti_ood = (gate * (1.0 - familiarity).detach()).mean()
    base_pred = dbg_dirty["logits_base"].detach().argmax(1)
    final_pred = logits_dirty.detach().argmax(1)
    harmful = ((base_pred == y) & (final_pred != y)).float().view(-1, 1, 1, 1)
    harmful_gate = (gate * harmful).mean()
    return anti_ood + harmful_gate


def run_epoch_phase3(model, loader, optimizer, train, config, device, criterion, antipareidolia_weight):
    model.train(train)
    configure_phase3(model, config)
    totals = {"loss": 0.0, "sigreg": 0.0, "acc": 0.0, "gate": 0.0, "gate_raw": 0.0, "gate_prior": 0.0, "n": 0}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_dirty = apply_train_corruption(x.clone(), prob=config.train_corrupt_prob) if train else x

        with torch.no_grad():
            _, z_clean_target = model.encode_z(x)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits_clean, mse_clean, vq_clean, dbg_clean = model(x, target_z=z_clean_target, update_ema=True, return_debug=True)
        if train and config.phase3_memory_grad_mode == "vq_clean_only":
            prev = _temporarily_set_requires_grad([model.projector, model.vq, model.unprojector], False, eval_mode=True)
            logits_dirty, mse_dirty, vq_dirty, dbg_dirty = model(x_dirty, target_z=z_clean_target, update_ema=False, return_debug=True)
            _restore_requires_grad(prev)
        else:
            logits_dirty, mse_dirty, vq_dirty, dbg_dirty = model(x_dirty, target_z=z_clean_target, update_ema=False, return_debug=True)

        ce_clean = criterion(logits_clean, y)
        ce_dirty = criterion(logits_dirty, y)
        anti = _antipareidolia_loss(dbg_dirty, logits_dirty, y) if antipareidolia_weight > 0 else torch.zeros((), device=x.device)
        entropy_reg = 0.5 * (gate_entropy_regularizer(dbg_clean["gate"]) + gate_entropy_regularizer(dbg_dirty["gate"]))
        raw_entropy_reg = 0.5 * (gate_entropy_regularizer(dbg_clean["gate_raw"]) + gate_entropy_regularizer(dbg_dirty["gate_raw"]))

        loss = 0.5 * (ce_clean + ce_dirty)
        loss = loss + 0.5 * config.denoise_weight * (mse_clean + mse_dirty)
        loss = loss + 0.5 * config.vq_weight * (vq_clean + vq_dirty)
        loss = loss + antipareidolia_weight * anti
        loss = loss + config.gate_entropy_reg * entropy_reg + config.gate_raw_entropy_reg * raw_entropy_reg
        sigreg_loss = 0.25 * (
            sigreg_latent_loss(dbg_clean["z"], config.weak_sigreg_sketch_dim)
            + sigreg_latent_loss(dbg_dirty["z"], config.weak_sigreg_sketch_dim)
            + sigreg_latent_loss(dbg_clean["z_completed"], config.weak_sigreg_sketch_dim)
            + sigreg_latent_loss(dbg_dirty["z_completed"], config.weak_sigreg_sketch_dim)
        )
        loss = loss + config.weak_sigreg_weight * sigreg_loss

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        avg_acc = 0.5 * ((logits_clean.argmax(1) == y).float().mean().item() + (logits_dirty.argmax(1) == y).float().mean().item())
        totals["loss"] += loss.item() * bs
        totals["sigreg"] += sigreg_loss.item() * bs
        totals["acc"] += avg_acc * bs
        totals["gate"] += dbg_dirty["gate"].mean().item() * bs
        totals["gate_raw"] += dbg_dirty["gate_raw"].mean().item() * bs
        totals["gate_prior"] += dbg_dirty["gate_prior"].mean().item() * bs
        totals["n"] += bs
    for k in ("loss", "sigreg", "acc", "gate", "gate_raw", "gate_prior"):
        totals[k] /= max(1, totals["n"])
    return totals


def train_phase3(model, train_loader, val_loader, config, device, criterion=None, verbose=True):
    criterion = criterion or nn.CrossEntropyLoss()
    if config.phase3_lock_familiarity:
        reset_gate_calibration_from_config(model, config)
    configure_phase3(model, config)
    opt = make_optimizer_phase3(model, config)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=config.scheduler_factor, patience=config.scheduler_patience)
    best_acc, best_state, no_imp = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(1, config.epochs_phase3_max + 1):
        tr = run_epoch_phase3(model, train_loader, opt, True, config, device, criterion, config.antipareidolia_weight)
        va = run_epoch_phase3(model, val_loader, opt, False, config, device, criterion, config.antipareidolia_weight)
        sch.step(va["acc"])
        if verbose:
            print(f"[P3 {ep:02d}] tr_acc={tr['acc']:.4f} val_acc={va['acc']:.4f} val_gate={va['gate']:.4f} val_raw={va['gate_raw']:.4f}")
        if va["acc"] > best_acc + config.early_stop_min_delta:
            best_acc, best_state, no_imp = va["acc"], copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= config.early_stop_patience:
                if verbose:
                    print(f"P3 early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model, best_acc


def train_dememte_full(model, train_loader, val_loader, config, device, verbose=True):
    """Convenience driver: run phases 1, 2, 3 in order."""
    train_phase1(model, train_loader, val_loader, config, device, verbose=verbose)
    train_phase2(model, train_loader, val_loader, config, device, verbose=verbose)
    return train_phase3(model, train_loader, val_loader, config, device, verbose=verbose)


# ---------------------------------------------------------------------------
# ResNet baseline phased training (mirrors phase methodology for fair comparison)
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
    """3-stage schedule for ResNetBaseline: warmup (clean), corrupt aug, joint refine.

    Mirrors the methodology used by DeMemte phases 1-3 so the comparison is fair.
    """
    criterion = criterion or nn.CrossEntropyLoss()

    # Stage 1: warmup classifier (and backbone if unfrozen) on clean data
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

    # Stage 2: corruption augmentation
    if config.freeze_backbone:
        opt = optim.AdamW(list(model.classifier.parameters()), lr=config.lr_cls, weight_decay=config.weight_decay)
    else:
        opt = optim.AdamW([
            {"params": model.backbone.parameters(), "lr": config.backbone_lr},
            {"params": model.classifier.parameters(), "lr": config.lr_cls},
        ], weight_decay=config.weight_decay)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=config.scheduler_factor, patience=config.scheduler_patience)
    _run_baseline_stage(model, train_loader, val_loader, opt, sch, device, criterion, config.epochs_corrupt, config.train_corrupt_prob, config.early_stop_patience, config.early_stop_min_delta, "BASE-corrupt", verbose)

    # Stage 3: joint refine with lower LR
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
