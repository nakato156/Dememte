"""Evaluation: clean accuracy, paired corruption suite, and VQSA diagnostics."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .corruptions import STRICT_SUITE, apply_eval_corruption


VQSA_KEYS = [
    "vq_loss",
    "codebook_loss",
    "commitment_loss",
    "dq_mean",
    "assignment_entropy",
    "codebook_perplexity",
    "hard_usage",
    "hard_perplexity",
    "dead_code_fraction",
    "attention_entropy",
]


# E7c-A diagnostics that compare the in-flight student against a frozen source
# teacher per batch. ``z_drift`` / ``zq_drift`` are RMS distances per sample,
# ``assignment_churn`` is the fraction of token positions whose hard codebook
# index moved relative to the teacher, and ``kl_assign_src`` is the KL of the
# teacher soft assignment relative to the student soft assignment. All are zero
# for ``source`` (student == teacher) and grow with codebook plasticity.
TEACHER_DIAG_KEYS = [
    "z_drift",
    "zq_drift",
    "assignment_churn",
    "kl_assign_src",
]


def _trapezoid(y, x=None, dx=1.0, axis=-1):
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(y, x=x, dx=dx, axis=axis)


def _signal_summary(values, prefix):
    if not values:
        return {f"{prefix}_mean": 0.0, f"{prefix}_p05": 0.0, f"{prefix}_p50": 0.0, f"{prefix}_p95": 0.0}
    flat = torch.cat(values).float()
    if flat.numel() == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_p05": 0.0, f"{prefix}_p50": 0.0, f"{prefix}_p95": 0.0}
    qs = torch.quantile(flat, torch.tensor([0.05, 0.50, 0.95], device=flat.device))
    return {
        f"{prefix}_mean": flat.mean().item(),
        f"{prefix}_p05": qs[0].item(),
        f"{prefix}_p50": qs[1].item(),
        f"{prefix}_p95": qs[2].item(),
    }


def _compute_ece(confs, corrects, n_bins=15):
    if len(confs) == 0:
        return 0.0
    confs = np.asarray(confs, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs < hi if hi < 1.0 else confs <= hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(corrects[mask].mean() - confs[mask].mean())
    return float(ece)


def _risk_coverage_auc(scores, corrects):
    scores = np.asarray(scores, dtype=float)
    corrects = np.asarray(corrects, dtype=bool)
    if scores.size == 0:
        return {"aurc": 0.0, "coverage_at_5pct_risk": 0.0}
    order = np.argsort(-scores)
    sorted_correct = corrects[order]
    coverage = np.arange(1, len(sorted_correct) + 1) / len(sorted_correct)
    risk = 1.0 - np.cumsum(sorted_correct) / np.arange(1, len(sorted_correct) + 1)
    aurc = float(_trapezoid(risk, coverage))
    ok = coverage[risk <= 0.05]
    return {"aurc": aurc, "coverage_at_5pct_risk": float(ok.max()) if ok.size else 0.0}


def _vqsa_batch_diagnostics(dbg: dict, batch_size: int) -> dict:
    device = dbg["z"].device
    scalar = lambda value: torch.full((batch_size,), float(value.detach().item()), device=device)
    dq_mean = dbg["dq_map"].flatten(1).mean(dim=1).detach()

    soft_assign = dbg.get("soft_assign")
    if soft_assign is None:
        assignment_entropy = torch.zeros(batch_size, device=device)
        codebook_perplexity = torch.zeros(batch_size, device=device)
    else:
        probs = soft_assign.clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1).mean(dim=(1, 2)).detach()
        assignment_entropy = entropy / np.log(probs.size(-1))
        codebook_perplexity = entropy.exp()

    attn = dbg.get("attention_weights")
    if attn is None:
        attention_entropy = torch.zeros(batch_size, device=device)
    else:
        probs = attn.clamp_min(1e-8)
        attention_entropy = (-(probs * probs.log()).sum(dim=-1).mean(dim=(1, 2, 3)) / np.log(probs.size(-1))).detach()

    hard_usage = torch.zeros(batch_size, device=device)
    hard_perplexity = torch.zeros(batch_size, device=device)
    dead_code_fraction = torch.zeros(batch_size, device=device)
    indices = dbg.get("encoding_indices")
    num_embeddings = int(dbg.get("num_embeddings", 0) or 0)
    if indices is not None and num_embeddings > 0:
        per_sample = indices.detach().reshape(batch_size, -1).long()
        usage_rows = []
        perplexity_rows = []
        dead_rows = []
        for sample_idx in per_sample:
            counts = torch.bincount(sample_idx, minlength=num_embeddings).float()
            used = counts > 0
            probs = counts / counts.sum().clamp_min(1.0)
            entropy = -(probs[used] * probs[used].clamp_min(1e-8).log()).sum()
            usage_rows.append(used.float().mean())
            perplexity_rows.append(entropy.exp())
            dead_rows.append((~used).float().mean())
        hard_usage = torch.stack(usage_rows).to(device)
        hard_perplexity = torch.stack(perplexity_rows).to(device)
        dead_code_fraction = torch.stack(dead_rows).to(device)

    return {
        "vq_loss": scalar(dbg["vq_loss"]),
        "codebook_loss": scalar(dbg["codebook_loss"]),
        "commitment_loss": scalar(dbg["commitment_loss"]),
        "dq_mean": dq_mean,
        "assignment_entropy": assignment_entropy,
        "codebook_perplexity": codebook_perplexity,
        "hard_usage": hard_usage,
        "hard_perplexity": hard_perplexity,
        "dead_code_fraction": dead_code_fraction,
        "attention_entropy": attention_entropy,
    }


def _teacher_batch_diagnostics(
    student_dbg: dict,
    teacher_dbg: dict,
    batch_size: int,
) -> dict:
    """Per-sample drift / churn / KL of the student against the frozen teacher."""
    device = student_dbg["z"].device
    zeros = torch.zeros(batch_size, device=device)
    z_s, z_t = student_dbg.get("z"), teacher_dbg.get("z")
    zq_s, zq_t = student_dbg.get("zq"), teacher_dbg.get("zq")
    if z_s is not None and z_t is not None:
        z_drift = (z_s - z_t).pow(2).mean(dim=(1, 2, 3)).clamp_min(0.0).sqrt().detach()
    else:
        z_drift = zeros
    if zq_s is not None and zq_t is not None:
        zq_drift = (zq_s - zq_t).pow(2).mean(dim=(1, 2, 3)).clamp_min(0.0).sqrt().detach()
    else:
        zq_drift = zeros

    idx_s = student_dbg.get("encoding_indices")
    idx_t = teacher_dbg.get("encoding_indices")
    if idx_s is not None and idx_t is not None:
        churn = (
            idx_s.detach().reshape(batch_size, -1) != idx_t.detach().reshape(batch_size, -1)
        ).float().mean(dim=1)
    else:
        churn = zeros

    soft_s = student_dbg.get("soft_assign")
    soft_t = teacher_dbg.get("soft_assign")
    if soft_s is not None and soft_t is not None:
        log_p = soft_s.clamp_min(1e-8).log()
        log_p_t = soft_t.clamp_min(1e-8).log()
        kl_per_token = (soft_t.detach() * (log_p_t.detach() - log_p.detach())).sum(dim=-1)
        kl = kl_per_token.mean(dim=(1, 2))
    else:
        kl = zeros

    return {
        "z_drift": z_drift,
        "zq_drift": zq_drift,
        "assignment_churn": churn,
        "kl_assign_src": kl,
    }


def _add_global_hard_counts(counts: torch.Tensor | None, dbg: dict) -> torch.Tensor | None:
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


def _global_hard_summary(counts: torch.Tensor | None) -> dict:
    if counts is None or counts.numel() == 0 or counts.sum().item() <= 0:
        return {"hard_usage_mean": 0.0, "hard_perplexity_mean": 0.0, "dead_code_fraction_mean": 0.0}
    probs = counts / counts.sum().clamp_min(1.0)
    used = counts > 0
    entropy = -(probs[used] * probs[used].log()).sum()
    return {
        "hard_usage_mean": used.float().mean().item(),
        "hard_perplexity_mean": entropy.exp().item(),
        "dead_code_fraction_mean": (~used).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# DeMemte VQSA evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dememte(
    model,
    loader,
    device: str = "cuda",
    corruption: Optional[str] = None,
    severity: float = 0.0,
    base_seed: int = 1234,
    return_predictions: bool = False,
):
    model.eval()
    total = 0
    correct = 0
    diag_values = {k: [] for k in VQSA_KEYS}
    hard_counts = None
    conf_all, corr_all, nll_all, brier_all = [], [], [], []
    prediction_rows = []

    g = torch.Generator(device=device)
    corr_offset = 0 if corruption is None else sum(ord(c) for c in corruption)
    g.manual_seed(base_seed + int(1000 * severity) + corr_offset)

    sample_offset = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_eval = apply_eval_corruption(x, corruption, severity, g)
        logits, _, dbg = model(x_eval, return_debug=True)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        ok = pred == y
        y_prob = probs.gather(1, y.view(-1, 1)).squeeze(1).clamp_min(1e-12)
        one_hot = F.one_hot(y, num_classes=probs.size(1)).float()
        brier = ((probs - one_hot) ** 2).sum(dim=1)
        diagnostics = _vqsa_batch_diagnostics(dbg, y.size(0))
        hard_counts = _add_global_hard_counts(hard_counts, dbg)

        total += y.size(0)
        correct += ok.sum().item()
        conf_all.extend(conf.detach().cpu().tolist())
        corr_all.extend(ok.detach().cpu().tolist())
        nll_all.extend((-torch.log(y_prob)).detach().cpu().tolist())
        brier_all.extend(brier.detach().cpu().tolist())
        for key in VQSA_KEYS:
            diag_values[key].append(diagnostics[key].detach().cpu())

        if return_predictions:
            for j in range(y.size(0)):
                row = {
                    "sample_id": sample_offset + j,
                    "corruption": "clean" if corruption is None else corruption,
                    "severity": float(severity),
                    "target": int(y[j].item()),
                    "pred": int(pred[j].item()),
                    "correct": bool(ok[j].item()),
                    "confidence": float(conf[j].item()),
                    "nll": float(-torch.log(y_prob[j]).item()),
                    "brier": float(brier[j].item()),
                }
                for key in VQSA_KEYS:
                    row[key] = float(diagnostics[key][j].item())
                prediction_rows.append(row)
            sample_offset += y.size(0)

    rc_conf = _risk_coverage_auc(conf_all, corr_all)
    result = {
        "acc": correct / max(1, total),
        "ece": _compute_ece(conf_all, corr_all),
        "nll": float(np.mean(nll_all)) if nll_all else 0.0,
        "brier": float(np.mean(brier_all)) if brier_all else 0.0,
        "aurc_confidence": rc_conf["aurc"],
    }
    for key, values in diag_values.items():
        result.update(_signal_summary(values, key))
    result.update(_global_hard_summary(hard_counts))
    if return_predictions:
        result["predictions"] = prediction_rows
    return result


def evaluate_dememte_suite(model, loader, device="cuda", return_predictions=False, suite=None):
    suite = suite or STRICT_SUITE
    clean = evaluate_dememte(model, loader, device=device, corruption=None, severity=0.0, return_predictions=return_predictions)
    corrupt_records = {}
    for corr, levels in suite.items():
        corrupt_records[corr] = [
            evaluate_dememte(model, loader, device=device, corruption=corr, severity=l, return_predictions=return_predictions)
            for l in levels
        ]
    acc_by_corr = {f"corrupt_acc_{c}": float(np.mean([r["acc"] for r in rows])) for c, rows in corrupt_records.items()}
    all_corrupt = [r for rows in corrupt_records.values() for r in rows]
    metrics = {
        "clean_acc": clean["acc"],
        "corrupt_acc_avg": float(np.mean(list(acc_by_corr.values()))),
        **acc_by_corr,
        "ece_clean": clean["ece"],
        "ece_corrupt_avg": float(np.mean([r["ece"] for r in all_corrupt])),
        "nll_clean": clean["nll"],
        "nll_corrupt_avg": float(np.mean([r["nll"] for r in all_corrupt])),
        "brier_clean": clean["brier"],
        "brier_corrupt_avg": float(np.mean([r["brier"] for r in all_corrupt])),
        "corruption_records": corrupt_records,
        "clean_record": clean,
    }
    for key in VQSA_KEYS:
        metrics[f"{key}_clean"] = clean[f"{key}_mean"]
        metrics[f"{key}_corrupt_avg"] = float(np.mean([r[f"{key}_mean"] for r in all_corrupt]))
    return metrics


def _tta_stats_dict(adapter) -> dict:
    stats = getattr(adapter, "stats", None)
    if stats is None:
        return {
            "tta_updates": 0,
            "tta_reliable": 0,
            "tta_selected": 0,
            "tta_seen": 0,
            "tta_selection_rate": 0.0,
        }
    seen = int(getattr(stats, "seen", 0))
    selected = int(getattr(stats, "selected", 0))
    return {
        "tta_updates": int(getattr(stats, "updates", 0)),
        "tta_reliable": int(getattr(stats, "reliable", 0)),
        "tta_selected": selected,
        "tta_seen": seen,
        "tta_selection_rate": float(selected / max(1, seen)),
    }


def evaluate_dememte_tta(
    adapter,
    loader,
    device: str = "cuda",
    corruption: Optional[str] = None,
    severity: float = 0.0,
    base_seed: int = 1234,
    return_predictions: bool = False,
    tta_method: str = "tta",
    tta_base_variant: str = "unknown",
    teacher_model=None,
):
    total = 0
    correct = 0
    diag_values = {k: [] for k in VQSA_KEYS}
    teacher_diag_values = {k: [] for k in TEACHER_DIAG_KEYS}
    hard_counts = None
    teacher_hard_counts = None
    conf_all, corr_all, nll_all, brier_all = [], [], [], []
    prediction_rows = []

    g = torch.Generator(device=device)
    corr_offset = 0 if corruption is None else sum(ord(c) for c in corruption)
    g.manual_seed(base_seed + int(1000 * severity) + corr_offset)

    sample_offset = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_eval = apply_eval_corruption(x, corruption, severity, g)
        logits, dbg = adapter(x_eval, return_debug=True)
        probs = torch.softmax(logits.detach(), dim=1)
        conf, pred = probs.max(1)
        ok = pred == y
        y_prob = probs.gather(1, y.view(-1, 1)).squeeze(1).clamp_min(1e-12)
        one_hot = F.one_hot(y, num_classes=probs.size(1)).float()
        brier = ((probs - one_hot) ** 2).sum(dim=1)
        diagnostics = _vqsa_batch_diagnostics(dbg, y.size(0))
        hard_counts = _add_global_hard_counts(hard_counts, dbg)
        teacher_diagnostics = None
        if teacher_model is not None:
            with torch.no_grad():
                _, _, teacher_dbg = teacher_model(x_eval, return_debug=True)
            teacher_diagnostics = _teacher_batch_diagnostics(dbg, teacher_dbg, y.size(0))
            teacher_hard_counts = _add_global_hard_counts(teacher_hard_counts, teacher_dbg)

        total += y.size(0)
        correct += ok.sum().item()
        conf_all.extend(conf.detach().cpu().tolist())
        corr_all.extend(ok.detach().cpu().tolist())
        nll_all.extend((-torch.log(y_prob)).detach().cpu().tolist())
        brier_all.extend(brier.detach().cpu().tolist())
        for key in VQSA_KEYS:
            diag_values[key].append(diagnostics[key].detach().cpu())
        if teacher_diagnostics is not None:
            for key in TEACHER_DIAG_KEYS:
                teacher_diag_values[key].append(teacher_diagnostics[key].detach().cpu())

        if return_predictions:
            for j in range(y.size(0)):
                row = {
                    "sample_id": sample_offset + j,
                    "corruption": "clean" if corruption is None else corruption,
                    "severity": float(severity),
                    "target": int(y[j].item()),
                    "pred": int(pred[j].item()),
                    "correct": bool(ok[j].item()),
                    "confidence": float(conf[j].item()),
                    "nll": float(-torch.log(y_prob[j]).item()),
                    "brier": float(brier[j].item()),
                    "tta_method": tta_method,
                    "tta_base_variant": tta_base_variant,
                }
                for key in VQSA_KEYS:
                    row[key] = float(diagnostics[key][j].item())
                if teacher_diagnostics is not None:
                    for key in TEACHER_DIAG_KEYS:
                        row[key] = float(teacher_diagnostics[key][j].item())
                prediction_rows.append(row)
            sample_offset += y.size(0)

    rc_conf = _risk_coverage_auc(conf_all, corr_all)
    result = {
        "acc": correct / max(1, total),
        "ece": _compute_ece(conf_all, corr_all),
        "nll": float(np.mean(nll_all)) if nll_all else 0.0,
        "brier": float(np.mean(brier_all)) if brier_all else 0.0,
        "aurc_confidence": rc_conf["aurc"],
        "tta_method": tta_method,
        "tta_base_variant": tta_base_variant,
    }
    result.update(_tta_stats_dict(adapter))
    for key, values in diag_values.items():
        result.update(_signal_summary(values, key))
    result.update(_global_hard_summary(hard_counts))
    if teacher_model is not None:
        for key, values in teacher_diag_values.items():
            result.update(_signal_summary(values, key))
        teacher_global = _global_hard_summary(teacher_hard_counts)
        result["hard_usage_delta_vs_src"] = result["hard_usage_mean"] - teacher_global["hard_usage_mean"]
        result["dead_code_fraction_delta_vs_src"] = (
            result["dead_code_fraction_mean"] - teacher_global["dead_code_fraction_mean"]
        )
        result["teacher_hard_usage_mean"] = teacher_global["hard_usage_mean"]
        result["teacher_dead_code_fraction_mean"] = teacher_global["dead_code_fraction_mean"]
    if return_predictions:
        result["predictions"] = prediction_rows
    return result


def evaluate_dememte_tta_suite(
    adapter_factory,
    loader,
    device="cuda",
    return_predictions=False,
    suite=None,
    tta_method="tta",
    tta_base_variant="unknown",
    teacher_model=None,
):
    suite = suite or STRICT_SUITE
    clean = evaluate_dememte_tta(
        adapter_factory(),
        loader,
        device=device,
        corruption=None,
        severity=0.0,
        return_predictions=return_predictions,
        tta_method=tta_method,
        tta_base_variant=tta_base_variant,
        teacher_model=teacher_model,
    )
    corrupt_records = {}
    for corr, levels in suite.items():
        corrupt_records[corr] = [
            evaluate_dememte_tta(
                adapter_factory(),
                loader,
                device=device,
                corruption=corr,
                severity=l,
                return_predictions=return_predictions,
                tta_method=tta_method,
                tta_base_variant=tta_base_variant,
                teacher_model=teacher_model,
            )
            for l in levels
        ]
    acc_by_corr = {f"corrupt_acc_{c}": float(np.mean([r["acc"] for r in rows])) for c, rows in corrupt_records.items()}
    all_corrupt = [r for rows in corrupt_records.values() for r in rows]
    metrics = {
        "clean_acc": clean["acc"],
        "corrupt_acc_avg": float(np.mean(list(acc_by_corr.values()))),
        **acc_by_corr,
        "ece_clean": clean["ece"],
        "ece_corrupt_avg": float(np.mean([r["ece"] for r in all_corrupt])),
        "nll_clean": clean["nll"],
        "nll_corrupt_avg": float(np.mean([r["nll"] for r in all_corrupt])),
        "brier_clean": clean["brier"],
        "brier_corrupt_avg": float(np.mean([r["brier"] for r in all_corrupt])),
        "tta_updates_clean": clean["tta_updates"],
        "tta_updates_corrupt_avg": float(np.mean([r["tta_updates"] for r in all_corrupt])),
        "tta_selection_rate_clean": clean["tta_selection_rate"],
        "tta_selection_rate_corrupt_avg": float(np.mean([r["tta_selection_rate"] for r in all_corrupt])),
        "tta_method": tta_method,
        "tta_base_variant": tta_base_variant,
        "corruption_records": corrupt_records,
        "clean_record": clean,
    }
    for key in VQSA_KEYS:
        metrics[f"{key}_clean"] = clean[f"{key}_mean"]
        metrics[f"{key}_corrupt_avg"] = float(np.mean([r[f"{key}_mean"] for r in all_corrupt]))
    if teacher_model is not None:
        for key in TEACHER_DIAG_KEYS:
            metrics[f"{key}_clean"] = clean[f"{key}_mean"]
            metrics[f"{key}_corrupt_avg"] = float(np.mean([r[f"{key}_mean"] for r in all_corrupt]))
        for key in ("hard_usage_delta_vs_src", "dead_code_fraction_delta_vs_src"):
            metrics[f"{key}_clean"] = clean[key]
            metrics[f"{key}_corrupt_avg"] = float(np.mean([r[key] for r in all_corrupt]))
    return metrics


def signal_curve_rows(variant_name, label, clean_record, corruption_records, suite=None):
    suite = suite or STRICT_SUITE
    rows = []
    clean_row = {"variant": variant_name, "model": label, "corruption": "clean", "severity": 0.0}
    clean_row.update({k: v for k, v in clean_record.items() if isinstance(v, (int, float, bool, np.floating))})
    rows.append(clean_row)
    for corr, records in corruption_records.items():
        for level, rec in zip(suite[corr], records):
            row = {"variant": variant_name, "model": label, "corruption": corr, "severity": level}
            row.update({k: v for k, v in rec.items() if isinstance(v, (int, float, bool, np.floating))})
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_baseline(model, loader, device="cuda", corruption=None, severity=0.0, base_seed=1234, return_predictions=False):
    model.eval()
    total, correct = 0, 0
    conf_all, corr_all, nll_all, brier_all = [], [], [], []
    rows = []
    g = torch.Generator(device=device)
    corr_offset = 0 if corruption is None else sum(ord(c) for c in corruption)
    g.manual_seed(base_seed + int(1000 * severity) + corr_offset)
    sample_offset = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_eval = apply_eval_corruption(x, corruption, severity, g)
        logits = model(x_eval)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        ok = pred == y
        y_prob = probs.gather(1, y.view(-1, 1)).squeeze(1).clamp_min(1e-12)
        one_hot = F.one_hot(y, num_classes=probs.size(1)).float()
        brier = ((probs - one_hot) ** 2).sum(dim=1)

        total += y.size(0)
        correct += ok.sum().item()
        conf_all.extend(conf.detach().cpu().tolist())
        corr_all.extend(ok.detach().cpu().tolist())
        nll_all.extend((-torch.log(y_prob)).detach().cpu().tolist())
        brier_all.extend(brier.detach().cpu().tolist())

        if return_predictions:
            for j in range(y.size(0)):
                rows.append({
                    "sample_id": sample_offset + j,
                    "corruption": "clean" if corruption is None else corruption,
                    "severity": float(severity),
                    "target": int(y[j].item()),
                    "pred": int(pred[j].item()),
                    "correct": bool(ok[j].item()),
                    "confidence": float(conf[j].item()),
                    "nll": float(-torch.log(y_prob[j]).item()),
                    "brier": float(brier[j].item()),
                })
            sample_offset += y.size(0)

    result = {
        "acc": correct / max(1, total),
        "ece": _compute_ece(conf_all, corr_all),
        "nll": float(np.mean(nll_all)) if nll_all else 0.0,
        "brier": float(np.mean(brier_all)) if brier_all else 0.0,
    }
    if return_predictions:
        result["predictions"] = rows
    return result


def evaluate_baseline_suite(model, loader, device="cuda", return_predictions=False, suite=None):
    suite = suite or STRICT_SUITE
    clean = evaluate_baseline(model, loader, device=device, return_predictions=return_predictions)
    corrupt_records = {}
    for corr, levels in suite.items():
        corrupt_records[corr] = [
            evaluate_baseline(model, loader, device=device, corruption=corr, severity=l, return_predictions=return_predictions)
            for l in levels
        ]
    acc_by_corr = {f"corrupt_acc_{c}": float(np.mean([r["acc"] for r in rows])) for c, rows in corrupt_records.items()}
    all_corrupt = [r for rows in corrupt_records.values() for r in rows]
    metrics = {
        "clean_acc": clean["acc"],
        "corrupt_acc_avg": float(np.mean(list(acc_by_corr.values()))),
        **acc_by_corr,
        "ece_clean": clean["ece"],
        "ece_corrupt_avg": float(np.mean([r["ece"] for r in all_corrupt])),
        "nll_clean": clean["nll"],
        "nll_corrupt_avg": float(np.mean([r["nll"] for r in all_corrupt])),
        "brier_clean": clean["brier"],
        "brier_corrupt_avg": float(np.mean([r["brier"] for r in all_corrupt])),
        "corruption_records": corrupt_records,
        "clean_record": clean,
    }
    return metrics


def signal_curve_rows_baseline(label, clean_record, corruption_records, suite=None):
    suite = suite or STRICT_SUITE
    rows = []
    clean_row = {"variant": label, "model": label, "corruption": "clean", "severity": 0.0}
    clean_row.update({k: v for k, v in clean_record.items() if isinstance(v, (int, float, bool, np.floating))})
    rows.append(clean_row)
    for corr, records in corruption_records.items():
        for level, rec in zip(suite[corr], records):
            row = {"variant": label, "model": label, "corruption": corr, "severity": level}
            row.update({k: v for k, v in rec.items() if isinstance(v, (int, float, bool, np.floating))})
            rows.append(row)
    return rows
