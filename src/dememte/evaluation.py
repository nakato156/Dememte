"""Evaluation: clean accuracy, paired corruption suite, gate/calibration metrics.

Mirrors `evaluate_extended` + `evaluate_attractor_suite` from `Dememte_e5y.ipynb`.
Provides a simpler `evaluate_baseline_suite` for the plain ResNet baseline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .corruptions import STRICT_SUITE, apply_eval_corruption


SIGNAL_KEYS = ["dq_norm", "uncertainty", "familiarity", "conflict", "ood_risk", "gate_prior", "gate_raw", "gate"]


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


# ---------------------------------------------------------------------------
# DeMemte evaluation
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
    final_correct = 0
    base_correct = 0
    pred_changed = 0
    beneficial = 0
    harmful = 0
    pareidolia = 0
    gate_values, gate_entropy_values = [], []
    signal_values = {k: [] for k in SIGNAL_KEYS}
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
        logits, _, _, dbg = model(x_eval, return_debug=True, update_ema=False)
        logits_base = dbg["logits_base"]
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        base_pred = logits_base.argmax(1)
        final_ok = pred == y
        base_ok = base_pred == y
        changed = pred != base_pred
        y_prob = probs.gather(1, y.view(-1, 1)).squeeze(1).clamp_min(1e-12)
        one_hot = F.one_hot(y, num_classes=probs.size(1)).float()
        brier = ((probs - one_hot) ** 2).sum(dim=1)

        final_correct += final_ok.sum().item()
        base_correct += base_ok.sum().item()
        pred_changed += changed.sum().item()
        beneficial += (changed & (~base_ok) & final_ok).sum().item()
        harmful += (changed & base_ok & (~final_ok)).sum().item()
        gate_flat = dbg["gate"].view(-1).detach()
        pareidolia += ((gate_flat > 0.5) & base_ok & (~final_ok)).sum().item()
        gclip = gate_flat.clamp(1e-6, 1.0 - 1e-6)
        gate_entropy = -(gclip * torch.log(gclip) + (1.0 - gclip) * torch.log(1.0 - gclip))
        gate_entropy_values.append(gate_entropy.cpu())
        gate_values.append(gate_flat.cpu())
        for key in SIGNAL_KEYS:
            signal_values[key].append(dbg[key].view(-1).detach().cpu())
        total += y.size(0)
        conf_all.extend(conf.detach().cpu().tolist())
        corr_all.extend(final_ok.detach().cpu().tolist())
        nll_all.extend((-torch.log(y_prob)).detach().cpu().tolist())
        brier_all.extend(brier.detach().cpu().tolist())

        if return_predictions:
            for j in range(y.size(0)):
                prediction_rows.append({
                    "sample_id": sample_offset + j,
                    "corruption": "clean" if corruption is None else corruption,
                    "severity": float(severity),
                    "target": int(y[j].item()),
                    "pred": int(pred[j].item()),
                    "base_pred": int(base_pred[j].item()),
                    "correct": bool(final_ok[j].item()),
                    "base_correct": bool(base_ok[j].item()),
                    "confidence": float(conf[j].item()),
                    "gate": float(gate_flat[j].item()),
                    "gate_raw": float(dbg["gate_raw"].view(-1)[j].item()),
                    "familiarity": float(dbg["familiarity"].view(-1)[j].item()),
                    "ood_risk": float(dbg["ood_risk"].view(-1)[j].item()),
                    "dq_norm": float(dbg["dq_norm"].view(-1)[j].item()),
                    "gate_entropy": float(gate_entropy[j].item()),
                    "nll": float(-torch.log(y_prob[j]).item()),
                    "brier": float(brier[j].item()),
                })
            sample_offset += y.size(0)

    gates = torch.cat(gate_values) if gate_values else torch.empty(0)
    gate_entropy_cat = torch.cat(gate_entropy_values) if gate_entropy_values else torch.empty(0)
    rc_conf = _risk_coverage_auc(conf_all, corr_all)
    rc_gate = _risk_coverage_auc(gates.numpy().tolist() if gates.numel() else [], corr_all)
    result = {
        "acc": final_correct / max(1, total),
        "base_acc": base_correct / max(1, total),
        "gate_mean": gates.mean().item() if gates.numel() else 0.0,
        "pred_change_rate": pred_changed / max(1, total),
        "beneficial_changes": beneficial / max(1, total),
        "harmful_changes": harmful / max(1, total),
        "pareidolia_rate": pareidolia / max(1, total),
        "gate_entropy": gate_entropy_cat.mean().item() if gate_entropy_cat.numel() else 0.0,
        "ece": _compute_ece(conf_all, corr_all),
        "nll": float(np.mean(nll_all)) if nll_all else 0.0,
        "brier": float(np.mean(brier_all)) if brier_all else 0.0,
        "aurc_confidence": rc_conf["aurc"],
        "aurc_gate": rc_gate["aurc"],
    }
    for key, values in signal_values.items():
        result.update(_signal_summary(values, key))
    if return_predictions:
        result["predictions"] = prediction_rows
    return result


def evaluate_dememte_suite(model, loader, device="cuda", return_predictions=False, suite=None):
    suite = suite or STRICT_SUITE
    clean = evaluate_dememte(model, loader, device=device, corruption=None, severity=0.0, return_predictions=return_predictions)
    corrupt_records = {}
    for corr, levels in suite.items():
        corrupt_records[corr] = [evaluate_dememte(model, loader, device=device, corruption=corr, severity=l, return_predictions=return_predictions) for l in levels]
    acc_by_corr = {f"corrupt_acc_{c}": float(np.mean([r["acc"] for r in rows])) for c, rows in corrupt_records.items()}
    all_corrupt = [r for rows in corrupt_records.values() for r in rows]
    corrupt_acc_avg = float(np.mean(list(acc_by_corr.values())))
    metrics = {
        "clean_acc": clean["acc"],
        "corrupt_acc_avg": corrupt_acc_avg,
        **acc_by_corr,
        "gate_mean_clean": clean["gate_mean"],
        "pred_change_rate": float(np.mean([r["pred_change_rate"] for r in all_corrupt])),
        "beneficial_changes": float(np.mean([r["beneficial_changes"] for r in all_corrupt])),
        "harmful_changes": float(np.mean([r["harmful_changes"] for r in all_corrupt])),
        "pareidolia_rate": float(np.mean([r["pareidolia_rate"] for r in all_corrupt])),
        "gate_entropy": clean["gate_entropy"],
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
        corrupt_records[corr] = [evaluate_baseline(model, loader, device=device, corruption=corr, severity=l, return_predictions=return_predictions) for l in levels]
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
