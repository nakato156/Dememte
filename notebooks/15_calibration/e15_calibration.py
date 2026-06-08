"""wtq.8 — calibration control for the E11 retrieval-logit memory (C8 gate, full slate).

Two post-hoc controls over the retrieval logits, measured across the slate:
  1. temperature scaling (fit T on clean-val, min NLL) — argmax-invariant, targets ECE/NLL.
  2. confidence gate (unfamiliarity: suppress the cache vote where the base is already confident)
     — targets the wtq.7 frontier (CIFAR-10 negative) + over-confidence.

Engineering: the adapter exposes base_logits + cache_logits + retrieval_margin per sample, so a
SINGLE forward pass per (domain, condition) is dumped and every variant (temp / gate / temp+gate,
any alpha) is computed OFFLINE — the full slate costs ~one eval per condition.

Outputs:
  out/e15_calibration.csv   per (domain, condition, kind, severity, variant): acc/ece/nll/brier
  out/e15_summary.csv       per (domain, variant): clean + corrupt-mean acc/ece/nll/brier
  out/e15_temperatures.json fitted T per (domain, variant)

Tracking: bead Demente-wtq.8. Write-up in insights.md.

Run:  PYTHONPATH=src uv run python notebooks/15_calibration/e15_calibration.py [--smoke] [--domains a,b]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dememte.calibration import apply_temperature, calibration_metrics, confidence_gate, fit_temperature
from dememte.config import CIFAR10Config, CIFAR100Config, E6Config, FlowersLegacyConfig, e6_config
from dememte.corruptions import STRICT_SUITE, apply_eval_corruption
from dememte.data import (
    ImageNetFolderDataset,
    build_cifar_loaders,
    build_imagenet_c_loaders,
    build_imagenet_loaders,
    build_flowers_loaders,
    load_imagenet_class_index,
    make_imagenet_eval_transform,
)
from dememte.io import load_checkpoint, write_csv, write_json
from dememte.models.dememte import make_dememte_variant
from dememte.retrieval import RetrievalConfig, RetrievalLogitAdapter, build_labeled_cache

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 1.0                 # canonical operating point (the C8 gate is about alpha~1.0)
GATE_MODE = "unfamiliarity"
KEY_SPACE = "z_pool"
BATCH = 128
NW = 4
ALL_DOMAINS = ["flowers", "cifar10", "cifar100", "imagenet_r", "imagenet_c"]


# --------------------------------------------------------------------------- #
# Adapter + collection
# --------------------------------------------------------------------------- #
def _adapter(model, cache, ncls):
    return RetrievalLogitAdapter(
        model,
        RetrievalConfig(key_space=KEY_SPACE, alpha_mode="fixed", alpha_max=ALPHA,
                        cache_source=True, episodic_size=0, top_k=16, beta=5.0),
        source_cache=cache, num_classes=ncls,
    )


@torch.no_grad()
def collect(adapter, cond, cap=None):
    """One forward pass over a condition -> per-sample {base, cache, margin, label}."""
    base, cache, margin, labels = [], [], [], []
    g = torch.Generator(device=DEVICE)
    sev = float(cond["corr_sev"])
    corr = cond["corruption"]
    g.manual_seed(1234 + int(1000 * sev) + (0 if corr is None else sum(ord(c) for c in corr)))
    seen = 0
    for x, y in cond["loader"]:
        x = x.to(DEVICE, non_blocking=True)
        x = apply_eval_corruption(x, corr, sev, g)
        _, dbg = adapter(x, return_debug=True)
        base.append(dbg["base_logits"].float().cpu())
        cache.append(dbg["cache_logits"].float().cpu())
        margin.append(dbg["retrieval_margin"].float().cpu())
        labels.append(y.cpu())
        seen += y.size(0)
        if cap is not None and seen >= cap:
            break
    return {
        "base": torch.cat(base), "cache": torch.cat(cache),
        "margin": torch.cat(margin), "label": torch.cat(labels).long(),
    }


# --------------------------------------------------------------------------- #
# Offline variants
# --------------------------------------------------------------------------- #
def _base_conf(base, mask):
    return torch.softmax(base + mask, dim=1).max(dim=1).values


def _gate_alpha(d, mask):
    bc = _base_conf(d["base"], mask)
    return confidence_gate(bc, torch.zeros_like(bc), d["margin"], alpha_max=ALPHA, mode=GATE_MODE)


def _logits(d, mask, alpha, T):
    a = alpha.unsqueeze(1) if torch.is_tensor(alpha) else alpha
    return apply_temperature(d["base"] + a * d["cache"], T) + mask


def fit_temps(clean, mask):
    """Fit T (min NLL on clean-val) per variant: source, retrieval, gate."""
    y = clean["label"]
    return {
        "source": fit_temperature(clean["base"] + mask, y),
        "retrieval": fit_temperature(_logits(clean, mask, ALPHA, 1.0), y),
        "gate": fit_temperature(_logits(clean, mask, _gate_alpha(clean, mask), 1.0), y),
    }


def eval_variants(d, mask, temps):
    y = d["label"]
    ga = _gate_alpha(d, mask)
    rows = {
        "source": calibration_metrics(d["base"] + mask, y),
        "retrieval": calibration_metrics(_logits(d, mask, ALPHA, 1.0), y),
        "retrieval_temp": calibration_metrics(_logits(d, mask, ALPHA, temps["retrieval"]), y),
        "gate_unfam": calibration_metrics(_logits(d, mask, ga, 1.0), y),
        "gate_unfam_temp": calibration_metrics(_logits(d, mask, ga, temps["gate"]), y),
    }
    return rows


# --------------------------------------------------------------------------- #
# Domain providers -> (adapter, ncls, mask, clean_val_cond, eval_conds)
# --------------------------------------------------------------------------- #
def _cond(label, kind, severity, loader, corruption=None, corr_sev=0.0):
    return {"label": label, "kind": kind, "severity": severity, "loader": loader,
            "corruption": corruption, "corr_sev": corr_sev}


def _ld(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=NW, pin_memory=False)


def provider_flowers(smoke):
    data_dir = (REPO / "experiments/data").as_posix()
    tr, va, te, _ = build_flowers_loaders(data_dir, batch_size=BATCH, num_workers=NW,
                                          val_ratio=0.2, split_seed=42,
                                          protocol="historical_trainval_resplit", download=False)
    cfg = e6_config("e6_ema_kmeans_restart", base=FlowersLegacyConfig())
    model = make_dememte_variant(cfg, device=DEVICE)
    load_checkpoint(model, REPO / "notebooks/06_e6_zq_alignment/out/e6_ema_kmeans_restart/best.pt",
                    device=DEVICE, strict=True)
    cache = build_labeled_cache(model, tr, device=DEVICE, key_space=KEY_SPACE, num_classes=102)
    mask = torch.zeros(102)
    conds = [_cond("clean", "clean", 0, te)]
    suite = {next(iter(STRICT_SUITE)): STRICT_SUITE[next(iter(STRICT_SUITE))]} if smoke else STRICT_SUITE
    for corr, levels in suite.items():
        for i, lv in enumerate(levels, 1):
            conds.append(_cond(f"{corr}@{lv}", "corrupt", i, te, corruption=corr, corr_sev=lv))
    return _adapter(model, cache, 102), 102, mask, _cond("val", "clean", 0, va), conds


def provider_cifar(ds, smoke):
    data_root = "/home/r0sewt/data"
    base_cls = CIFAR10Config if ds == "cifar10" else CIFAR100Config
    ncls = 10 if ds == "cifar10" else 100
    tr, va, te, _ = build_cifar_loaders(data_root, ds, batch_size=BATCH, num_workers=NW,
                                        split_seed=42, download=True, pin_memory=False)
    cfg = e6_config("e6_ema_kmeans_restart", base=base_cls())
    model = make_dememte_variant(cfg, device=DEVICE)
    load_checkpoint(model, REPO / f"notebooks/14_cifar_c/out/{ds}_vqsa.pt", device=DEVICE, strict=True)
    cache = build_labeled_cache(model, tr, device=DEVICE, key_space=KEY_SPACE, num_classes=ncls)
    mask = torch.zeros(ncls)
    from dememte.data import CIFAR_C_CORRUPTIONS, CIFARCDataset
    sevs = [1] if smoke else [1, 2, 3, 4, 5]
    corrs = CIFAR_C_CORRUPTIONS[:1] if smoke else CIFAR_C_CORRUPTIONS
    conds = [_cond("clean", "clean", 0, te)]
    for sev in sevs:
        for corr in corrs:
            cl = _ld(CIFARCDataset(data_root, ds, corr, sev))
            conds.append(_cond(f"{corr}@s{sev}", "corrupt", sev, cl))
    return _adapter(model, cache, ncls), ncls, mask, _cond("val", "clean", 0, va), conds


def _imagenet_rn50():
    tc = REPO / "experiments/imagenet_dememte/out/train_config.json"
    with open(tc) as f:
        cfg_json = json.load(f)["config"]
    valid = {fld.name for fld in dataclasses.fields(E6Config)}
    cfg = E6Config(**{k: v for k, v in cfg_json.items() if k in valid})
    model = make_dememte_variant(cfg, device=DEVICE)
    load_checkpoint(model, REPO / "experiments/imagenet_dememte/out/dememte_imagenet_resnet50_vqsa_best.pt",
                    device=DEVICE, strict=True)
    return model


def _folder(root, split, cap=None):
    return ImageNetFolderDataset(root, split, transform=make_imagenet_eval_transform(),
                                 class_index_path=None, download_class_index=True, max_samples=cap, seed=42)


def provider_imagenet_r(smoke):
    clean_dir = (REPO / "experiments/data/imagenet-clean-5k").as_posix()
    r_root, r_split = "/home/r0sewt/data", "imagenet-r"
    model = _imagenet_rn50()
    wnid_to_idx = load_imagenet_class_index(download=True)
    r_dir = Path(r_root) / r_split
    allowed = sorted(wnid_to_idx[p.name] for p in r_dir.iterdir() if p.is_dir() and p.name in wnid_to_idx)
    mask = torch.full((1000,), float("-inf")); mask[allowed] = 0.0
    cap = 1000 if smoke else None
    clean_train = _folder(clean_dir, "train", cap)
    clean_val = _folder(clean_dir, "val", cap)
    aset = set(allowed)
    ref_idx = [i for i, (_, t) in enumerate(clean_val.samples) if t in aset]
    clean_ref = Subset(clean_val, ref_idx)
    cache = build_labeled_cache(model, _ld(clean_train), device=DEVICE, key_space=KEY_SPACE, num_classes=1000)
    conds = [_cond("clean", "clean", 0, _ld(clean_ref)),
             _cond("imagenet_r", "corrupt", 0, _ld(_folder(r_root, r_split, cap)))]
    return _adapter(model, cache, 1000), 1000, mask, _cond("val", "clean", 0, _ld(clean_ref)), conds


def provider_imagenet_c(smoke):
    clean_dir = (REPO / "experiments/data/imagenet-clean-5k").as_posix()
    c_dir = (REPO / "experiments/data/imagenet-c-subset").as_posix()
    model = _imagenet_rn50()
    mask = torch.zeros(1000)
    tr_loader, va_loader, _ = build_imagenet_loaders(clean_dir, batch_size=BATCH, num_workers=NW, pin_memory=False)
    cache = build_labeled_cache(model, tr_loader, device=DEVICE, key_space=KEY_SPACE, num_classes=1000)
    sevs = [5] if smoke else [3, 5]
    corrs = ["gaussian_noise"] if smoke else ["gaussian_noise", "motion_blur", "pixelate", "jpeg_compression"]
    loaders, _ = build_imagenet_c_loaders(c_dir, corruptions=corrs, severities=sevs,
                                          batch_size=BATCH, num_workers=NW, pin_memory=False)
    conds = [_cond("clean", "clean", 0, va_loader)]
    for (corr, sev), ld in loaders.items():
        conds.append(_cond(f"{corr}@s{sev}", "corrupt", sev, ld))
    return _adapter(model, cache, 1000), 1000, mask, _cond("val", "clean", 0, va_loader), conds


PROVIDERS = {
    "flowers": lambda s: provider_flowers(s),
    "cifar10": lambda s: provider_cifar("cifar10", s),
    "cifar100": lambda s: provider_cifar("cifar100", s),
    "imagenet_r": lambda s: provider_imagenet_r(s),
    "imagenet_c": lambda s: provider_imagenet_c(s),
}


# --------------------------------------------------------------------------- #
def run_domain(name, smoke, cap):
    adapter, ncls, mask, clean_cond, conds = PROVIDERS[name](smoke)
    mask = mask.to("cpu")
    clean_dump = collect(adapter, clean_cond, cap)
    temps = fit_temps(clean_dump, mask)
    print(f"[{name}] T(source/retr/gate) = "
          f"{temps['source']:.3f}/{temps['retrieval']:.3f}/{temps['gate']:.3f}")
    del clean_dump
    rows = []
    for cond in conds:
        d = collect(adapter, cond, cap)
        for variant, m in eval_variants(d, mask, temps).items():
            rows.append({"domain": name, "condition": cond["label"], "kind": cond["kind"],
                         "severity": cond["severity"], "variant": variant,
                         "acc": m["acc"], "ece": m["ece"], "nll": m["nll"], "brier": m["brier"]})
        del d
    print(f"[{name}] {len(conds)} conditions x 5 variants done")
    return rows, temps


def summarize(rows):
    out = []
    domains = sorted({r["domain"] for r in rows})
    variants = ["source", "retrieval", "retrieval_temp", "gate_unfam", "gate_unfam_temp"]
    for dom in domains:
        for var in variants:
            cl = [r for r in rows if r["domain"] == dom and r["variant"] == var and r["kind"] == "clean"]
            co = [r for r in rows if r["domain"] == dom and r["variant"] == var and r["kind"] == "corrupt"]
            if not cl and not co:
                continue
            row = {"domain": dom, "variant": var}
            for m in ("acc", "ece", "nll", "brier"):
                row[f"clean_{m}"] = (sum(r[m] for r in cl) / len(cl)) if cl else None
                row[f"corrupt_{m}"] = (sum(r[m] for r in co) / len(co)) if co else None
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--domains", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    domains = (args.domains.split(",") if args.domains
               else (["cifar100"] if args.smoke else ALL_DOMAINS))
    cap = 512 if args.smoke else None
    print(f"device={DEVICE} smoke={args.smoke} domains={domains}")

    rows, all_temps = [], {}
    for name in domains:
        try:
            r, t = run_domain(name, args.smoke, cap)
            rows.extend(r)
            all_temps[name] = t
        except Exception as e:  # one domain failing shouldn't kill the slate
            print(f"[{name}] SKIPPED: {type(e).__name__}: {e}")

    write_csv(rows, OUT / "e15_calibration.csv")
    write_csv(summarize(rows), OUT / "e15_summary.csv")
    write_json({"alpha": ALPHA, "gate_mode": GATE_MODE, "temperatures": all_temps}, OUT / "e15_temperatures.json")
    print(f"[done] wrote {OUT/'e15_calibration.csv'} ({len(rows)} rows) + summary + temps")


if __name__ == "__main__":
    main()
