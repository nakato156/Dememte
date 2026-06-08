"""wtq.7 — Port E11 retrieval-logit memory to CIFAR-10-C / CIFAR-100-C (severity curve).

The slate axis that ImageNet-C (sev 3,5 only) and Flowers (ad-hoc 3-grid) don't give cleanly:
**how the retrieval gain scales with severity 1–5**, over the 15 canonical Hendrycks corruptions.
Complements wtq.6 (ImageNet-R = nature of the shift); CIFAR-C = the severity curve.

Unlike wtq.5/wtq.6 there is no CIFAR VQSA checkpoint and CIFAR labels don't map to ImageNet, so
this trains the substrate first: frozen RN18 + VQSA + a 10/100-class head (the E6 winner recipe,
identical to Flowers). Then it builds a clean source cache (z_pool) and evaluates source vs
retrieval@alpha across severity 1–5 on CIFAR-C.

Two outputs:
  out/e14_cifar_c.csv        per (dataset, variant, corruption, severity)  — detail
  out/e14_cifar_c_curve.csv  mean over the 15 corruptions per (dataset, variant, severity) — curve

Tracking: bead Demente-wtq.7. Results write-up in insights.md.

Run:  PYTHONPATH=src uv run python notebooks/14_cifar_c/e14_cifar_c.py [--smoke]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dememte.config import CIFAR10Config, CIFAR100Config, e6_config
from dememte.data import CIFAR_C_CORRUPTIONS, CIFARCDataset, build_cifar_loaders
from dememte.evaluation import evaluate_dememte, evaluate_dememte_tta
from dememte.io import load_checkpoint, save_checkpoint, write_csv, write_json
from dememte.models.dememte import make_dememte_variant
from dememte.retrieval import RetrievalConfig, RetrievalLogitAdapter, build_labeled_cache
from dememte.training import train_dememte_vqsa

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATA_ROOT = os.environ.get("CIFAR_DATA_ROOT", "/home/r0sewt/data")  # off /shared (disk full); override per-host
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_WORKERS = 4
ALPHA_GRID = [0.5, 1.0, 2.0]
SEVERITIES = [1, 2, 3, 4, 5]
RUN_TRAINING = os.environ.get("RUN_TRAINING", "1") != "0"  # train if checkpoint missing
DATASETS = {
    "cifar10": (CIFAR10Config, 10),
    "cifar100": (CIFAR100Config, 100),
}


def _ckpt_path(dataset):
    return OUT / f"{dataset}_vqsa.pt"


def get_model(dataset, base_cls, num_classes, tr_loader, va_loader, smoke):
    """Reconstruct the VQSA model; train it (E6 winner recipe) if the checkpoint is missing."""
    cfg = e6_config("e6_ema_kmeans_restart", base=base_cls())
    if smoke:
        cfg.epochs_vqsa_max = 1
    model = make_dememte_variant(cfg, device=DEVICE)
    ckpt = _ckpt_path(dataset)
    if ckpt.exists() and not smoke:
        load_checkpoint(model, ckpt, device=DEVICE, strict=True)
        print(f"[{dataset}] loaded checkpoint {ckpt.name}")
    elif RUN_TRAINING:
        print(f"[{dataset}] training VQSA (e6_ema_kmeans_restart, epochs_max={cfg.epochs_vqsa_max})")
        model, best = train_dememte_vqsa(model, tr_loader, va_loader, cfg, DEVICE)
        if not smoke:
            save_checkpoint(model, ckpt, extra={"best_val": best, "config": cfg.__dict__})
        print(f"[{dataset}] trained best_val={best:.4f}")
    else:
        raise FileNotFoundError(f"missing {ckpt} and RUN_TRAINING=0")
    model.eval()
    return model, num_classes


def cifar_c_loader(dataset, corruption, severity, max_samples=None):
    ds = CIFARCDataset(DATA_ROOT, dataset, corruption, severity, max_samples=max_samples)  # defaults to eval transform
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)


def _row(dataset, variant, corruption, severity, m):
    return {"dataset": dataset, "variant": variant, "corruption": corruption, "severity": severity,
            "acc": m["acc"], "ece": m["ece"], "nll": m["nll"], "brier": m["brier"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE} smoke={args.smoke} data_root={DATA_ROOT}")

    datasets = ["cifar10"] if args.smoke else list(DATASETS)
    corruptions = CIFAR_C_CORRUPTIONS[:1] if args.smoke else CIFAR_C_CORRUPTIONS
    severities = [1] if args.smoke else SEVERITIES
    alpha_grid = ALPHA_GRID[:1] if args.smoke else ALPHA_GRID
    cap = 512 if args.smoke else None

    rows = []
    for dataset in datasets:
        base_cls, num_classes = DATASETS[dataset]
        tr, va, te, meta = build_cifar_loaders(DATA_ROOT, dataset, batch_size=BATCH_SIZE,
                                               num_workers=NUM_WORKERS, split_seed=42, pin_memory=False)
        print(f"[{dataset}] train={meta['train_size']} val={meta['val_size']} test={meta['test_size']}")
        model, ncls = get_model(dataset, base_cls, num_classes, tr, va, args.smoke)
        source_cache = build_labeled_cache(model, tr, device=DEVICE, key_space="z_pool", num_classes=ncls)
        print(f"[{dataset}] source cache: {source_cache.size} keys (z_pool)")

        for sev in severities:
            for corr in corruptions:
                loader = cifar_c_loader(dataset, corr, sev, max_samples=cap)
                rows.append(_row(dataset, "source", corr, sev,
                                 evaluate_dememte(model, loader, device=DEVICE, corruption=None)))
                for alpha in alpha_grid:
                    adapter = RetrievalLogitAdapter(
                        model,
                        RetrievalConfig(key_space="z_pool", alpha_mode="fixed", alpha_max=alpha,
                                        cache_source=True, episodic_size=0, top_k=16, beta=5.0),
                        source_cache=source_cache, num_classes=ncls,
                    )
                    rows.append(_row(dataset, f"retrieval_z_pool_fixed_alpha@{alpha}", corr, sev,
                                     evaluate_dememte_tta(adapter, loader, device=DEVICE, corruption=None,
                                                          tta_method="retrieval_logit", tta_base_variant=dataset)))
            done = [r for r in rows if r["dataset"] == dataset and r["severity"] == sev and r["variant"] == "source"]
            print(f"[{dataset} sev {sev}] source acc(mean over {len(done)} corr)="
                  f"{sum(r['acc'] for r in done)/max(len(done),1):.4f}")

    write_csv(rows, OUT / "e14_cifar_c.csv")

    # severity curve: mean over corruptions per (dataset, variant, severity)
    curve = []
    keys = sorted({(r["dataset"], r["variant"], r["severity"]) for r in rows})
    for ds, var, sev in keys:
        sel = [r for r in rows if r["dataset"] == ds and r["variant"] == var and r["severity"] == sev]
        n = len(sel)
        curve.append({
            "dataset": ds, "variant": var, "severity": sev, "n_corruptions": n,
            "acc": sum(r["acc"] for r in sel) / n,
            "ece": sum(r["ece"] for r in sel) / n,
            "nll": sum(r["nll"] for r in sel) / n,
            "brier": sum(r["brier"] for r in sel) / n,
        })
    write_csv(curve, OUT / "e14_cifar_c_curve.csv")
    write_json({"datasets": datasets, "severities": severities, "alpha_grid": alpha_grid,
                "corruptions": list(corruptions), "curve": curve}, OUT / "e14_cifar_c.json")
    print(f"[done] wrote {OUT/'e14_cifar_c.csv'} and {OUT/'e14_cifar_c_curve.csv'}")


if __name__ == "__main__":
    main()
