"""wtq.6 — Port E11 retrieval-logit memory to ImageNet-R (natural shift).

One output, one CSV (feeds the ImageNet-R row of the OOD matrix + reinforces C3/C4
with a NON-synthetic shift — refutes "the effect is only denoising of synthetic noise").

Setup: frozen VQSA ImageNet RN50 (E6 checkpoint), source retrieval cache from clean
ImageNet, evaluated on ImageNet-R with logits masked to its 200 classes (Hendrycks
protocol). Reuses existing code; new pieces are the ImageNet-R loader glue + the
200-class masking wrappers.

Tracking: bead Demente-wtq.6. Results write-up in insights.md.

Run:  PYTHONPATH=src uv run python notebooks/13_imagenet_r/e13_imagenet_r.py [--smoke]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dememte.config import E6Config
from dememte.data import ImageNetFolderDataset, load_imagenet_class_index, make_imagenet_eval_transform
from dememte.evaluation import evaluate_dememte, evaluate_dememte_tta
from dememte.io import write_csv, write_json
from dememte.models.dememte import make_dememte_variant
from dememte.retrieval import RetrievalConfig, RetrievalLogitAdapter, build_labeled_cache

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CLEAN_DIR = (REPO / "experiments/data/imagenet-clean-5k").as_posix()   # source cache + clean ref
R_ROOT = "/home/r0sewt/data"                                           # ImageNet-R lives off /shared (disk full)
R_SPLIT = "imagenet-r"
CKPT = REPO / "experiments/imagenet_dememte/out/dememte_imagenet_resnet50_vqsa_best.pt"
TRAIN_CONFIG = REPO / "experiments/imagenet_dememte/out/train_config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1000
BATCH_SIZE = 64
NUM_WORKERS = 4
ALPHA_GRID = [0.5, 1.0, 2.0]
SUMMARY_COLS = ["variant", "domain", "acc", "ece", "nll", "brier"]


# --------------------------------------------------------------------------- #
# Model (frozen RN50 VQSA, rebuilt from the saved train config)
# --------------------------------------------------------------------------- #
def load_model():
    cfg_json = json.load(open(TRAIN_CONFIG))["config"]
    valid = {f.name for f in dataclasses.fields(E6Config)}
    cfg = E6Config(**{k: v for k, v in cfg_json.items() if k in valid})
    model = make_dememte_variant(cfg, device=DEVICE)
    payload = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# 200-class mask (ImageNet-R) + masking wrappers (keep evaluation.py untouched)
# --------------------------------------------------------------------------- #
def imagenet_r_mask():
    wnid_to_idx = load_imagenet_class_index(download=True)
    r_dir = Path(R_ROOT) / R_SPLIT
    wnids = sorted(p.name for p in r_dir.iterdir() if p.is_dir() and p.name in wnid_to_idx)
    allowed = sorted(wnid_to_idx[w] for w in wnids)
    mask = torch.full((NUM_CLASSES,), float("-inf"), device=DEVICE)
    mask[allowed] = 0.0
    return allowed, mask


class MaskedModel(nn.Module):
    """Wrap a DeMemte model; mask logits to the allowed classes. 3-tuple forward."""

    def __init__(self, inner, mask):
        super().__init__()
        self.inner = inner
        self.register_buffer("mask", mask)

    def forward(self, x, return_debug=False):
        logits, vq_loss, dbg = self.inner(x, return_debug=True)
        logits = logits + self.mask
        return (logits, vq_loss, dbg) if return_debug else logits


class MaskedRetrieval(nn.Module):
    """Wrap a RetrievalLogitAdapter; mask logits. 2-tuple forward + shared stats."""

    def __init__(self, adapter, mask):
        super().__init__()
        self.adapter = adapter
        self.register_buffer("mask", mask)
        self.stats = adapter.stats  # evaluate_dememte_tta reads adapter.stats

    def forward(self, x, return_debug=False):
        logits, dbg = self.adapter(x, return_debug=True)
        logits = logits + self.mask
        return (logits, dbg) if return_debug else logits


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _folder_ds(root, split, max_samples=None):
    return ImageNetFolderDataset(
        root, split, transform=make_imagenet_eval_transform(),
        class_index_path=None, download_class_index=True, max_samples=max_samples, seed=42,
    )


def _loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=False)


def make_loaders(allowed, smoke=False):
    cap = 1000 if smoke else None
    clean_train = _folder_ds(CLEAN_DIR, "train", max_samples=cap)
    clean_val = _folder_ds(CLEAN_DIR, "val", max_samples=cap)
    allowed_set = set(allowed)
    ref_idx = [i for i, (_, t) in enumerate(clean_val.samples) if t in allowed_set]  # clean ref restricted to R's 200
    clean_ref = Subset(clean_val, ref_idx)
    r_eval = _folder_ds(R_ROOT, R_SPLIT, max_samples=cap)
    return _loader(clean_train, shuffle=False), _loader(clean_ref), _loader(r_eval)


def _row(variant, domain, m):
    return {"variant": variant, "domain": domain, "acc": m["acc"], "ece": m["ece"], "nll": m["nll"], "brier": m["brier"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE} smoke={args.smoke}")

    model = load_model()
    allowed, mask = imagenet_r_mask()
    print(f"[mask] {len(allowed)} ImageNet-R classes")
    clean_train, clean_ref, r_eval = make_loaders(allowed, smoke=args.smoke)
    print(f"[data] cache_train={len(clean_train.dataset)} clean_ref={len(clean_ref.dataset)} imagenet_r={len(r_eval.dataset)}")

    source_cache = build_labeled_cache(model, clean_train, device=DEVICE, key_space="z_pool", num_classes=NUM_CLASSES)
    print(f"[cache] {source_cache.size} keys (z_pool)")

    mmodel = MaskedModel(model, mask).to(DEVICE)
    rows = []
    # source (no retrieval), masked to 200
    rows.append(_row("source", "clean", evaluate_dememte(mmodel, clean_ref, device=DEVICE, corruption=None)))
    rows.append(_row("source", "imagenet_r", evaluate_dememte(mmodel, r_eval, device=DEVICE, corruption=None)))
    print("[source]", rows[-2]["acc"], "clean |", rows[-1]["acc"], "R")

    grid = ALPHA_GRID[:1] if args.smoke else ALPHA_GRID
    for alpha in grid:
        def make(a=alpha):
            adapter = RetrievalLogitAdapter(
                model,
                RetrievalConfig(key_space="z_pool", alpha_mode="fixed", alpha_max=a,
                                cache_source=True, episodic_size=0, top_k=16, beta=5.0),
                source_cache=source_cache, num_classes=NUM_CLASSES,
            )
            return MaskedRetrieval(adapter, mask)
        rows.append(_row(f"retrieval_z_pool_fixed_alpha@{alpha}", "clean",
                         evaluate_dememte_tta(make(), clean_ref, device=DEVICE, corruption=None,
                                              tta_method="retrieval_logit", tta_base_variant="imagenet_rn50")))
        rows.append(_row(f"retrieval_z_pool_fixed_alpha@{alpha}", "imagenet_r",
                         evaluate_dememte_tta(make(), r_eval, device=DEVICE, corruption=None,
                                              tta_method="retrieval_logit", tta_base_variant="imagenet_rn50")))
        print(f"[retrieval α={alpha}]", rows[-2]["acc"], "clean |", rows[-1]["acc"], "R")

    write_csv(rows, OUT / "e13_imagenet_r.csv")
    write_json({"allowed_classes": len(allowed), "alpha_grid": grid, "rows": rows}, OUT / "e13_imagenet_r.json")
    print(f"[done] wrote {OUT/'e13_imagenet_r.csv'}")


if __name__ == "__main__":
    main()
