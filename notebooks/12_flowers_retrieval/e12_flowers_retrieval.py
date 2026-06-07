"""wtq.5 — Port E11 retrieval-logit memory to Flowers + FT-clean baseline.

Two independent outputs, two CSVs (they feed C1 and C4 separately in THESIS.md):

  Output 1 (C1): FT-clean baseline  -> out/ft_clean_baseline.csv
  Output 2 (C4): E11 retrieval      -> out/e11_retrieval_flowers.csv

Reuses existing code only — no new algorithms. Tracking: bead Demente-wtq.5;
results write-up in notebooks/12_flowers_retrieval/insights.md.

Run:  PYTHONPATH=src uv run python notebooks/12_flowers_retrieval/e12_flowers_retrieval.py [--section all|ftclean|retrieval] [--smoke]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dememte.config import BaselineConfig, FlowersLegacyConfig, e6_config
from dememte.corruptions import STRICT_SUITE
from dememte.data import build_flowers_loaders
from dememte.evaluation import evaluate_baseline_suite, evaluate_dememte_suite, evaluate_dememte_tta_suite
from dememte.io import save_checkpoint, write_csv, write_json
from dememte.models.baseline import ResNetBaseline
from dememte.models.dememte import make_dememte_variant
from dememte.retrieval import RetrievalConfig, RetrievalLogitAdapter, build_labeled_cache
from dememte.training import train_baseline_phased

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATA_DIR = (HERE / "../../experiments/data").resolve().as_posix()
E6_CKPT = (HERE / "../06_e6_zq_alignment/out/e6_ema_kmeans_restart/best.pt").resolve()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS = 4
ALPHA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
SUMMARY_COLS = [
    "variant", "clean_acc", "corrupt_acc_avg",
    "ece_clean", "ece_corrupt_avg", "nll_clean", "nll_corrupt_avg",
    "brier_clean", "brier_corrupt_avg",
]


def _loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    return build_flowers_loaders(
        DATA_DIR, batch_size=batch_size, num_workers=num_workers,
        val_ratio=0.2, split_seed=42, protocol="historical_trainval_resplit", download=False,
    )


def _row(variant, m):
    return {k: (variant if k == "variant" else m.get(k)) for k in SUMMARY_COLS}


# --------------------------------------------------------------------------- #
# Output 1 — FT-clean baseline (feeds C1)
# --------------------------------------------------------------------------- #
def run_ft_clean(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te, meta = _loaders()
    cfg = BaselineConfig(
        dataset="flowers102", data_dir=DATA_DIR, num_classes=102,
        backbone_name="resnet18", backbone_out_channels=512, backbone_pretrained=True,
        freeze_backbone=False,          # full fine-tune
        train_corrupt_prob=0.0,         # CLEAN training — the whole point (vs nb04 FT-aug)
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,  # match _loaders so saved config reproduces
        device=DEVICE, out_dir=str(OUT), seed=42,
    )
    if smoke:
        cfg.epochs_warmup, cfg.epochs_corrupt, cfg.epochs_joint = 1, 1, 1
    model = ResNetBaseline(num_classes=102, freeze_backbone=False, backbone_name="resnet18", pretrained=True).to(DEVICE)
    model, best = train_baseline_phased(model, tr, va, cfg, DEVICE, verbose=True)
    save_checkpoint(model, OUT / "ft_clean.pt", extra={"best_val": best, "config": cfg.__dict__})
    m = evaluate_baseline_suite(model, te, device=DEVICE, suite=STRICT_SUITE)
    row = _row("ft_clean", m)
    write_csv([row], OUT / "ft_clean_baseline.csv")
    write_json({"meta": meta, "best_val": best, "summary": row}, OUT / "ft_clean_baseline.json")
    print("[ft_clean]", row)
    return row


# --------------------------------------------------------------------------- #
# Output 2 — E11 retrieval on Flowers (feeds C4)
# --------------------------------------------------------------------------- #
def _load_e6_flowers():
    cfg = e6_config("e6_ema_kmeans_restart", base=FlowersLegacyConfig())
    model = make_dememte_variant(cfg, device=DEVICE)
    payload = torch.load(E6_CKPT, map_location=DEVICE, weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)   # precondition: must load clean
    model.eval()
    return model, cfg


def run_retrieval(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te, meta = _loaders()
    model, cfg = _load_e6_flowers()
    print(f"[retrieval] E6 Flowers loaded (num_classes={cfg.num_classes}, quantizer={cfg.quantizer_type})")

    # source cache from CLEAN train, key = z_pool (C5 winner)
    source_cache = build_labeled_cache(model, tr, device=DEVICE, key_space="z_pool", num_classes=102)
    print(f"[retrieval] source cache built: {source_cache.size} keys")

    rows = []
    # reference row: E6 without retrieval
    m_src = evaluate_dememte_suite(model, te, device=DEVICE, suite=STRICT_SUITE)
    rows.append(_row("source", m_src))
    print("[retrieval] source:", rows[-1])

    grid = ALPHA_GRID[:1] if smoke else ALPHA_GRID
    for alpha in grid:
        def factory(a=alpha):
            return RetrievalLogitAdapter(
                model,
                RetrievalConfig(key_space="z_pool", alpha_mode="fixed", alpha_max=a,
                                cache_source=True, episodic_size=0, top_k=16, beta=5.0),
                source_cache=source_cache, num_classes=102,
            )
        m = evaluate_dememte_tta_suite(
            factory, te, device=DEVICE, suite=STRICT_SUITE,
            tta_method="retrieval_logit", tta_base_variant="e6_ema_kmeans_restart",
        )
        rows.append(_row(f"retrieval_z_pool_fixed_alpha@{alpha}", m))
        print("[retrieval]", rows[-1])

    write_csv(rows, OUT / "e11_retrieval_flowers.csv")
    write_json({"meta": meta, "alpha_grid": grid, "rows": rows}, OUT / "e11_retrieval_flowers.json")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=["all", "ftclean", "retrieval"], default="all")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    print(f"device={DEVICE} section={args.section} smoke={args.smoke}")
    if args.section in ("all", "retrieval"):
        run_retrieval(smoke=args.smoke)
    if args.section in ("all", "ftclean"):
        run_ft_clean(smoke=args.smoke)


if __name__ == "__main__":
    main()
