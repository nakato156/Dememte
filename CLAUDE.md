# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DeMemte** is a PyTorch research project that augments a ResNet18 backbone with a Vector Quantized VAE memory module, an attractor (pattern completion residual MLP) and an adaptive multi-signal gate for robust image classification on Flowers-102 (102 classes).

The central thesis the codebase is set up to defend: the standard recipe for robustness (data augmentation with corruptions + full fine-tuning) buys corrupt accuracy at the cost of clean accuracy. DeMemte's frozen-backbone + memory-gate approach is designed to break that trade-off.

Code is split between a reusable Python module (`src/dememte/`) and 4 thin orchestration notebooks under `notebooks/`. Older exploratory notebooks live under `archive/notebooks/`.

## Architecture (E5 winner)

```
Image (224×224)
  → [ResNet18 (frozen)] → feats (B, 512, 7, 7)
       ├─ aux_classifier(feats) → aux_logits         (used for gate uncertainty signal)
       └─ projector(1×1)        → z (B, 128, 7, 7)
              ├─ VectorQuantizer2D (K=1024)          → zq, vq_loss, dq_map, soft_assign
              ├─ AttractorMemory (residual MLP)      → z_completed = z + MLP(z)
              ├─ AmbiguityGate(aux_logits, dq_map, soft_assign)
              │       inputs: [uncertainty, familiarity, conflict, 1-ood_risk]
              │       → gate ∈ [0, 1]
              └─ z_final = z + gate · (z_completed − z)
                  → unprojector(1×1) → feats_final
                  → classifier(GAP(feats_final))   → logits (B, 102)
```

Gate signals:
- `uncertainty` — softmax entropy of `aux_logits`
- `familiarity` — Gaussian over EMA-normalized dq score (midpoint, log_width learnable)
- `conflict` — entropy of soft VQ assignment
- `ood_risk` — sigmoid(ood_beta · (dq_norm − ood_tau))

E5 overrides: `ood_tau=1.5`, `ood_beta=8.0`, `familiarity_width=0.5`, `gate_dropout=0.1`, `lr_gate=1e-4`, `phase3_lock_familiarity=True`, `gate_raw_entropy_reg=0.01`.

## Training Protocol (3 phases, backbone frozen)

**Phase 1 — Latent pretrain (≤4 epochs)**: 35% feature dropout, loss `0.5·MSE + 0.25·vq_loss + 0.01·sigreg`, AdamW lr=3e-4.

**Phase 2 — Attractor + gate + classifier (≤6 epochs)**: train on clean + dirty (corruption p=0.7), loss = `MSE_dirty + 0.3·MSE_clean + CE_clean + entropy_reg`, optimizer groups: attractor lr=3e-4, gate lr=1e-4, classifiers lr=1e-4.

**Phase 3 — Joint refine (≤10 epochs)**: clean+dirty pairs, loss `0.5(CE_clean + CE_dirty) + 0.5·w_d·(MSE+MSE) + 0.5·w_vq·(vq+vq) + antipareidolia + entropy_reg`. ReduceLROnPlateau, early stop patience=3.

**Corruption suite** (training: `apply_train_corruption`, p=0.7. Eval: `apply_eval_corruption`, fixed grid 4×3):
- gaussian_noise: σ ∈ [0.4, 1.3] (train); eval {0.5, 1.0, 1.5}
- pixel_mask: ratio ∈ [0.20, 0.65] (train); eval {0.25, 0.5, 0.75}
- cutout: ratio ∈ [0.20, 0.45] (train); eval {0.2, 0.35, 0.5}
- blur: 7×7 box, ratio ∈ [0.30, 0.80] (train); eval {0.35, 0.6, 0.85}

## Repo layout

```
src/dememte/                    Shared library (single source of truth)
  config.py                       BaselineConfig, E5Config, AblationConfig, ABLATION_SPECS
  data.py                         Flowers102 loaders (historical_trainval_resplit, seed=42)
  corruptions.py                  apply_train_corruption, apply_eval_corruption, STRICT_SUITE
  models/{baseline,vq,attractor,dememte}.py
  training.py                     train_phase1/2/3, train_dememte_full, train_baseline_phased
  evaluation.py                   evaluate_{dememte,baseline}_suite, signal_curve_rows
  io.py                           save/load checkpoint, write_json, write_csv

notebooks/                      Thin orchestration notebooks (each with its own out/)
  01_baseline/baseline.ipynb           ResNet18 frozen, 3-stage methodology, fair 1:1 baseline
  02_e5_winner/e5_winner.ipynb         DeMemte E5 reproduction end-to-end
  03_ablations/ablations.ipynb         8 critical-set variants (clean + corrupt)
  04_finetune_vs_frozen/...            ResNet18 FT (heavy aug) vs DeMemte E5 frozen

experiments/                    Legacy artifacts preserved as reference
  data/flowers-102/                    Dataset (downloaded)
  atracctor/out/artifacts/             Pre-computed E5 critical run (5 seeds × 8 variants)

archive/notebooks/              Original notebooks (no longer in active path)
scripts/build_notebooks.py      One-shot generator for the 4 notebooks

CLAUDE.md, README.md, requirements.txt, .gitignore
```

## Running Code

All execution is notebook-cell-by-cell. Each notebook follows the same skeleton: Config → Data → Model → (train if `RUN_TRAINING=True`, else load checkpoint) → Evaluate clean+corrupt → Persist outputs to `out/`.

CUDA GPU required. Dependencies in `requirements.txt`. Build the notebooks from source with `python scripts/build_notebooks.py` (also useful when changing prompts/structure).

The E5 winner notebook will, on first run with `RUN_TRAINING=False`, copy the legacy checkpoint from `experiments/atracctor/out/artifacts/dememte_e5_critical/seed_42/.../*best.pt` into `notebooks/02_e5_winner/out/e5_best.pt`. Same trick for ablations (notebook 03).

**Parity check** (run after touching `src/dememte/models/`): load the legacy checkpoint with the new module and confirm `clean_acc≈0.811839`. The current refactor reproduces this number to 6 decimals.

## Results Reference (test set, historical_trainval_resplit, seed=42)

E5 winner against the legacy critical run:

| Metric | E5 (reference) |
|---|---|
| Clean accuracy | 0.8118 |
| Corrupt accuracy (avg) | 0.5213 |
| Gate mean (clean) | 0.2348 |
| Gate order margin (blur/cutout − gauss-heavy) | 0.0920 |
| Harmful changes | 0.0046 |

The notebook 04 comparison demonstrates the trade-off-breaking thesis: ResNet18 FT with heavy augmentation moves along the clean↔corrupt frontier, while DeMemte frozen+gate aims off-frontier.
