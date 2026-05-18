# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DeMemte** is a PyTorch research project that augments a frozen ResNet18 backbone with a Vector Quantized Variational Autoencoder (VQ-VAE) memory module and an adaptive gating mechanism for robust image classification on Flowers-102 (102 classes).

All code lives in Jupyter notebooks — there are no Python modules, no build system, and no automated tests.

## Architecture

```
Image (224×224)
  → [ResNet18 (frozen)] → feature maps (B, 512, 7, 7)
       ├─ [AttentionSpatialVQVAE]
       │    ├─ 1×1 conv: 512 → 256
       │    ├─ Learnable positional embeddings (1, 256, 7, 7)
       │    ├─ 2-layer Transformer (self-attention over spatial tokens)
       │    ├─ VectorQuantizer2D (K=1024 codebook embeddings)
       │    ├─ 1×1 conv: 256 → 512
       │    └─ outputs: reconstructed_features, vq_loss, dq_map (per-location quantization error)
       │
       └─ [Adaptive Gate]
            ├─ EMA-normalizes dq_map (momentum=0.99)
            ├─ gate σ = sigmoid(α · (dq_norm − τ))  [α, τ learnable]
            └─ output: enhanced = (1−σ)·original + σ·reconstructed
  → GlobalAvgPool → Linear(512 → 102) → logits
```

The gate routes toward reconstructed features when quantization error is high (unusual/corrupted input) and toward original features when error is low (clean, well-represented input).

## Training Protocol

**Two phases, both with backbone frozen:**

**Phase 1 — VQ-VAE pre-training (4 epochs)**
- 35% of spatial features randomly masked (forces reconstruction)
- Loss: `0.5 · MSE(features, reconstructed) + 0.25 · vq_loss`
- Optimizer: AdamW (lr=3e-4, wd=1e-4)

**Phase 2 — Joint fine-tuning (10 epochs, patience=3)**
- Each batch processed twice: clean and corrupted
- Denoising target for corrupted pass: clean features
- Loss: `0.5(CE_clean + CE_corrupt) + 0.5·w_d·(MSE + MSE) + 0.5·w_vq·(vq_loss + vq_loss)` where w_d=0.5, w_vq=0.25
- Learning rates: VQ module=3e-4, classifier=1e-4
- Scheduler: ReduceLROnPlateau

**Corruption suite (applied with p=0.70 during training):**
- Gaussian noise: σ ∈ [0.4, 1.3]
- Pixel mask: ratio ∈ [0.20, 0.65]
- Cutout: ratio ∈ [0.20, 0.45]
- Blur: 7×7 box filter, ratio ∈ [0.30, 0.80]

## Key Files

| File | Purpose |
|------|---------|
| `experiments/NOTEBOOKS.md` | Active notebook lineage toward the final E5 model |
| `experiments/VQ/baseline.ipynb` | Fair ResNet18 vs DeMemte + VQ-VAE baseline reference |
| `experiments/VQ/dememte_variants.ipynb` | Reduced `dememte_transformer` implementation, the VQ starting point |
| `experiments/atracctor/attractor_memory.ipynb` | Main DeMemteAttractor experiment with E0-E5 screening |
| `experiments/atracctor/no_ood_debug.ipynb` | Isolated debug notebook for the no-OOD gate behavior |
| `experiments/atracctor/e5_final_clean.ipynb` | Final clean E5 notebook for checkpoint loading, metrics, gate curves, optional eval/retraining |
| `README.md` | Full methodology, results tables, and discussion (Spanish) |
| `explain.md` | Component-by-component architecture walkthrough |

Legacy, duplicate, or side-track notebooks live under `archive/notebooks/` and are not part of the active execution path.

## Running Code

All execution is notebook-cell-by-cell. No CLI commands. CUDA GPU is expected. Dependencies: `torch`, `torchvision`, `numpy`, `sklearn` — install manually if missing.

## Results Reference

| Metric | Baseline | DeMemte |
|--------|----------|---------|
| Clean accuracy | 67.44% | 74.92% (+7.48 pp) |
| Corrupt accuracy (avg) | 51.12% | 53.14% (+2.02 pp) |
| Parameters | 11.23M | 13.48M |

DeMemte underperforms on Gaussian noise (−1.14 pp) and pixel mask (−4.00 pp) — corruptions applied before the backbone hurt most because the codebook never sees the distorted distribution at training time.
