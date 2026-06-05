# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DeMemte** is a PyTorch research project that augments a frozen ResNet18 backbone with a **Vector Quantization + Self-Attention (VQSA)** memory module for robust image classification on Flowers-102 (102 classes). The implementation is adapted from *Vector Quantization With Self-Attention for Quality-Independent Representation Learning* (Yang et al., CVPR 2023).

The thesis the codebase is set up to defend: the standard recipe for robustness (data augmentation + full fine-tuning) buys corrupt accuracy at the cost of clean accuracy. DeMemte's **frozen-backbone + VQSA memory** approach is designed to break that trade-off, and to be a substrate for *test-time pattern completion* (the project's "memory" claim, Q5 in `RESPONSES.md`).

**Important — history**: an earlier version of this project used an `AttractorMemory + AmbiguityGate` route (entropy / familiarity / conflict / OOD-risk signals). That route has been **entirely removed**; only legacy artefacts remain under `experiments/atracctor/` and `archive/notebooks/`. Old attractor/gate checkpoints are not loadable into the current model. If you see references to `gate_mean`, `pareidolia_rate`, `AttractorMemory`, `AmbiguityGate`, `ood_tau`, etc., assume the source is stale.

Code is split between a reusable Python module (`src/dememte/`) and orchestration notebooks under `notebooks/`.

## Architecture (current = VQSA)

```
Image (224×224)
  → [ResNet18 (frozen)]            → feats   (B, 512, 7, 7)
  → projector (1×1 conv → BN → GELU) → z   (B, 256, 7, 7)
  → VectorQuantizer2D (K=1024)     → zq, vq_loss, codebook_loss, commitment_loss,
                                     dq_map, soft_assign, encoding_indices
  → GAP(z), GAP(zq)                → z_pool, zq_pool   (B, 256)
  → tokens = stack([z_pool, zq_pool], dim=1)   (B, 2, 256)
  → SelfAttentionBlock × vqsa_layers          (heads=4, mlp_ratio=4, dropout=0.1)
  → flatten                                    (B, 512)
  → classifier: Linear(512→512) → GELU → Dropout → Linear(512→num_classes)
```

Fusion mode (`vqsa_fusion_mode`) controls how the two tokens are built:
- `concat`: `[z_pool, zq_pool]` (default)
- `replace`: `[zq_pool, zq_pool]`
- `add`: `[z_pool + zq_pool, z_pool + zq_pool]`

Quantizer types (`quantizer_type` in config / `make_quantizer2d` in `src/dememte/models/vq.py`):
- `vq` — vanilla VQ-VAE codebook (gradient-updated).
- `ema_vq` — EMA-updated codebook, supports k-means init and dead-code restart.
- `simvq_linear` — SimVQ-style: a small parameter codebook passed through a learned linear `codebook_transform` (≈16k params on the transform); used in E7c-A because the transform is a batch-agnostic adaptation surface.
- `fsq` — finite scalar quantization (lookup-free).

## Training Protocol (VQSA, backbone frozen)

Single loop, **not** the legacy 3-phase recipe. Each step builds a mixed batch:

```
x_mixed = concat(clean_images, corrupt_train(clean_images))     # p=0.7 per image
y_mixed = concat(labels,        labels)
loss    = cross_entropy(logits, y_mixed) + vq_weight · vq_loss
          [+ align_weight · alignment_loss   if vqsa_align_mode != "none"]
vq_loss = codebook_loss + commitment_cost · commitment_loss
```

Defaults (`E5Config` in `src/dememte/config.py`):
- `latent_dim=256`, `num_embeddings=1024`, `commitment_cost=0.25`, `vq_weight=1.0`
- `vqsa_heads=4`, `vqsa_layers=2`, `vqsa_dropout=0.1`, `vqsa_fusion_mode="concat"`
- `lr_vq=3e-4`, `lr_cls=1e-4`, `weight_decay=1e-4`, optimizer AdamW
- `epochs_vqsa_max=10`, `ReduceLROnPlateau` (`factor=0.5`, `patience=2`), early stop `patience=3`
- `train_corrupt_prob=0.7`
- Backbone frozen by default (`vqsa_train_backbone=False`); backbone stays in `eval()` so BN running stats don't drift.

Driver: `train_dememte_vqsa(model, train_loader, val_loader, config, device)` (alias `train_dememte_full`). Baselines use `train_baseline_phased`.

**Corruption suite** (`src/dememte/corruptions.py`):
- Training: `apply_train_corruption`, p=0.7.
- Eval: `apply_eval_corruption`, fixed 4×3 grid (`STRICT_SUITE`).
- gaussian_noise: train σ ∈ [0.4, 1.3]; eval {0.5, 1.0, 1.5}
- pixel_mask: train ratio ∈ [0.20, 0.65]; eval {0.25, 0.5, 0.75}
- cutout: train ratio ∈ [0.20, 0.45]; eval {0.2, 0.35, 0.5}
- blur: train 7×7 box, ratio ∈ [0.30, 0.80]; eval {0.35, 0.6, 0.85}

## Repo layout

```
src/dememte/                    Single source of truth for models, training, eval, TTA
  config.py                       BaselineConfig, E5Config, E6Config, AblationConfig,
                                  ABLATION_SPECS, E6_SPECS, ablation_config(), e6_config()
  data.py                         Flowers102 loaders (historical_trainval_resplit, seed=42)
  corruptions.py                  apply_train_corruption, apply_eval_corruption, STRICT_SUITE
  models/
    baseline.py                     ResNet18 backbone factory + frozen/FT baseline
    vq.py                           VectorQuantizer2D, EMAVectorQuantizer2D,
                                    SimVQLinearQuantizer2D, FSQQuantizer2D,
                                    make_quantizer2d, LatentProjector
    dememte.py                      DeMemteVQSA, VQSAFusion, SelfAttentionBlock,
                                    make_dememte_e5 / e6 / variant
  training.py                     train_dememte_vqsa / train_dememte_full,
                                  train_baseline_phased, configure_vqsa_training,
                                  initialize_vqsa_codebook (k-means + dead-code restart)
  evaluation.py                   evaluate_dememte, evaluate_dememte_suite,
                                  evaluate_dememte_tta, evaluate_dememte_tta_suite,
                                  evaluate_baseline_suite, signal_curve_rows*
  tta.py                          Test-time adaptation surface (E7 / E7b / E7c):
                                    TentAdapter, EATALiteAdapter, NoUpdateAdapter,
                                    MemoryTentAdapter, SourceFilterEATAAdapter,
                                    SoftAssignTentAdapter, CodebookLossAdapter,
                                    AlphaBNStatsAdapter,
                                    configure_tta_{model,layernorm,codebook},
                                    collect_tta_{bn,ln,codebook}_params,
                                    latent_memory_loss, softmax_entropy
  io.py                           save/load checkpoint, write_json, write_csv

notebooks/                      Orchestration notebooks (each with its own out/)
  01_baseline/baseline.ipynb            ResNet18 frozen + FT baselines, 3-stage methodology
  02_e5_winner/e5_winner.ipynb          DeMemte VQSA reference run (vqsa_full / E5Config)
  03_ablations/ablations.ipynb          5 paper-style VQSA ablations
  04_finetune_vs_frozen/...             ResNet18 FT (heavy aug) vs DeMemte VQSA frozen
  05_oracle_gate/oracle_gate.ipynb      Post-hoc oracle: gate-style upper bound on combining
                                        baseline and memory predictions
  06_e6_zq_alignment/e6_zq_alignment.ipynb
                                        Anti-collapse quantizer sweep (E6_SPECS); produces
                                        the E7/E7b/E7c base checkpoint (e6_ema_kmeans_restart)
                                        and a SimVQ checkpoint used by E7c-A
  07_e7_tta/e7_tta.ipynb                E7 v1 TTA: TENT-BN + EATA-Lite over E6 winner
  08_e7b_tta/e7b_tta.ipynb              E7b: LayerNorm + memory-anchored TTA (conservative)
  09_e7c_codebook/e7c_codebook.ipynb    E7c-A: codebook plasticity (SimVQ codebook_transform)

experiments/                    Legacy artefacts preserved as reference
  data/flowers-102/                     Dataset (downloaded)
  atracctor/                            Old attractor/gate runs (do NOT load into VQSA)

archive/notebooks/              Original attractor/gate notebooks (out of active path)
tests/test_vqsa.py              30 tests covering VQ math, training step, TTA adapters
papers/                         Local PDFs of cited works
RESPONSES.md                    Methodology Q&A (Q1–Q5) — the reasoning behind E7/E7b/E7c
NOTES.md, VALIDATIONS.md, HANDOFF.md   Working notes / handoff scratchpads
inform.tex, inform.pdf          Project report
CLAUDE.md, AGENTS.md, README.md, pyproject.toml, uv.lock, requirements-scraper.txt, .gitignore
```

There is no `scripts/build_notebooks.py` in the current tree — edit notebooks directly.

## Running Code

All execution is notebook-cell-by-cell. CUDA GPU required. Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`); set up the environment with `uv sync --extra dev`. Each notebook follows: Config → Data → Model → (train if `RUN_TRAINING=True`, else load checkpoint from its `out/`) → Evaluate clean+corrupt → Persist to `out/`.

E7 / E7b / E7c notebooks **require** an E6 checkpoint on disk. They read from `notebooks/06_e6_zq_alignment/out/<variant>/best.pt` — train E6 first (or restore those checkpoints) before running TTA notebooks. E7c-A specifically reads the SimVQ variant; the others read `e6_ema_kmeans_restart`.

Run the tests with:

```bash
uv run pytest          # uv-managed env (package installed editable from src/)
# or, against any interpreter: PYTHONPATH=src python -m pytest
```

## Configs and variants

`E5Config` is the strict VQSA default. `E6Config` extends it with `variant_name` / `variant_label` and is used with `e6_config(name)` to apply one of:

- `e6_paper_faithful` — vanilla VQ, no alignment.
- `e6_zq_align_mse` — vanilla VQ + clean→corrupt `zq` MSE alignment (`align_weight=0.1`).
- `e6_ema_kmeans_restart` — EMA VQ + k-means init + dead-code restart (the **E6 winner**).
- `e6_winner` — stable alias for the current winning recipe (same as `e6_ema_kmeans_restart`).
- `e6_simvq_linear` — SimVQ codebook (the base for E7c-A).
- `e6_fsq` — finite scalar quantization (lookup-free).

`AblationConfig` + `ABLATION_SPECS` cover the paper-style VQSA ablations: `vqsa_full`, `no_codebook`, `replace`, `add`, `concat_no_sa`.

## Results Reference (test set, `historical_trainval_resplit`, seed=42)

E6 sweep (`notebooks/06_e6_zq_alignment/out/e6_summary.md`):

| variant | quantizer | clean_acc | corrupt_acc_avg | hard_usage_clean | dead_code_fraction_clean |
|---|---|---:|---:|---:|---:|
| `e6_ema_kmeans_restart` ★ winner | ema_vq | **0.7523** | 0.5030 | 0.738 | 0.262 |
| `e6_fsq` | fsq | 0.7413 | **0.5039** | 0.500 | 0.500 |
| `e6_paper_faithful` (= E5 reference, `vqsa_full`) | vq | 0.7361 | 0.4957 | 0.002 | 0.998 |
| `e6_simvq_linear` (E7c-A base) | simvq_linear | 0.7317 | 0.4807 | 0.011 | 0.989 |
| `e6_zq_align_mse` | vq | 0.7161 | 0.4739 | 0.002 | 0.998 |

Hard codebook usage on vanilla VQ is ~0.2% (severe collapse); EMA + k-means + dead-code restart restores ~74% usage at materially better accuracy — this is the headline E6 result.

E7 / E7b / E7c headline (see `notebooks/0{7,8,9}_.../insights.md` for full numbers):
- **E7 v1 (BN surface)**: `tent_bn` and `eata_lite_*` collapse to ~3% clean / ~3% corrupt — switching BN to batch stats blows up at this batch size.
- **E7b (LayerNorm surface, memory-anchored)**: all conservative variants land within ±0.003 of source. Adaptation is structurally inert because `z` / `zq` are upstream of the only LayerNorm parameters, so `latent_memory_loss` has zero gradient onto LN — documented as a useful negative result.
- **E7c-A (SimVQ codebook plasticity)**: codebook drift becomes non-zero (`tent_codebook_softassign` → `zq_drift_corrupt=0.0036`); memory-regularized variants pin drift to exactly 0 (memory dominates). Accuracy stays within ±0.001 of source under conservative hyperparameters — Q5 is *mechanistically* validated but not yet *numerically*.

When updating numbers, prefer the CSV/JSON sources in each `out/` over copy-pasted tables in markdown.

## Test-time adaptation surfaces (mental model for E7*)

The codebook is the project's "memory". The TTA experiments are organised around *which surface upstream of / inside / downstream of the codebook is being adapted*:

- **BN affine / running stats** (E7 v1, `AlphaBNStatsAdapter`) — historically the standard surface but here too fragile.
- **LayerNorm affine inside SelfAttentionBlock** (E7b, `configure_tta_layernorm`) — batch-agnostic, safe, but downstream of `z/zq` so `latent_memory_loss` is a structural no-op.
- **SimVQ `codebook_transform.weight`** (E7c-A, `configure_tta_codebook`) — the codebook itself, batch-agnostic, with live gradient through `soft_assign` and `codebook_loss` (but **not** through `q_st = z + (q − z).detach()`, hence why `SoftAssignTentAdapter` / `CodebookLossAdapter` are the variants that actually move the codebook, while a pure `TentAdapter` on the codebook is provably inert — there is a regression test for this).

`MemoryTentAdapter` and `SourceFilterEATAAdapter` add `latent_memory_loss` (`MSE(z, z_src) + MSE(zq, zq_src) + KL(p_src ‖ p)`) against a frozen *source teacher* (à la EcoTTA) — this is what we mean by "memory preservation" in the project narrative.

## Conventions

- All experiment scripts assume `seed=42` and the `historical_trainval_resplit` protocol; changing either breaks comparability with the reference table above.
- Notebooks read `RUN_TRAINING` from a top cell — flip to `False` to evaluate from existing `out/best.pt` / `out/vqsa_best.pt`.
- Diagnostics reported by `evaluate_dememte*`: `clean_acc`, `corrupt_acc_avg`, per-corruption accuracy, `ece`, `nll`, `brier`, plus VQSA-specific `vq_loss`, `codebook_loss`, `commitment_loss`, `dq_mean`, `assignment_entropy`, `codebook_perplexity`, `hard_usage`, `hard_perplexity`, `dead_code_fraction`, `attention_entropy`. Gate-era metrics (`gate_mean`, `pareidolia_rate`, prediction-change counts) are gone — don't reintroduce them.
- TTA evaluators additionally report `tta_updates`, `tta_selection_rate`, and (for E7c) `zq_drift`, `assignment_churn`, `kl_src` against a frozen teacher.


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
