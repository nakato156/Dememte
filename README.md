# DeMemte VQSA

DeMemte is now a strict implementation of **Vector Quantization With Self-Attention for Quality-Independent Representation Learning** (CVPR 2023) adapted to the Flowers-102 robustness experiments in this repository.

The current model no longer uses the previous attractor/gate memory route. Vector quantization is the main representation path:

```text
image
  -> ResNet18 backbone
  -> feature map (B, 512, 7, 7)
  -> 1x1 projector: z (B, 256, 7, 7)
  -> VQ codebook: zq + vq/codebook/commitment losses
  -> GAP(z), GAP(zq)
  -> tokens [z_pool, zq_pool]
  -> self-attention blocks
  -> concat attended tokens (B, 512)
  -> MLP classifier (512 -> 512 -> num_classes)
```

Reference paper:

- Local PDF: `papers/Quantization_With_Self-Attention_for_Quality-Independent_Representation_Learning_CVPR_2023.pdf`
- Official CVF page: https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Vector_Quantization_With_Self-Attention_for_Quality-Independent_Representation_Learning_CVPR_2023_paper.html

## Repository Layout

- `src/dememte/models/dememte.py` — `DeMemteVQSA`, `VQSAFusion`, and public factories.
- `src/dememte/models/vq.py` — spatial vector quantizer and VQ losses.
- `src/dememte/training.py` — strict VQSA training loop plus ResNet baseline training.
- `src/dememte/evaluation.py` — clean/corrupt evaluation plus VQSA diagnostics.
- `src/dememte/config.py` — default VQSA config and paper-style ablations.
- `notebooks/` — experiment notebooks.
- `archive/notebooks/` — legacy notebooks kept only as historical material.

## Training Objective

For training, DeMemte builds a mixed batch:

```text
x_mixed = concat(clean_images, corrupt(clean_images))
y_mixed = concat(labels, labels)
```

The optimized loss is:

```text
loss = cross_entropy(logits, y_mixed) + vq_weight * vq_loss
vq_loss = codebook_loss + commitment_cost * commitment_loss
```

The default `E5Config` uses:

- `latent_dim = 256`
- `num_embeddings = 1024`
- `quantizer_type = "vq"`
- `vqsa_heads = 4`
- `vqsa_layers = 2`
- `vqsa_dropout = 0.1`
- `commitment_cost = 0.25`
- `vq_weight = 1.0`
- `train_corrupt_prob = 0.7`

E6 adds anti-collapse quantizer variants:

- `e6_paper_faithful` — original gradient-updated VQ control.
- `e6_zq_align_mse` — original VQ plus clean/corrupt `zq` alignment.
- `e6_ema_kmeans_restart` — EMA codebook updates, k-means initialization, and dead-code restarts.
- `e6_winner` — stable alias for the best VQ codebook variant, currently EMA + k-means + dead-code restarts.
- `e6_simvq_linear` — SimVQ-style codebook from a learned global linear transform.
- `e6_fsq` — finite scalar quantization without a learned lookup table.

## Ablations

The ablation registry now follows the paper's module breakdown:

- `vqsa_full` — codebook + concat fusion + self-attention.
- `no_codebook` — no VQ codebook, classifier sees duplicated original descriptors.
- `replace` — use the quantized descriptor as a replacement.
- `add` — add original and quantized descriptors.
- `concat_no_sa` — concatenate original and quantized descriptors without self-attention.

## Evaluation Metrics

Evaluation keeps the clean/corrupt classification metrics:

- `clean_acc`
- `corrupt_acc_avg`
- per-corruption accuracy
- `ece`, `nll`, `brier`

It also reports VQSA diagnostics:

- `vq_loss`
- `codebook_loss`
- `commitment_loss`
- `dq_mean`
- `assignment_entropy`
- `codebook_perplexity`
- `hard_usage`
- `hard_perplexity`
- `dead_code_fraction`
- `attention_entropy`

Gate-specific metrics such as `gate_mean`, `gate_raw`, `pareidolia_rate`, and prediction-change metrics are intentionally removed.

## Checkpoint Compatibility

This migration intentionally breaks compatibility with legacy attractor/gate checkpoints. Public factories such as `make_dememte_e5()` and `make_dememte_variant()` now build `DeMemteVQSA`; old checkpoints should not be loaded into the new model.
