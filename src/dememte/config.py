"""Configuration dataclasses for baseline, E5, and ablation experiments.

The defaults mirror the values used in the original `Dememte_e5y.ipynb` so that
checkpoints produced under the legacy code load and reproduce the saved metrics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class BaselineConfig:
    """ResNet18 baseline (frozen or finetuned). Reuses the E5 phase schedule."""

    data_dir: str = "../../experiments/data"
    num_classes: int = 102
    batch_size: int = 16
    num_workers: int = 2
    device: str = "cuda"

    val_ratio: float = 0.2
    split_seed: int = 42
    benchmark_protocol: str = "historical_trainval_resplit"

    freeze_backbone: bool = True
    backbone_lr: float = 1e-4

    lr_cls: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4

    epochs_warmup: int = 3
    epochs_corrupt: int = 6
    epochs_joint: int = 10

    train_corrupt_prob: float = 0.7

    out_dir: str = "./out"
    seed: int = 42


@dataclass
class E5Config:
    """DeMemte E5 winner configuration — the final reproducible variant."""

    data_dir: str = "../../experiments/data"
    num_classes: int = 102
    batch_size: int = 16
    num_workers: int = 2
    device: str = "cuda"

    val_ratio: float = 0.2
    split_seed: int = 42
    benchmark_protocol: str = "historical_trainval_resplit"

    lr_vq: float = 3e-4
    lr_cls: float = 1e-4
    lr_attractor: float = 3e-4
    lr_gate: float = 1e-4
    weight_decay: float = 1e-4

    epochs_phase1_max: int = 4
    epochs_phase2_max: int = 6
    epochs_phase3_max: int = 10

    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    latent_dim: int = 128
    attractor_hidden: int = 512
    gate_hidden: int = 16
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    vq_temperature: float = 1.0

    familiarity_midpoint: float = 0.0
    familiarity_width: float = 0.5
    ood_tau: float = 1.5
    ood_beta: float = 8.0

    denoise_weight: float = 0.5
    vq_weight: float = 0.25
    antipareidolia_weight: float = 0.1
    gate_entropy_reg: float = 0.01
    gate_raw_entropy_reg: float = 0.01
    weak_sigreg_weight: float = 0.01
    weak_sigreg_sketch_dim: int = 64
    gate_init_prob: float = 0.1
    gate_prior_floor: float = 0.02
    gate_dropout: float = 0.1

    masked_feature_ratio: float = 0.35
    train_corrupt_prob: float = 0.7

    phase3_memory_grad_mode: str = "full"
    phase3_lock_familiarity: bool = True
    phase3_backbone_train_mode: str = "frozen"
    gate_order_loss_weight: float = 0.0
    gate_order_loss_margin: float = 0.03
    gate_order_gauss_severity: float = 1.5
    gate_order_blur_severity: float = 0.6
    gate_order_cutout_severity: float = 0.35

    use_uncertainty: bool = True
    use_familiarity: bool = True
    use_conflict: bool = True
    use_ood: bool = True
    disable_attractor: bool = False

    out_dir: str = "./out"
    seed: int = 42


@dataclass
class AblationConfig(E5Config):
    """Single variant within the critical ablation set. Inherits all E5 defaults."""

    variant_name: str = "e5_combined_dropout_ood_tau_150"
    variant_label: str = "E5 combined dropout + OOD tau 1.50"


# -- Ablation registry --------------------------------------------------------

ABLATION_SPECS = {
    "e5_combined_dropout_ood_tau_150": {
        "label": "E5 combined dropout + OOD tau 1.50",
        "overrides": {},
    },
    "no_ood": {
        "label": "Ablation no OOD",
        "overrides": {"use_ood": False},
    },
    "no_familiarity": {
        "label": "Ablation no familiarity",
        "overrides": {"use_familiarity": False},
    },
    "no_antipareidolia": {
        "label": "Ablation no anti-pareidolia",
        "overrides": {"antipareidolia_weight": 0.0},
    },
    "freeze_vq_phase3": {
        "label": "Ablation freeze VQ Phase 3",
        "overrides": {"phase3_memory_grad_mode": "freeze_vq"},
    },
    "partial_unfreeze_backbone": {
        "label": "Ablation partial backbone unfreeze",
        "overrides": {"phase3_backbone_train_mode": "partial_unfreeze"},
    },
    "attractor_disabled": {
        "label": "Ablation attractor disabled",
        "overrides": {"disable_attractor": True},
    },
    "resnet18_transfer_baseline": {
        "label": "ResNet18 transfer baseline",
        "overrides": {},  # bypass: handled by notebook as pure baseline
    },
}


def ablation_config(variant_name: str, base: Optional[E5Config] = None) -> AblationConfig:
    """Return an AblationConfig pre-configured for a given variant name."""
    if variant_name not in ABLATION_SPECS:
        raise KeyError(f"Unknown variant: {variant_name!r}. Choices: {list(ABLATION_SPECS)}")
    spec = ABLATION_SPECS[variant_name]
    base_dict = asdict(base if base is not None else E5Config())
    base_dict["variant_name"] = variant_name
    base_dict["variant_label"] = spec["label"]
    for k, v in spec["overrides"].items():
        base_dict[k] = v
    return AblationConfig(**base_dict)


def resolve_data_dir(config) -> str:
    """Find the Flowers102 root, accepting common relative paths from notebooks."""
    candidates = [
        config.data_dir,
        "../../experiments/data",
        "../experiments/data",
        "experiments/data",
        "./data",
    ]
    for c in candidates:
        p = Path(c).expanduser().resolve()
        if (p / "flowers-102").exists() or p.name == "flowers-102":
            return str(p)
    return str(Path(candidates[0]).expanduser().resolve())
