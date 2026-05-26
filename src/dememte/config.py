"""Configuration dataclasses for baseline, strict VQSA, and ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class BaselineConfig:
    """ResNet18 baseline (frozen or finetuned)."""

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
    """Strict DeMemte VQSA configuration."""

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
    weight_decay: float = 1e-4

    epochs_vqsa_max: int = 10

    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    latent_dim: int = 256
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    vq_temperature: float = 1.0
    vq_weight: float = 1.0
    vqsa_heads: int = 4
    vqsa_layers: int = 2
    vqsa_dropout: float = 0.1
    vqsa_fusion_mode: str = "concat"
    vqsa_use_codebook: bool = True
    vqsa_use_self_attention: bool = True
    vqsa_train_backbone: bool = False
    train_corrupt_prob: float = 0.7

    out_dir: str = "./out"
    seed: int = 42


@dataclass
class AblationConfig(E5Config):
    """Single VQSA ablation variant."""

    variant_name: str = "vqsa_full"
    variant_label: str = "VQSA full"


# -- Ablation registry --------------------------------------------------------

ABLATION_SPECS = {
    "vqsa_full": {
        "label": "VQSA full",
        "overrides": {},
    },
    "no_codebook": {
        "label": "Ablation no codebook",
        "overrides": {"vqsa_use_codebook": False, "vq_weight": 0.0},
    },
    "replace": {
        "label": "Ablation VQ replace fusion",
        "overrides": {"vqsa_fusion_mode": "replace", "vqsa_use_self_attention": False},
    },
    "add": {
        "label": "Ablation VQ add fusion",
        "overrides": {"vqsa_fusion_mode": "add", "vqsa_use_self_attention": False},
    },
    "concat_no_sa": {
        "label": "Ablation concat without self-attention",
        "overrides": {"vqsa_fusion_mode": "concat", "vqsa_use_self_attention": False},
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
