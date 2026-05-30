"""Configuration dataclasses for baseline, strict VQSA, and ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class BaselineConfig:
    """ImageNet baseline (frozen or finetuned)."""

    dataset: str = "imagenet_c"
    data_dir: str = "../../experiments/data/imagenet-c-subset"
    num_classes: int = 1000
    batch_size: int = 16
    num_workers: int = 2
    device: str = "cuda"
    backbone_name: str = "resnet50"
    backbone_pretrained: bool = True
    backbone_out_channels: int = 2048

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

    dataset: str = "imagenet_c"
    data_dir: str = "../../experiments/data/imagenet-c-subset"
    num_classes: int = 1000
    batch_size: int = 16
    num_workers: int = 2
    device: str = "cuda"
    backbone_name: str = "resnet50"
    backbone_pretrained: bool = True
    backbone_out_channels: int = 2048

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
    quantizer_type: str = "vq"
    vq_ema_decay: float = 0.99
    vq_kmeans_init: bool = False
    vq_kmeans_steps: int = 10
    dead_code_restart: bool = False
    dead_code_restart_after_epoch: int = 1
    dead_code_threshold: float = 1.0
    dead_code_restart_jitter: float = 0.01
    fsq_levels: int = 8
    vqsa_heads: int = 4
    vqsa_layers: int = 2
    vqsa_dropout: float = 0.1
    vqsa_fusion_mode: str = "concat"
    vqsa_use_codebook: bool = True
    vqsa_use_self_attention: bool = True
    vqsa_train_backbone: bool = False
    vqsa_align_mode: str = "none"
    align_weight: float = 0.0
    train_corrupt_prob: float = 0.7

    out_dir: str = "./out"
    seed: int = 42


@dataclass
class E6Config(E5Config):
    """E6 VQSA experiment configuration with optional clean/corrupt zq alignment."""

    variant_name: str = "e6_paper_faithful"
    variant_label: str = "E6 paper-faithful VQSA"


@dataclass
class AblationConfig(E5Config):
    """Single VQSA ablation variant."""

    variant_name: str = "vqsa_full"
    variant_label: str = "VQSA full"


@dataclass
class FlowersLegacyConfig(E5Config):
    """Legacy Flowers-102 settings kept for reproducing pre-ImageNet runs."""

    dataset: str = "flowers102"
    data_dir: str = "../../experiments/data"
    num_classes: int = 102
    backbone_name: str = "resnet18"
    backbone_out_channels: int = 512


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


E6_SPECS = {
    "e6_paper_faithful": {
        "label": "E6 paper-faithful VQSA",
        "overrides": {"vqsa_align_mode": "none", "align_weight": 0.0},
    },
    "e6_zq_align_mse": {
        "label": "E6 zq MSE alignment",
        "overrides": {"vqsa_align_mode": "zq_mse", "align_weight": 0.1},
    },
    "e6_ema_kmeans_restart": {
        "label": "E6 EMA VQ + k-means init + dead-code restart",
        "overrides": {
            "quantizer_type": "ema_vq",
            "vq_kmeans_init": True,
            "vq_kmeans_steps": 10,
            "dead_code_restart": True,
            "dead_code_restart_after_epoch": 1,
            "dead_code_threshold": 1.0,
            "dead_code_restart_jitter": 0.01,
            "vqsa_align_mode": "none",
            "align_weight": 0.0,
        },
    },
    "e6_winner": {
        "label": "E6 winner: EMA VQ + k-means init + dead-code restart",
        "overrides": {
            "quantizer_type": "ema_vq",
            "vq_kmeans_init": True,
            "vq_kmeans_steps": 10,
            "dead_code_restart": True,
            "dead_code_restart_after_epoch": 1,
            "dead_code_threshold": 1.0,
            "dead_code_restart_jitter": 0.01,
            "vqsa_align_mode": "none",
            "align_weight": 0.0,
        },
    },
    "e6_simvq_linear": {
        "label": "E6 SimVQ linear codebook",
        "overrides": {
            "quantizer_type": "simvq_linear",
            "vqsa_align_mode": "none",
            "align_weight": 0.0,
        },
    },
    "e6_fsq": {
        "label": "E6 finite scalar quantization",
        "overrides": {
            "quantizer_type": "fsq",
            "fsq_levels": 8,
            "vqsa_align_mode": "none",
            "align_weight": 0.0,
        },
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


def e6_config(variant_name: str, base: Optional[E5Config] = None) -> E6Config:
    """Return an E6Config pre-configured for a paper-faithful or aligned E6 variant."""
    if variant_name not in E6_SPECS:
        raise KeyError(f"Unknown E6 variant: {variant_name!r}. Choices: {list(E6_SPECS)}")
    spec = E6_SPECS[variant_name]
    base_dict = asdict(base if base is not None else E6Config())
    base_dict["variant_name"] = variant_name
    base_dict["variant_label"] = spec["label"]
    for k, v in spec["overrides"].items():
        base_dict[k] = v
    return E6Config(**base_dict)


def resolve_data_dir(config) -> str:
    """Resolve the configured dataset root, preferring ImageNet-C by default."""
    dataset = getattr(config, "dataset", "imagenet_c")
    candidates = [
        config.data_dir,
        "../../experiments/data/imagenet-c-subset",
        "../experiments/data/imagenet-c-subset",
        "experiments/data/imagenet-c-subset",
        "../../experiments/data",
        "../experiments/data",
        "experiments/data",
        "./data",
    ]
    for c in candidates:
        p = Path(c).expanduser().resolve()
        if dataset == "flowers102" and ((p / "flowers-102").exists() or p.name == "flowers-102"):
            return str(p)
        if dataset == "imagenet_c" and (
            p.name == "imagenet-c-subset" or any((p / corr).exists() for corr in ("gaussian_noise", "motion_blur", "pixelate", "jpeg_compression"))
        ):
            return str(p)
    return str(Path(candidates[0]).expanduser().resolve())
