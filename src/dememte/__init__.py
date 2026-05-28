"""DeMemte: strict VQSA models, baselines, data, and experiment helpers."""

from .config import BaselineConfig, E5Config, E6Config, AblationConfig
from .data import build_loaders, seed_everything
from .tta import (
    EATALiteAdapter,
    MemoryTentAdapter,
    NoUpdateAdapter,
    SourceFilterEATAAdapter,
    TentAdapter,
    collect_tta_bn_params,
    collect_tta_ln_params,
    configure_tta_layernorm,
    configure_tta_model,
    latent_memory_loss,
    make_tta_optimizer,
    softmax_entropy,
)

__all__ = [
    "BaselineConfig",
    "E5Config",
    "E6Config",
    "AblationConfig",
    "build_loaders",
    "seed_everything",
    "TentAdapter",
    "EATALiteAdapter",
    "NoUpdateAdapter",
    "MemoryTentAdapter",
    "SourceFilterEATAAdapter",
    "collect_tta_bn_params",
    "collect_tta_ln_params",
    "configure_tta_model",
    "configure_tta_layernorm",
    "latent_memory_loss",
    "make_tta_optimizer",
    "softmax_entropy",
]
