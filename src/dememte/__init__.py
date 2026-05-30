"""DeMemte: strict VQSA models, baselines, data, and experiment helpers."""

from .config import BaselineConfig, E5Config, E6Config, AblationConfig
from .data import build_loaders, seed_everything
from .codebook_repair import (
    CodebookRepairConfig,
    LocalCodebookRepairAdapter,
    calibrate_repair_thresholds,
    quantize_with_codebook_view,
)
from .memory import (
    EpisodicBuffer,
    HippocampalConfig,
    HippocampalMemoryAdapter,
    associative_recall,
    effective_codebook,
    familiarity_gate,
    pattern_completion,
)
from .retrieval import (
    RetrievalCache,
    RetrievalConfig,
    RetrievalLogitAdapter,
    build_labeled_cache,
    extract_retrieval_key,
)
from .tta import (
    AlphaBNStatsAdapter,
    CodebookLossAdapter,
    EATALiteAdapter,
    MemoryTentAdapter,
    NoUpdateAdapter,
    SoftAssignTentAdapter,
    SourceFilterEATAAdapter,
    TentAdapter,
    collect_tta_bn_params,
    collect_tta_codebook_params,
    collect_tta_ln_params,
    configure_tta_codebook,
    configure_tta_layernorm,
    configure_tta_model,
    latent_memory_loss,
    make_tta_optimizer,
    soft_assign_entropy,
    softmax_entropy,
)

__all__ = [
    "BaselineConfig",
    "E5Config",
    "E6Config",
    "AblationConfig",
    "build_loaders",
    "seed_everything",
    # E10-A local codebook repair.
    "CodebookRepairConfig",
    "LocalCodebookRepairAdapter",
    "calibrate_repair_thresholds",
    "quantize_with_codebook_view",
    "TentAdapter",
    "EATALiteAdapter",
    "NoUpdateAdapter",
    "MemoryTentAdapter",
    "SourceFilterEATAAdapter",
    "SoftAssignTentAdapter",
    "CodebookLossAdapter",
    "AlphaBNStatsAdapter",
    "collect_tta_bn_params",
    "collect_tta_ln_params",
    "collect_tta_codebook_params",
    "configure_tta_model",
    "configure_tta_layernorm",
    "configure_tta_codebook",
    "latent_memory_loss",
    "make_tta_optimizer",
    "softmax_entropy",
    "soft_assign_entropy",
    # E10 hippocampal memory module.
    "HippocampalConfig",
    "HippocampalMemoryAdapter",
    "EpisodicBuffer",
    "associative_recall",
    "familiarity_gate",
    "pattern_completion",
    "effective_codebook",
    # E11 retrieval-logit memory.
    "RetrievalConfig",
    "RetrievalCache",
    "RetrievalLogitAdapter",
    "build_labeled_cache",
    "extract_retrieval_key",
]
