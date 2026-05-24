"""Model components for DeMemte."""

from .baseline import ResNetBaseline, make_backbone
from .vq import VectorQuantizer2D, LatentProjector, LatentUnprojector
from .attractor import AttractorMemory, AmbiguityGate
from .dememte import DeMemteAttractor, make_dememte_e5, make_dememte_variant

__all__ = [
    "ResNetBaseline",
    "make_backbone",
    "VectorQuantizer2D",
    "LatentProjector",
    "LatentUnprojector",
    "AttractorMemory",
    "AmbiguityGate",
    "DeMemteAttractor",
    "make_dememte_e5",
    "make_dememte_variant",
]
