"""Inference-engine adapters."""

from .base import TargetScoreProvider
from .deepspec import DeepSpecDSparkAdapter

__all__ = [
    "DeepSpecDSparkAdapter",
    "TargetScoreProvider",
]
