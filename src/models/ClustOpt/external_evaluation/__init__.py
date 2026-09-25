"""External (ground-truth) evaluation helpers, attached at runtime to a
search algorithm so its log includes metrics such as ARI alongside the CVIs.
"""
from __future__ import annotations

from models.ClustOpt.external_evaluation.external_metrics import (
    ExternalEvaluationConfig,
    ExternalEvaluator,
)

__all__ = ["ExternalEvaluationConfig", "ExternalEvaluator"]
