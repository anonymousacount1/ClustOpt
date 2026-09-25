# metrics/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional
import numpy as np

SpaceT = Literal["decision", "full"]

@dataclass(frozen=True)
class MetricContext:
    n_samples: int
    n_clusters: int

class BaseMetric(ABC):
    name: str
    space: SpaceT = "decision"
    higher_is_better: bool = True
    needs_original_labels: bool = False

    @abstractmethod
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        ...

    @abstractmethod
    def normalize(self, raw: float, ctx: MetricContext) -> float:
        ...

    # ------------------------------------------------------------- ctx-aware
    # Optimised metrics may override this to read pre-computed artefacts from
    # a :class:`GlobalMetricContext` / :class:`PartitionMetricContext`. The
    # default implementation forwards to the legacy ``evaluate(X, labels)``
    # so existing metrics keep working unchanged.
    def evaluate_ctx(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        context: Optional[Any] = None,
        partition_context: Optional[Any] = None,
    ) -> float:
        return self.evaluate(X, labels)

    def safe_evaluate(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        context: Optional[Any] = None,
        partition_context: Optional[Any] = None,
    ) -> float:
        try:
            v = float(self.evaluate_ctx(
                X, labels, context=context, partition_context=partition_context,
            ))
            if np.isnan(v) or np.isinf(v):
                return float("-inf")
            return v
        except Exception:
            return float("-inf")

    def safe_normalize(self, raw: float, ctx: MetricContext) -> float:
        if raw == float("-inf") or np.isnan(raw) or np.isinf(raw):
            return 0.0
        try:
            v = float(self.normalize(raw, ctx))
        except Exception:
            return 0.0
        return float(min(1.0, max(0.0, v)))
