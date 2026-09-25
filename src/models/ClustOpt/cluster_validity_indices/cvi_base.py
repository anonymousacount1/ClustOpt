"""Abstract base class for Cluster Validity Indices (CVI).

A CVI implements three responsibilities: producing raw metric values,
mapping them to a deterministic ``[0, 1]`` range where 1 is best, and
aggregating the normalised values into a single scalar in ``[0, 1]``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class BaseCVI(ABC):
    """Contract for CVI evaluators consumed by the search algorithms."""

    @abstractmethod
    def evaluate(
        self,
        labels: np.ndarray,
        X_decision: np.ndarray,
        X_full: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute raw per-metric values. Values may be unbounded."""

    @abstractmethod
    def normalize_scores(
        self,
        raw_scores: Dict[str, float],
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Map raw values to deterministic scores in ``[0, 1]``."""

    @abstractmethod
    def aggregate_score(self, normalized_scores: Dict[str, float]) -> float:
        """Reduce normalised scores to a single scalar in ``[0, 1]``."""

    def evaluate_normalize_aggregate(
        self,
        labels: np.ndarray,
        X_decision: np.ndarray,
        X_full: Optional[np.ndarray] = None,
    ) -> float:
        """Convenience helper that chains the three primitive steps."""
        raw = self.evaluate(labels, X_decision, X_full)
        norm = self.normalize_scores(raw, labels)
        return self.aggregate_score(norm)
