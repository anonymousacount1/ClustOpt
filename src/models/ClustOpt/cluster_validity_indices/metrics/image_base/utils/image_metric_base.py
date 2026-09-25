# metrics/image/image_metric_base.py
from __future__ import annotations
from abc import abstractmethod
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import compute_global_bounds, Bounds2D

class ImagePerClusterMetric(BaseMetric):
    min_cluster_points: int = 30

    @abstractmethod
    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        ...

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X); labels = np.asarray(labels)
        if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.shape[0] or X.shape[1] < 2:
            return float("-inf")
        uniq = np.unique(labels)
        if uniq.size < 1:
            return float("-inf")

        bounds = compute_global_bounds(X)

        total = 0.0
        total_w = 0.0
        for lab in uniq:
            Xi = X[labels == lab]
            if Xi.shape[0] < int(self.min_cluster_points):
                continue
            s = self._cluster_score(Xi, bounds)
            if not np.isfinite(s) or s == float("-inf"):
                continue
            w = float(Xi.shape[0])
            total += w * float(s)
            total_w += w

        return float("-inf") if total_w <= 0 else float(total / total_w)
