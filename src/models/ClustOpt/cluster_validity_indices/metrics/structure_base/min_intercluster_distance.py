import numpy as np
from scipy.spatial.distance import cdist
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class MinInterclusterDistanceMetric(BaseMetric):
    """
    Minimum distance between any pair of clusters (single-link separation).
    Higher is better.
    Raw is unbounded (>=0), normalize via smooth squash to [0,1).
    """
    name = "min_intercluster_distance"
    space = "decision"
    higher_is_better = True
    needs_original_labels = False

    def __init__(self, eps: float = 1e-12, scale: float = 1.0):
        self.eps = float(eps)
        self.scale = float(scale)

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            return float("-inf")

        min_dist = np.inf
        for i, li in enumerate(unique_labels):
            Xi = X[labels == li]
            if Xi.size == 0:
                continue
            for lj in unique_labels[i + 1:]:
                Xj = X[labels == lj]
                if Xj.size == 0:
                    continue
                curr_min = float(np.min(cdist(Xi, Xj)))
                if curr_min < min_dist:
                    min_dist = curr_min

        if not np.isfinite(min_dist):
            return float("-inf")

        return float(min_dist)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # raw >= 0. squash to [0,1)
        s = max(self.scale, self.eps)
        return float(1.0 - np.exp(-max(0.0, raw) / s))
