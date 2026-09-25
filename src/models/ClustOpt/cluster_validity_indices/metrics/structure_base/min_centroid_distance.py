import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class MinCentroidDistanceMetric(BaseMetric):
    """
    Robust inter-cluster separation using centroid distances.
    Higher is better.

    raw >= 0 (unbounded), normalize with smooth squash to [0,1).
    """
    name = "min_centroid_distance"
    space = "decision"
    higher_is_better = True
    needs_original_labels = False

    def __init__(self, eps: float = 1e-12, scale: float = 1.0):
        self.eps = float(eps)
        self.scale = float(scale)

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        uniq = np.unique(labels)
        if uniq.size < 2:
            return float("-inf")

        centroids = []
        for l in uniq:
            pts = X[labels == l]
            if pts.shape[0] == 0:
                continue
            centroids.append(pts.mean(axis=0))

        if len(centroids) < 2:
            return float("-inf")

        C = np.vstack(centroids)  # (k, d)
        # pairwise distances between centroids
        # compute via broadcasting (no scipy)
        diffs = C[:, None, :] - C[None, :, :]
        D = np.sqrt(np.sum(diffs * diffs, axis=2))  # (k, k)
        # ignore diagonal (use fill_diagonal to avoid 0*inf = nan propagation)
        np.fill_diagonal(D, np.inf)
        v = float(np.min(D))
        if not np.isfinite(v):
            return float("-inf")
        return v

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        s = max(self.scale, self.eps)
        return float(1.0 - np.exp(-max(0.0, raw) / s))
