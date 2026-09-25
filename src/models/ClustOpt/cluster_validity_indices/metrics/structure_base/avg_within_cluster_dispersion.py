import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class AvgWithinClusterDispersionMetric(BaseMetric):
    """
    Measures true compactness via average distance-to-centroid per cluster.
    Lower dispersion is better, but we normalize to [0,1] where higher is better.

    raw >= 0, normalize with inverse squash: exp(-raw/scale) in (0,1].
    """
    name = "avg_within_cluster_dispersion"
    space = "decision"
    higher_is_better = True
    needs_original_labels = False

    def __init__(self, eps: float = 1e-12, scale: float = 1.0, agg: str = "mean"):
        self.eps = float(eps)
        self.scale = float(scale)
        self.agg = str(agg).lower()  # "mean" | "median"
        if self.agg not in {"mean", "median"}:
            raise ValueError("agg must be 'mean' or 'median'")

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        uniq = np.unique(labels)
        if uniq.size < 2:
            return float("-inf")

        per_cluster = []
        for l in uniq:
            pts = X[labels == l]
            if pts.shape[0] < 2:
                continue
            c = pts.mean(axis=0)
            d = np.sqrt(np.sum((pts - c) ** 2, axis=1))
            per_cluster.append(float(np.mean(d)))

        if not per_cluster:
            return float("-inf")

        vals = np.asarray(per_cluster, dtype=float)
        return float(np.median(vals) if self.agg == "median" else np.mean(vals))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        s = max(self.scale, self.eps)
        return float(np.exp(-max(0.0, raw) / s))
