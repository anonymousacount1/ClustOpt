import numpy as np
from sklearn.decomposition import PCA
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class AvgPCAIsotropyMetric(BaseMetric):
    """
    Measures cluster isotropy in 2D using PCA eigenvalue ratio.
    Higher is better: more isotropic (round-ish) clusters.
    Output is already in (0, 1] before normalization, but we still clamp in safe_normalize.
    """
    name = "avg_pca_isotropy"
    space = "full"          # recommended for geometric structure
    higher_is_better = True
    needs_original_labels = False

    def __init__(self, eps: float = 1e-8, agg: str = "mean"):
        self.eps = float(eps)
        self.agg = agg  # "mean" or "median"

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        # must be at least 2D for PCA(2)
        if X.ndim != 2 or X.shape[1] < 2:
            return float("-inf")

        unique = np.unique(labels)
        ratios_inv = []

        for l in unique:
            pts = X[labels == l]
            if pts.shape[0] < 3:
                continue

            pca = PCA(n_components=2).fit(pts)
            v = pca.explained_variance_
            if v.shape[0] < 2:
                continue

            # ratio = v1 / v2 ; we use inverted ratio to map isotropy->high score
            ratio = v[0] / (v[1] + self.eps)
            ratios_inv.append(1.0 / (ratio + self.eps))

        if not ratios_inv:
            return float("-inf")

        if self.agg == "median":
            return float(np.median(ratios_inv))
        return float(np.mean(ratios_inv))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # raw is already roughly in (0,1] for most cases; keep it as-is
        return raw
