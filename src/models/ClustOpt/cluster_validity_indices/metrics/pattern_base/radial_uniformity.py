# metrics/pattern/radial_uniformity.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class RadialUniformityMetric(BaseMetric):
    name = "radial_uniformity"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 50,
        n_bins: int = 20,
        inner_frac: float = 0.2,
        eps: float = 1e-12,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.n_bins = int(n_bins)
        self.inner_frac = float(inner_frac)
        self.eps = float(eps)

    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if Xc.shape[1] == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return (U[:, :2] * S[:2]).astype(float)

    def _entropy_norm(self, p: np.ndarray) -> float:
        p = p[p > 0]
        if p.size == 0:
            return 0.0
        H = -float(np.sum(p * np.log(p + self.eps)))
        Hmax = float(np.log(max(2, p.size)))
        return float(np.clip(H / max(Hmax, self.eps), 0.0, 1.0))

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points or Xi.shape[1] < 2:
            return float("-inf")
        Z = self._to_2d(Xi)
        c = np.mean(Z, axis=0)
        r = np.linalg.norm(Z - c[None, :], axis=1)
        rmax = float(np.max(r))
        if rmax <= self.eps:
            return 0.0
        rn = r / (rmax + self.eps)

        B = max(10, self.n_bins)
        h, _ = np.histogram(rn, bins=B, range=(0.0, 1.0))
        p = h.astype(float) / (np.sum(h) + self.eps)

        # want "filled" center: penalize low inner mass
        inner_bins = int(np.clip(np.floor(self.inner_frac * B), 1, B - 2))
        inner_mass = float(np.sum(p[:inner_bins]))  # blob tends to have decent mass near center

        # radial smoothness/uniformity: moderate/high entropy (not spiky ring), but not too flat either.
        # We'll simply use entropy as smoothness proxy and multiply by inner_mass.
        Hn = self._entropy_norm(p)

        score = inner_mass * Hn
        return float(np.clip(score, 0.0, 1.0))

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X); labels = np.asarray(labels)
        if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.shape[0] or X.shape[1] < 2:
            return float("-inf")
        uniq = np.unique(labels)
        if uniq.size < 1:
            return float("-inf")

        total = 0.0; total_w = 0.0
        for lab in uniq:
            Xi = X[labels == lab]
            s = self._cluster_score(Xi)
            if not np.isfinite(s) or s == float("-inf"):
                continue
            w = float(Xi.shape[0])
            total += w * s; total_w += w
        return float("-inf") if total_w <= 0 else float(total / total_w)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.5)
