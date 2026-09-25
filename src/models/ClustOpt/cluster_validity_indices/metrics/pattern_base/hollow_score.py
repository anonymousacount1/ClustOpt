# metrics/pattern/hollow_score.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class HollowScoreMetric(BaseMetric):
    name = "hollow_score"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 40,
        n_radial_bins: int = 20,
        inner_frac: float = 0.25,
        eps: float = 1e-12,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.n_radial_bins = int(n_radial_bins)
        self.inner_frac = float(inner_frac)
        self.eps = float(eps)

    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if Xc.shape[1] == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return (U[:, :2] * S[:2]).astype(float)

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points or Xi.shape[1] < 2:
            return float("-inf")
        Z = self._to_2d(Xi)

        # use centroid as center
        c = np.mean(Z, axis=0)
        r = np.linalg.norm(Z - c[None, :], axis=1)
        rmax = float(np.max(r))
        if rmax <= self.eps:
            return 0.0

        # normalize radius to [0,1]
        rn = r / (rmax + self.eps)

        B = max(8, self.n_radial_bins)
        h, edges = np.histogram(rn, bins=B, range=(0.0, 1.0))
        p = h.astype(float) / (np.sum(h) + self.eps)

        # "center mass" = sum bins near 0
        inner_edge = float(self.inner_frac)
        inner_bins = int(np.clip(np.floor(inner_edge * B), 1, B - 2))
        inner_mass = float(np.sum(p[:inner_bins]))

        # "ring mass" = best band away from center (avoid last bin only)
        # take max over bands of width ~B/5 starting after inner_bins
        band_w = max(2, B // 5)
        start = inner_bins
        best = 0.0
        for s in range(start, B - band_w):
            best = max(best, float(np.sum(p[s:s + band_w])))

        # hollow: low center mass + strong ring mass
        # clamp into [0,1]
        score = (1.0 - inner_mass) * best
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
