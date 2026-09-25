# metrics/pattern/axial_symmetry.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class AxialSymmetryMetric(BaseMetric):
    name = "axial_symmetry"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 40,
        n_bins: int = 16,
        min_bin_points: int = 8,
        eps: float = 1e-12,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.n_bins = int(n_bins)
        self.min_bin_points = int(min_bin_points)
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
        # choose PC1 direction in Z-space: x-axis if projected, else eigvec
        C = np.cov(Z.T)
        vals, vecs = np.linalg.eigh(C)
        u = vecs[:, np.argmax(vals)]
        u = u / (np.linalg.norm(u) + self.eps)
        nvec = np.array([-u[1], u[0]], dtype=float)
        nvec = nvec / (np.linalg.norm(nvec) + self.eps)

        t = Z @ u
        d = Z @ nvec  # signed

        tmin, tmax = float(np.min(t)), float(np.max(t))
        if (tmax - tmin) <= self.eps:
            return 0.0

        B = max(6, self.n_bins)
        edges = np.linspace(tmin, tmax, B + 1)

        diffs = []
        for b in range(B):
            mask = (t >= edges[b]) & (t < edges[b + 1] if b < B - 1 else t <= edges[b + 1])
            if int(np.sum(mask)) < self.min_bin_points:
                continue
            db = d[mask]
            pos = db[db >= 0]
            neg = -db[db < 0]  # magnitudes of negative side

            if pos.size < 3 or neg.size < 3:
                continue

            # compare robust stats on each side
            mp = float(np.median(pos))
            mn = float(np.median(neg))
            # normalized difference
            denom = max(mp + mn, self.eps)
            diffs.append(abs(mp - mn) / denom)

        if len(diffs) < max(3, B // 4):
            return 0.0

        asym = float(np.mean(diffs))  # 0 is perfect symmetry
        score = float(np.exp(-3.0 * asym))  # squash to [0,1]
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
        return float(raw)
