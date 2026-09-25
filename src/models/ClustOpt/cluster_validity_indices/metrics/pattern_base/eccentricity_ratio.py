# metrics/pattern/eccentricity_ratio.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class EccentricityRatioMetric(BaseMetric):
    name = "eccentricity_ratio"
    space = "full"
    higher_is_better = True

    def __init__(self, min_cluster_points: int = 12, r0: float = 2.0, eps: float = 1e-12):
        self.min_cluster_points = int(min_cluster_points)
        self.r0 = float(r0)
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
        C = np.cov(Z.T)
        vals = np.linalg.eigvalsh(C)
        lam1 = float(np.max(vals))
        lam2 = float(np.min(vals)) + self.eps
        ratio = np.sqrt(max(lam1, self.eps) / lam2)  # >=1
        # map ratio -> [0,1]
        score = 1.0 - np.exp(-ratio / max(self.r0, self.eps))
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
        return float(v ** 0.7)
