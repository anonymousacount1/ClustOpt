# metrics/pattern/curvature_consistency.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class CurvatureConsistencyMetric(BaseMetric):
    """
    Curvature Consistency score in [0,1], higher is better.

    Per cluster:
      - project to 2D (PCA via SVD if d>2; centered if d==2)
      - sort points by PC1 to induce a 1D ordering
      - compute discrete signed curvature proxy k_i from local turning angles
      - consistency = sign_consistency * (1 - variability)
      - blend with elongation gate

    Aggregate: size-weighted mean across clusters.
    """
    name = "curvature_consistency"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        min_cluster_points: int = 18,
        min_triplets: int = 8,     # need at least this many curvature samples
        eps: float = 1e-12,
        w_sign: float = 0.6,
        w_var: float = 0.4,
        var_scale: float = 1.0,    # variability squash
        w_elon: float = 0.30,
        elong_r0: float = 2.0,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.min_triplets = int(min_triplets)
        self.eps = float(eps)

        self.w_sign = float(w_sign)
        self.w_var = float(w_var)
        self.var_scale = float(var_scale)

        self.w_elon = float(w_elon)
        self.elong_r0 = float(elong_r0)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for curvature_consistency.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _elongation_score(self, Z: np.ndarray) -> float:
        x = Z[:, 0]
        y = Z[:, 1]
        s1 = float(np.std(x))
        s2 = float(np.std(y)) + self.eps
        ratio = s1 / s2
        return float(np.clip(1.0 - np.exp(-ratio / max(self.elong_r0, self.eps)), 0.0, 1.0))

    def _signed_curvature_samples(self, P: np.ndarray) -> np.ndarray:
        """
        P: (n,2) ordered polyline points
        returns k: (n-2,) signed curvature proxy
        """
        n = P.shape[0]
        if n < 3:
            return np.array([], dtype=float)

        a = P[1:-1] - P[:-2]    # v_prev
        b = P[2:] - P[1:-1]     # v_next

        na = np.linalg.norm(a, axis=1) + self.eps
        nb = np.linalg.norm(b, axis=1) + self.eps

        # angle between vectors
        dot = np.sum(a * b, axis=1) / (na * nb)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)  # [0,pi]

        # sign via 2D cross product (scalar z-component)
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        sign = np.sign(cross)   # -1,0,1

        # curvature proxy: theta / local_length
        local_len = 0.5 * (na + nb)
        k = sign * (theta / local_len)
        return k.astype(float)

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        Z = self._to_2d(Xi)
        # induce ordering by PC1
        order = np.argsort(Z[:, 0])
        P = Z[order]

        k = self._signed_curvature_samples(P)
        if k.size < self.min_triplets:
            return 0.0

        # ignore near-zero curvature samples (straight segments)
        k_abs = np.abs(k)
        thr = float(np.percentile(k_abs, 25))  # adaptive small threshold
        mask = k_abs > max(thr, 1e-8)
        k2 = k[mask]
        if k2.size < self.min_triplets:
            return 0.0

        # 1) sign consistency: majority direction of curvature
        pos = float(np.mean(k2 > 0))
        neg = float(np.mean(k2 < 0))
        sign_consistency = float(max(pos, neg))  # in [0.5,1]
        # map [0.5,1] -> [0,1]
        sign_score = float(np.clip((sign_consistency - 0.5) / 0.5, 0.0, 1.0))

        # 2) variability: coefficient of variation of |k|
        km = float(np.mean(np.abs(k2))) + self.eps
        ks = float(np.std(np.abs(k2)))
        cv = ks / km  # >=0
        # map low cv -> high score (smooth), squash
        var_score = float(np.exp(-cv / max(self.var_scale, self.eps)))  # (0,1]

        # combine curvature consistency
        wsum = self.w_sign + self.w_var
        if wsum <= self.eps:
            wsum = 1.0
        curvature_consistency = (self.w_sign * sign_score + self.w_var * var_score) / wsum
        curvature_consistency = float(np.clip(curvature_consistency, 0.0, 1.0))

        # elongation gate / blend
        s_elon = self._elongation_score(Z)
        w = float(np.clip(self.w_elon, 0.0, 1.0))
        score = (1.0 - w) * curvature_consistency + w * s_elon
        return float(np.clip(score, 0.0, 1.0))

    # ---------- contract ----------
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.shape[0]:
            return float("-inf")
        if X.shape[1] < 2:
            return float("-inf")

        uniq = np.unique(labels)
        if uniq.size < 1:
            return float("-inf")

        total = 0.0
        total_w = 0.0
        for lab in uniq:
            Xi = X[labels == lab]
            s = self._cluster_score(Xi)
            if not np.isfinite(s) or s == float("-inf"):
                continue
            w = float(Xi.shape[0])
            total += w * s
            total_w += w

        if total_w <= 0.0:
            return float("-inf")
        return float(total / total_w)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.6)


