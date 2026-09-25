# metrics/pattern/ribbon_thickness.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class RibbonThicknessMetric(BaseMetric):
    """
    Ribbon Thickness score in [0,1], higher means:
      - the cluster is elongated (ribbon-like),
      - has small perpendicular thickness,
      - thickness is consistent along the ribbon.

    Per cluster:
      - project to 2D (PCA/SVD)
      - take principal direction u (PC1), normal n
      - compute projections t (along) and d (perp signed)
      - bin along t, compute robust local thickness widths w_b
      - thinness: 1/(1+mean_w)
      - consistency: exp(-CV(w_b))
      - elongation gate: exp-style on std ratio
      - score = elon * (w_thin*thinness + w_cons*consistency)

    Expected noise removed by CVI.
    """
    name = "ribbon_thickness"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        min_cluster_points: int = 30,
        n_bins: int = 20,
        min_bin_points: int = 8,
        q_low: float = 10.0,
        q_high: float = 90.0,
        eps: float = 1e-12,
        w_thin: float = 0.5,
        w_cons: float = 0.5,
        elong_r0: float = 2.0,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.n_bins = int(n_bins)
        self.min_bin_points = int(min_bin_points)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)
        self.w_thin = float(w_thin)
        self.w_cons = float(w_cons)
        self.elong_r0 = float(elong_r0)

    # ---------- helpers ----------
    def _to_2d_and_basis(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          Z: (n,2) centered + PCA projected coords
          u: unit vector along PC1 in Z-space
          n: unit normal (perp) in Z-space
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n_s, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for ribbon_thickness.")
        Xc = X - np.mean(X, axis=0, keepdims=True)

        if d == 2:
            Z = Xc.astype(float)
        else:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            Z = (U[:, :2] * S[:2]).astype(float)

        # PC1 in Z-space is simply x-axis after SVD projection
        # but if d==2 and we didn't rotate, we need basis from covariance
        if d == 2:
            C = np.cov(Z.T)
            vals, vecs = np.linalg.eigh(C)
            u = vecs[:, np.argmax(vals)]
            u = u / (np.linalg.norm(u) + self.eps)
        else:
            u = np.array([1.0, 0.0], dtype=float)

        nvec = np.array([-u[1], u[0]], dtype=float)
        nvec = nvec / (np.linalg.norm(nvec) + self.eps)
        return Z, u, nvec

    def _elongation_score(self, t: np.ndarray, d: np.ndarray) -> float:
        s1 = float(np.std(t))
        s2 = float(np.std(d)) + self.eps
        ratio = s1 / s2
        return float(np.clip(1.0 - np.exp(-ratio / max(self.elong_r0, self.eps)), 0.0, 1.0))

    def _cluster_score(self, Xi: np.ndarray) -> float:
        n = Xi.shape[0]
        if n < self.min_cluster_points:
            return float("-inf")

        Z, u, nvec = self._to_2d_and_basis(Xi)
        # projections
        t = Z @ u
        d = Z @ nvec

        # bin along t
        tmin, tmax = float(np.min(t)), float(np.max(t))
        if (tmax - tmin) <= self.eps:
            return 0.0

        B = max(5, self.n_bins)
        edges = np.linspace(tmin, tmax, B + 1)

        widths = []
        for b in range(B):
            mask = (t >= edges[b]) & (t < edges[b + 1] if b < B - 1 else t <= edges[b + 1])
            if int(np.sum(mask)) < self.min_bin_points:
                continue
            dd = np.abs(d[mask])
            lo = float(np.percentile(dd, self.q_low))
            hi = float(np.percentile(dd, self.q_high))
            widths.append(max(0.0, hi - lo))

        if len(widths) < max(3, B // 4):
            return 0.0

        w = np.array(widths, dtype=float)
        mean_w = float(np.mean(w))
        std_w = float(np.std(w))
        cv = std_w / (mean_w + self.eps)

        # thinness: smaller mean thickness -> closer to 1
        thin = 1.0 / (1.0 + max(0.0, mean_w))

        # consistency: lower cv -> closer to 1
        cons = float(np.exp(-cv))

        # elongation gate
        elon = self._elongation_score(t, d)

        wsum = self.w_thin + self.w_cons
        if wsum <= self.eps:
            wsum = 1.0
        base = (self.w_thin * thin + self.w_cons * cons) / wsum

        score = elon * base
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
        return float(v ** 0.7)


