# metrics/pattern/turning_points_count.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class TurningPointsCountMetric(BaseMetric):
    """
    Turning Points (curvature sign changes) score in [0,1], higher is more zigzag/oscillatory.

    Per cluster:
      - project to 2D (PCA via SVD if d>2; centered if d==2)
      - sort by PC1 to induce ordering
      - compute signed curvature proxy k_i from local turning angles
      - filter near-zero curvature samples
      - count sign changes in sign(k)
      - normalize as rate in [0,1]
      - blend with elongation (to avoid blobs)

    Aggregate: size-weighted mean over clusters.
    """
    name = "turning_points_count"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        min_cluster_points: int = 18,
        min_samples: int = 10,        # after filtering
        eps: float = 1e-12,
        near_zero_percentile: float = 25.0,  # filter threshold for |k|
        w_elon: float = 0.30,
        elong_r0: float = 2.0,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.min_samples = int(min_samples)
        self.eps = float(eps)
        self.near_zero_percentile = float(near_zero_percentile)
        self.w_elon = float(w_elon)
        self.elong_r0 = float(elong_r0)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for turning_points_count.")
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

    def _signed_curvature(self, P: np.ndarray) -> np.ndarray:
        n = P.shape[0]
        if n < 3:
            return np.array([], dtype=float)

        a = P[1:-1] - P[:-2]
        b = P[2:] - P[1:-1]
        na = np.linalg.norm(a, axis=1) + self.eps
        nb = np.linalg.norm(b, axis=1) + self.eps

        dot = np.sum(a * b, axis=1) / (na * nb)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)

        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        sign = np.sign(cross)

        local_len = 0.5 * (na + nb)
        k = sign * (theta / local_len)
        return k.astype(float)

    def _count_sign_changes(self, s: np.ndarray) -> int:
        # s is array of -1/+1 (no zeros)
        if s.size < 2:
            return 0
        return int(np.sum(s[1:] != s[:-1]))

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        Z = self._to_2d(Xi)
        order = np.argsort(Z[:, 0])
        P = Z[order]

        # Bin points along PC1 and average the PC2 coordinate per bin. This
        # gives a y(x) trajectory of the underlying curve, robust to the
        # local noise within each x slice.
        n = P.shape[0]
        n_bins = int(np.clip(np.sqrt(n), 12, 64))
        x = P[:, 0]
        xmin, xmax = float(x.min()), float(x.max())
        if xmax - xmin <= self.eps:
            return 0.0
        bin_edges = np.linspace(xmin, xmax, n_bins + 1)
        bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, n_bins - 1)
        ys = []
        within_stds = []
        for b in range(n_bins):
            mask = bin_idx == b
            if int(mask.sum()) >= max(3, n // (n_bins * 4)):
                ys.append(float(np.mean(P[mask, 1])))
                within_stds.append(float(np.std(P[mask, 1])))
        if len(ys) < 5:
            return 0.0
        y_traj = np.asarray(ys, dtype=float)
        # Significance threshold: a "real" extremum has to step by more than
        # the typical within-bin scatter so that random noise inside a thin
        # band doesn't fake an oscillation.
        typical_noise = float(np.median(within_stds)) if within_stds else 0.0
        sig_thr = max(1e-6, 1.5 * typical_noise)

        diffs = np.diff(y_traj)
        sign = np.sign(diffs)
        small = np.abs(diffs) < sig_thr
        sign[small] = 0
        nz = sign[sign != 0]
        if nz.size < 2:
            extrema = 0
        else:
            extrema = int(np.sum(nz[1:] != nz[:-1]))

        # Saturate so 4+ extrema reach ~0.9 and a smooth curve (0-1 extrema)
        # stays near 0.
        score = float(1.0 - np.exp(-float(extrema) / 1.8))

        s_elon = self._elongation_score(Z)
        w = float(np.clip(self.w_elon, 0.0, 1.0))
        score = (1.0 - w) * score + w * s_elon
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
