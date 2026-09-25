# metrics/pattern/parabolicity.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class ParabolicityMetric(BaseMetric):
    """
    Parabolicity score in [0,1]:
    per cluster:
      - project to 2D (PCA if d>2, centered if d==2)
      - z-score standardize both axes
      - fit quadratic in two orientations: y(x) and x(y), keep best R^2
      - curvature_score from |a| using scaled exponential
      - score_cluster = (R^2 >= r2_min) ? R^2 * curvature_score : 0
    global: size-weighted average across clusters

    Higher is better.
    """
    name = "parabolicity"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        a_scale: float = 0.1,
        a_clip: float = 5.0,
        r2_min: float = 0.2,
        min_cluster_points: int = 5,
        eps: float = 1e-12,
    ):
        self.a_scale = float(a_scale)
        self.a_clip = float(a_clip)
        self.r2_min = float(r2_min)
        self.min_cluster_points = int(min_cluster_points)
        self.eps = float(eps)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D [n_samples, n_features].")
        n, d = X.shape
        if n == 0:
            return np.empty((0, 2), dtype=float)
        if d < 2:
            raise ValueError("Need at least 2 dims for parabolicity.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        # PCA to 2D via SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _fit_quad_r2(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Fit y ≈ a x^2 + b x + c and return (a, r2) clipped.
        """
        if x.size < self.min_cluster_points:
            return 0.0, 0.0
        try:
            a, b, c = [float(v) for v in np.polyfit(x, y, deg=2)]
        except np.linalg.LinAlgError:
            return 0.0, 0.0

        y_pred = a * x**2 + b * x + c
        y_mean = float(np.mean(y))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        ss_res = float(np.sum((y - y_pred) ** 2))
        if ss_tot <= self.eps:
            r2 = 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot
        r2 = float(np.clip(r2, 0.0, 1.0))
        return a, r2

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        Z = self._to_2d(Xi)  # (n,2)
        x_raw = Z[:, 0]
        y_raw = Z[:, 1]

        sx = float(np.std(x_raw))
        sy = float(np.std(y_raw))
        if sx <= self.eps or sy <= self.eps:
            return 0.0  # degenerate cluster -> no parabola signal

        x = (x_raw - float(np.mean(x_raw))) / (sx + self.eps)
        y = (y_raw - float(np.mean(y_raw))) / (sy + self.eps)

        a_xy, r2_xy = self._fit_quad_r2(x, y)
        a_yx, r2_yx = self._fit_quad_r2(y, x)

        if r2_xy >= r2_yx:
            best_a, best_r2 = a_xy, r2_xy
        else:
            best_a, best_r2 = a_yx, r2_yx

        if best_r2 < self.r2_min:
            return 0.0

        abs_a = abs(best_a)
        abs_a_norm = min(abs_a / max(self.a_scale, 1e-8), self.a_clip)
        curvature = 1.0 - float(np.exp(-abs_a_norm))  # [0,1)
        score = best_r2 * curvature
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
            if Xi.shape[0] < self.min_cluster_points:
                continue
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
