# metrics/pattern/line_straightness.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

try:
    import cv2
except Exception as e:
    cv2 = None
    _cv2_import_error = e


class LineStraightnessMetric(BaseMetric):
    """
    Line Straightness score in [0,1] using robust cv2.fitLine.
    Higher is better.

    Per cluster:
      - project to 2D (PCA if d>2; centered if d==2)
      - fit line -> t along direction, d perpendicular
      - s = var(t)/(var(t)+var(d))
      - thickness = exp(-std(d)/(std(t)+eps))
      - score = 0.5*s + 0.5*thickness

    Aggregate: size-weighted mean over clusters.
    """
    name = "line_straightness"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        min_cluster_points: int = 3,
        eps: float = 1e-8,
        w_energy: float = 0.5,
        w_thickness: float = 0.5,
    ):
        if cv2 is None:
            raise ImportError(f"OpenCV (cv2) is required for LineStraightnessMetric: {_cv2_import_error}")
        self.min_cluster_points = int(min_cluster_points)
        self.eps = float(eps)
        self.w_energy = float(w_energy)
        self.w_thickness = float(w_thickness)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for line_straightness.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _fitline_geometry(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = Z.shape[0]
        if n <= 1:
            return np.zeros(n, dtype=float), np.zeros(n, dtype=float)

        pts = Z.astype(np.float32)
        line = cv2.fitLine(pts, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        vx, vy, x0, y0 = [float(v) for v in line.reshape(-1)]

        u = np.array([vx, vy], dtype=float)
        nu = float(np.linalg.norm(u))
        if nu <= 0.0:
            u = np.array([1.0, 0.0], dtype=float)
        else:
            u /= nu
        nvec = np.array([-u[1], u[0]], dtype=float)

        p0 = np.array([x0, y0], dtype=float)
        delta = Z - p0[None, :]
        t = delta @ u
        d = delta @ nvec
        return t.astype(float), d.astype(float)

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        Z = self._to_2d(Xi)
        t, d = self._fitline_geometry(Z)

        var_par = float(np.var(t))
        var_perp = float(np.var(d))
        denom = var_par + var_perp
        if denom <= 0.0:
            return 0.0

        s = float(var_par / denom)  # [0,1]
        std_par = float(np.std(t)) + self.eps
        std_perp = float(np.std(d))
        thickness = float(np.exp(-std_perp / std_par))  # (0,1]

        wsum = self.w_energy + self.w_thickness
        if wsum <= 0.0:
            wsum = 1.0
        score = (self.w_energy * s + self.w_thickness * thickness) / wsum
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
        # already [0,1]
        return float(raw)
