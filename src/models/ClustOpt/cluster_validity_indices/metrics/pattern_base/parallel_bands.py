# metrics/pattern/parallel_bands.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

try:
    import cv2
except Exception as e:
    cv2 = None
    _cv2_import_error = e


class ParallelBandsMetric(BaseMetric):
    """
    Parallel Bands / Lanes metric (cluster-level):
    - each cluster should look like a straight band (line-like)
    - all bands should be parallel (orientation consistency, orientation-invariant)
    - band offsets along a shared global normal should be regularly spaced

    Output in [0,1], higher is better.
    """
    name = "parallel_bands"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        straightness_floor: float = 0.6,
        w_angle: float = 0.5,
        w_spacing: float = 0.5,
        min_cluster_points: int = 8,
        eps: float = 1e-12,
    ):
        if cv2 is None:
            raise ImportError(f"OpenCV (cv2) is required for ParallelBandsMetric: {_cv2_import_error}")

        self.straightness_floor = float(straightness_floor)
        self.w_angle = float(w_angle)
        self.w_spacing = float(w_spacing)
        self.min_cluster_points = int(min_cluster_points)
        self.eps = float(eps)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for parallel bands.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _fit_line_stats(self, Xc: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Returns:
          u2: unit direction in 2D (orientation-invariant later)
          s_straight: var_parallel / (var_parallel + var_perp) in [0,1]
        """
        Z = self._to_2d(Xc)
        pts = Z.astype(np.float32)

        line = cv2.fitLine(pts, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        vx, vy, x0, y0 = [float(v) for v in line.reshape(-1)]

        u = np.array([vx, vy], dtype=float)
        nu = float(np.linalg.norm(u))
        if nu <= self.eps:
            u = np.array([1.0, 0.0], dtype=float)
        else:
            u /= nu

        # normal
        n = np.array([-u[1], u[0]], dtype=float)

        delta = Z - np.array([x0, y0], dtype=float)[None, :]
        t = delta @ u
        dperp = delta @ n

        var_par = float(np.var(t))
        var_perp = float(np.var(dperp))
        denom = var_par + var_perp
        s_straight = 0.0 if denom <= self.eps else float(var_par / denom)
        s_straight = float(np.clip(s_straight, 0.0, 1.0))

        return u, s_straight

    def _line_orientation_score(self, u_list: np.ndarray, w: np.ndarray) -> float:
        """
        Orientation-invariant angular consistency for lines:
        use angles θ but treat θ and θ+π as same by using 2θ trick.
        score = R where R = sqrt(mean(cos(2θ))^2 + mean(sin(2θ))^2)
        """
        # angles in [-pi, pi]
        thetas = np.arctan2(u_list[:, 1], u_list[:, 0])
        # 2θ trick
        c = np.average(np.cos(2.0 * thetas), weights=w)
        s = np.average(np.sin(2.0 * thetas), weights=w)
        R = float(np.sqrt(c * c + s * s))
        return float(np.clip(R, 0.0, 1.0))  # 1=aligned, 0=spread

    def _global_normal_from_u(self, u_list: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute a global direction for lines (orientation-invariant) via
        eigenvector of weighted second-moment matrix of directions.
        """
        # Each u defines a line, so u and -u equivalent.
        # Build M = sum w * (u u^T), take principal eigenvector.
        M = np.zeros((2, 2), dtype=float)
        for u, wi in zip(u_list, w):
            M += float(wi) * np.outer(u, u)
        # principal eigenvector
        vals, vecs = np.linalg.eigh(M)
        u_global = vecs[:, int(np.argmax(vals))]
        u_global = u_global / (np.linalg.norm(u_global) + self.eps)
        n_global = np.array([-u_global[1], u_global[0]], dtype=float)
        n_global = n_global / (np.linalg.norm(n_global) + self.eps)
        return n_global

    def _regular_spacing_score(self, offsets: np.ndarray) -> float:
        if offsets.size < 3:
            return 0.0
        diffs = np.diff(np.sort(offsets))
        diffs = diffs[diffs > self.eps]
        if diffs.size < 2:
            return 0.0
        m = float(np.mean(diffs))
        s = float(np.std(diffs))
        if m <= self.eps:
            return 0.0
        cv = s / m
        return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

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

        u_list = []
        centroids = []
        weights = []

        for lab in uniq:
            Xi = X[labels == lab]
            if Xi.shape[0] < self.min_cluster_points:
                continue

            u, s_straight = self._fit_line_stats(Xi)

            gate = 0.0 if s_straight < self.straightness_floor else s_straight
            w = float(Xi.shape[0]) * gate
            if w <= 0.0:
                continue

            u_list.append(u)
            centroids.append(np.mean(Xi, axis=0)[:2])  # offsets computed in 2D frame
            weights.append(w)

        if len(u_list) < 2:
            return float("-inf")

        u_arr = np.vstack(u_list)
        w_arr = np.asarray(weights, dtype=float)

        # angle consistency score
        score_angle = self._line_orientation_score(u_arr, w_arr)

        # spacing score: use shared global normal
        n_global = self._global_normal_from_u(u_arr, w_arr)
        C = np.vstack(centroids)  # (k,2)
        offsets = C @ n_global
        score_spacing = self._regular_spacing_score(offsets)

        wsum = self.w_angle + self.w_spacing
        if wsum <= self.eps:
            wsum = 1.0
        score = (self.w_angle * score_angle + self.w_spacing * score_spacing) / wsum
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.7)
