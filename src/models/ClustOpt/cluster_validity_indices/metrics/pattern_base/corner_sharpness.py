# metrics/pattern/corner_sharpness.py
from __future__ import annotations
import numpy as np
from scipy.spatial import ConvexHull
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class CornerSharpnessMetric(BaseMetric):
    name = "corner_sharpness"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 30,
        angle_threshold_deg: float = 130.0,  # "corner" if interior angle < threshold
        eps: float = 1e-12,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.angle_thr = np.deg2rad(float(angle_threshold_deg))
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

        # convex hull polygon
        try:
            hull = ConvexHull(Z)
        except Exception:
            return 0.0
        verts = hull.vertices
        if verts.size < 4:
            return 0.0

        P = Z[verts]
        m = P.shape[0]

        # compute interior angle at each hull vertex (cyclic)
        corners = 0
        angles = []
        for i in range(m):
            prev = P[(i - 1) % m]
            cur = P[i]
            nxt = P[(i + 1) % m]
            a = prev - cur
            b = nxt - cur
            na = np.linalg.norm(a) + self.eps
            nb = np.linalg.norm(b) + self.eps
            cosang = float(np.dot(a, b) / (na * nb))
            cosang = float(np.clip(cosang, -1.0, 1.0))
            ang = float(np.arccos(cosang))  # [0,pi]
            angles.append(ang)
            if ang < self.angle_thr:
                corners += 1

        # corner density on hull
        frac = corners / max(1, m)

        # also sharpen: encourage discrete corners vs round
        # use dispersion of angles (round shapes: angles almost uniform)
        angs = np.array(angles, dtype=float)
        disp = float(np.std(angs) / (float(np.mean(angs)) + self.eps))
        disp_score = float(np.clip(1.0 - np.exp(-disp), 0.0, 1.0))

        score = 0.6 * float(frac) + 0.4 * disp_score
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
