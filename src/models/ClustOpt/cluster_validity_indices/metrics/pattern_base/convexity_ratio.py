# metrics/pattern/convexity_ratio.py
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import MultiLineString
from shapely.ops import polygonize, unary_union

from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class ConvexityRatioMetric(BaseMetric):
    """
    Convexity ratio in [0,1], higher is more convex.

    CR = Area(alpha-shape) / Area(convex hull)

    - Uses alpha-shape (concave hull) via Delaunay triangle filtering by circumradius.
    - Alpha is either fixed or auto from median NN distance (scale-invariant-ish).
    - Aggregate: size-weighted mean over clusters.

    Expected noise removed by CVI.
    When a ``partition_context`` is provided, the per-cluster alpha shape +
    convex hull are reused with ``alpha_shape_compactness``.
    """
    name = "convexity_ratio"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        alpha: float | None = None,
        alpha_scale: float = 1.5,
        min_cluster_points: int = 10,
        eps: float = 1e-12,
    ):
        self.alpha = None if alpha is None else float(alpha)
        self.alpha_scale = float(alpha_scale)
        self.min_cluster_points = int(min_cluster_points)
        self.eps = float(eps)

    # ---------------- legacy helpers (used when no context) ----------------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for convexity_ratio.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _auto_alpha(self, pts2d: np.ndarray) -> float:
        n = pts2d.shape[0]
        if n < 3:
            return 1.0
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts2d)
            d, _ = tree.query(pts2d, k=2)
            nn = d[:, 1]
        except Exception:
            D = np.sqrt(((pts2d[:, None, :] - pts2d[None, :, :]) ** 2).sum(axis=2))
            np.fill_diagonal(D, np.inf)
            nn = np.min(D, axis=1)
        nn_finite = nn[np.isfinite(nn)]
        if nn_finite.size == 0:
            base = 1.0
        else:
            base = float(np.quantile(nn_finite, 0.9))
        scale = max(base * self.alpha_scale, self.eps)
        return float(1.0 / scale)

    def _circumradius(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        A = float(np.linalg.norm(b - c))
        B = float(np.linalg.norm(a - c))
        C = float(np.linalg.norm(a - b))
        area2 = float(abs(np.cross(b - a, c - a)))
        if area2 <= self.eps:
            return float("inf")
        return (A * B * C) / (2.0 * area2)

    def _alpha_shape_area_legacy(self, pts2d: np.ndarray, alpha: float) -> float:
        if pts2d.shape[0] < 4:
            return 0.0
        tri = Delaunay(pts2d)
        triangles = tri.simplices
        inv_alpha = 1.0 / max(alpha, self.eps)
        edge_count: dict[tuple[int, int], int] = {}
        for (i, j, k) in triangles:
            pa, pb, pc = pts2d[i], pts2d[j], pts2d[k]
            R = self._circumradius(pa, pb, pc)
            if R < inv_alpha:
                for u, v in ((i, j), (j, k), (k, i)):
                    e = (u, v) if u < v else (v, u)
                    edge_count[e] = edge_count.get(e, 0) + 1
        boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
        if not boundary_edges:
            return 0.0
        lines = [(tuple(pts2d[u]), tuple(pts2d[v])) for (u, v) in boundary_edges]
        geom = unary_union(list(polygonize(MultiLineString(lines))))
        if geom is None or geom.is_empty:
            return 0.0
        return float(geom.area)

    def _convex_hull_area_legacy(self, pts2d: np.ndarray) -> float:
        if pts2d.shape[0] < 3:
            return 0.0
        try:
            hull = ConvexHull(pts2d)
            return float(hull.volume)
        except Exception:
            return 0.0

    def _cluster_score_legacy(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        Z = self._to_2d(Xi)
        hull_area = self._convex_hull_area_legacy(Z)
        if hull_area <= self.eps:
            return 0.0
        alpha = self.alpha if self.alpha is not None else self._auto_alpha(Z)
        shape_area = self._alpha_shape_area_legacy(Z, alpha=alpha)
        ratio = shape_area / hull_area
        return float(np.clip(ratio, 0.0, 1.0))

    def _cluster_score_with_ctx(
        self, label: int, Xi: np.ndarray, partition_context,
    ) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        hull_area = partition_context.cluster_hull_area(label, Xi)
        if hull_area <= self.eps:
            return 0.0
        _geom, shape_area = partition_context.cluster_alpha_shape(
            label, Xi, alpha=self.alpha, alpha_scale=self.alpha_scale, eps=self.eps,
        )
        ratio = shape_area / hull_area
        return float(np.clip(ratio, 0.0, 1.0))

    # ---------------- contract ----------------
    def _evaluate_core(
        self, X: np.ndarray, labels: np.ndarray, partition_context=None,
    ) -> float:
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
            if partition_context is not None:
                s = self._cluster_score_with_ctx(int(lab), Xi, partition_context)
            else:
                s = self._cluster_score_legacy(Xi)
            if not np.isfinite(s) or s == float("-inf"):
                continue
            w = float(Xi.shape[0])
            total += w * s
            total_w += w
        if total_w <= 0.0:
            return float("-inf")
        return float(total / total_w)

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        return self._evaluate_core(X, labels, partition_context=None)

    def evaluate_ctx(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        context: Optional[Any] = None,
        partition_context: Optional[Any] = None,
    ) -> float:
        return self._evaluate_core(X, labels, partition_context=partition_context)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.5)
