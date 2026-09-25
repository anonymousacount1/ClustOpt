# metrics/pattern/alpha_shape_compactness.py
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString, Polygon
from shapely.ops import polygonize, unary_union

from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class AlphaShapeCompactnessMetric(BaseMetric):
    """
    Alpha-shape compactness score in [0,1], higher is better.

    For each cluster:
      - project to 2D (PCA via SVD if d>2; centered if d==2)
      - build alpha-shape boundary using Delaunay triangles filtered by circumradius
      - compute isoperimetric quotient: IQ = 4*pi*A / P^2  (clipped to [0,1])
    Aggregate: size-weighted mean over clusters.

    Notes:
      - expects noise removed by CVI (remove_noise=True)
      - space='full' (geometry)
      - When a ``partition_context`` is provided the per-cluster alpha
        shape + 2D projection are reused with ``convexity_ratio``.
    """
    name = "alpha_shape_compactness"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        alpha: float | None = None,        # if None -> auto from kNN scale
        alpha_scale: float = 1.5,          # used only when alpha=None
        min_cluster_points: int = 10,
        eps: float = 1e-12,
    ):
        self.alpha = None if alpha is None else float(alpha)
        self.alpha_scale = float(alpha_scale)
        self.min_cluster_points = int(min_cluster_points)
        self.eps = float(eps)

    # ---------------- helpers (legacy, used when no context) ----------------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for alpha-shape compactness.")
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

    def _alpha_shape_polygon(self, pts2d: np.ndarray, alpha: float):
        if pts2d.shape[0] < 4:
            return None
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
            return None
        lines = [(tuple(pts2d[u]), tuple(pts2d[v])) for (u, v) in boundary_edges]
        polys = list(polygonize(MultiLineString(lines)))
        if not polys:
            return None
        geom = unary_union(polys)
        return geom

    def _isoperimetric_quotient(self, geom) -> float:
        if geom is None or geom.is_empty:
            return 0.0
        if geom.geom_type == "Polygon":
            A = float(geom.area)
            P = float(geom.length)
            if A <= self.eps or P <= self.eps:
                return 0.0
            return float(np.clip((4.0 * np.pi * A) / (P * P), 0.0, 1.0))
        if geom.geom_type == "MultiPolygon":
            totalA = 0.0
            acc = 0.0
            for poly in geom.geoms:
                A = float(poly.area)
                P = float(poly.length)
                if A <= self.eps or P <= self.eps:
                    continue
                iq = (4.0 * np.pi * A) / (P * P)
                iq = float(np.clip(iq, 0.0, 1.0))
                acc += A * iq
                totalA += A
            return float(acc / totalA) if totalA > 0 else 0.0
        return 0.0

    def _cluster_score_legacy(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        Z = self._to_2d(Xi)
        alpha = self.alpha if self.alpha is not None else self._auto_alpha(Z)
        geom = self._alpha_shape_polygon(Z, alpha=alpha)
        iq = self._isoperimetric_quotient(geom)
        return float(np.clip(iq, 0.0, 1.0))

    def _cluster_score_with_ctx(
        self, label: int, Xi: np.ndarray, partition_context,
    ) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        geom, _area = partition_context.cluster_alpha_shape(
            label, Xi, alpha=self.alpha, alpha_scale=self.alpha_scale, eps=self.eps,
        )
        iq = self._isoperimetric_quotient(geom)
        return float(np.clip(iq, 0.0, 1.0))

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
