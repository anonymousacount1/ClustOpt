"""Per-partition lazy cache for shared metric inputs.

Built once per evaluated configuration (i.e. per ``labels`` vector). Caches:

* simple partition statistics — cluster indices, sizes, noise mask
* per-cluster 2D projections
* per-cluster alpha shapes and convex hulls (used by both
  ``alpha_shape_compactness`` and ``convexity_ratio``)
* per-cluster rasters, binary masks and skeletons (used by both
  ``skeleton_connectivity`` and ``thickness_uniformity``)

The class also exposes :func:`zhang_suen_thinning` — a vectorised skeleton
helper that replaces the per-pixel Python loop used by the legacy metrics.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


# --------------------------------------------------------------------- thinning
def zhang_suen_thinning(bin_img: np.ndarray, max_iter: int = 80) -> np.ndarray:
    """Vectorised Zhang-Suen thinning.

    Drop-in replacement for the per-pixel double-loop version used by the
    legacy image metrics; produces an identical 0/255 ``uint8`` skeleton but
    is ~50-100x faster on a 256x256 binary mask.
    """
    img = (bin_img > 0).astype(np.uint8)
    h, w = img.shape
    if h < 3 or w < 3:
        return (img * 255).astype(np.uint8)

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p1 = img[1:-1, 1:-1]
            p2 = img[:-2, 1:-1]   # N
            p3 = img[:-2, 2:]     # NE
            p4 = img[1:-1, 2:]    # E
            p5 = img[2:, 2:]      # SE
            p6 = img[2:, 1:-1]    # S
            p7 = img[2:, :-2]     # SW
            p8 = img[1:-1, :-2]   # W
            p9 = img[:-2, :-2]    # NW

            B = p2.astype(np.int16) + p3 + p4 + p5 + p6 + p7 + p8 + p9

            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.int16)
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )

            cond = (p1 == 1) & (B >= 2) & (B <= 6) & (transitions == 1)
            if step == 0:
                cond &= (p2 * p4 * p6 == 0)
                cond &= (p4 * p6 * p8 == 0)
            else:
                cond &= (p2 * p4 * p8 == 0)
                cond &= (p2 * p6 * p8 == 0)

            if not cond.any():
                continue
            interior = img[1:-1, 1:-1]
            img[1:-1, 1:-1] = np.where(cond, 0, interior)
            changed = True

        if not changed:
            break

    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------- 2D PCA per cluster
def _project_to_2d(Xi: np.ndarray) -> Optional[np.ndarray]:
    if Xi.ndim != 2 or Xi.shape[1] < 2 or Xi.shape[0] < 3:
        return None
    Xc = Xi - np.mean(Xi, axis=0, keepdims=True)
    if Xc.shape[1] == 2:
        return Xc.astype(float)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :2] * S[:2]).astype(float)


# ----------------------------------------------------- alpha shape per cluster
def _circumradius(a, b, c, eps: float = 1e-12) -> float:
    A = float(np.linalg.norm(b - c))
    B = float(np.linalg.norm(a - c))
    C = float(np.linalg.norm(a - b))
    area2 = float(abs(np.cross(b - a, c - a)))
    if area2 <= eps:
        return float("inf")
    return (A * B * C) / (2.0 * area2)


def _build_alpha_shape(
    pts2d: np.ndarray,
    alpha: float,
    eps: float = 1e-12,
):
    """Build alpha-shape geometry from a 2D point cloud. Returns (geom, area).

    ``geom`` is a shapely Polygon / MultiPolygon (or ``None`` on degeneracy).
    """
    if pts2d.shape[0] < 4:
        return None, 0.0

    from scipy.spatial import Delaunay
    from shapely.geometry import MultiLineString
    from shapely.ops import polygonize, unary_union

    tri = Delaunay(pts2d)
    triangles = tri.simplices
    inv_alpha = 1.0 / max(alpha, eps)

    edge_count: Dict[Tuple[int, int], int] = {}
    for (i, j, k) in triangles:
        R = _circumradius(pts2d[i], pts2d[j], pts2d[k], eps=eps)
        if R < inv_alpha:
            for u, v in ((i, j), (j, k), (k, i)):
                e = (u, v) if u < v else (v, u)
                edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return None, 0.0

    lines = [(tuple(pts2d[u]), tuple(pts2d[v])) for (u, v) in boundary_edges]
    polys = list(polygonize(MultiLineString(lines)))
    if not polys:
        return None, 0.0

    geom = unary_union(polys)
    if geom is None or geom.is_empty:
        return None, 0.0
    return geom, float(geom.area)


def _convex_hull_area(pts2d: np.ndarray) -> float:
    if pts2d.shape[0] < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts2d).volume)  # 2D ConvexHull.volume == area
    except Exception:
        return 0.0


def _auto_alpha(pts2d: np.ndarray, alpha_scale: float, eps: float = 1e-12) -> float:
    """90th-percentile nearest-neighbour distance (cheap, O(n log n) via cKDTree)."""
    n = pts2d.shape[0]
    if n < 3:
        return 1.0
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts2d)
        d, _ = tree.query(pts2d, k=2)
        nn = d[:, 1]
    except Exception:
        # numpy O(n^2) fallback
        D = np.sqrt(((pts2d[:, None, :] - pts2d[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        nn = np.min(D, axis=1)
    nn_finite = nn[np.isfinite(nn)]
    base = float(np.quantile(nn_finite, 0.9)) if nn_finite.size else 1.0
    scale = max(base * alpha_scale, eps)
    return float(1.0 / scale)


# ---------------------------------------------------------------- main context
class PartitionMetricContext:
    """Per-config lazy cache for partition-derived artefacts."""

    def __init__(
        self,
        labels: np.ndarray,
        global_context,
        noise_label: int = -1,
    ):
        self.labels = np.asarray(labels)
        self.global_context = global_context
        self.noise_label = int(noise_label)

        # Lazy caches
        self._noise_mask: Optional[np.ndarray] = None
        self._cluster_indices: Optional[Dict[int, np.ndarray]] = None
        self._cluster_sizes: Optional[Dict[int, int]] = None
        self._cluster_2d: Dict[int, Optional[np.ndarray]] = {}
        self._cluster_alpha: Dict[Tuple[int, float], Tuple[Any, float]] = {}
        self._cluster_hull_area: Dict[int, float] = {}
        self._cluster_raster: Dict[Tuple[int, Any], Optional[np.ndarray]] = {}
        self._cluster_skeleton: Dict[Tuple[int, Any, Any, int], Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        self._pairwise_view: Dict[Any, Optional[np.ndarray]] = {}
        self._noise_filtered_distance: Dict[Any, Optional[np.ndarray]] = {}
        # Default memory cap for distance caches. 10 000 points -> ~720 MB
        # for a float64 square matrix; chosen so the 9 484-point PRI view
        # still fits (it benefits the most from sharing distance work
        # between silhouette / noise-aware-silhouette / DBCV).
        self.max_distance_cache_n: int = 10_000

    # -------------------------------------------- noise / indices / sizes
    @property
    def noise_mask(self) -> np.ndarray:
        if self._noise_mask is None:
            self._noise_mask = self.labels != self.noise_label
        return self._noise_mask

    @property
    def cluster_indices(self) -> Dict[int, np.ndarray]:
        if self._cluster_indices is None:
            self._cluster_indices = {
                int(lab): np.where(self.labels == lab)[0]
                for lab in np.unique(self.labels)
            }
        return self._cluster_indices

    @property
    def cluster_sizes(self) -> Dict[int, int]:
        if self._cluster_sizes is None:
            self._cluster_sizes = {lab: int(idx.size) for lab, idx in self.cluster_indices.items()}
        return self._cluster_sizes

    # -------------------------------------------------- 2D PCA per cluster
    def cluster_2d(self, label: int, Xi: np.ndarray) -> Optional[np.ndarray]:
        label = int(label)
        if label not in self._cluster_2d:
            self._cluster_2d[label] = _project_to_2d(Xi)
        return self._cluster_2d[label]

    # ----------------------------------------- alpha shape + hull per cluster
    def cluster_alpha_shape(
        self,
        label: int,
        Xi: np.ndarray,
        *,
        alpha: Optional[float] = None,
        alpha_scale: float = 1.5,
        eps: float = 1e-12,
    ) -> Tuple[Any, float]:
        """Return ``(geom, alpha_area)`` for the cluster. Cached per (label, alpha)."""
        label = int(label)
        Z = self.cluster_2d(label, Xi)
        if Z is None:
            return None, 0.0
        if alpha is None:
            alpha = _auto_alpha(Z, alpha_scale=alpha_scale, eps=eps)
        key = (label, float(alpha))
        if key not in self._cluster_alpha:
            self._cluster_alpha[key] = _build_alpha_shape(Z, alpha=alpha, eps=eps)
        return self._cluster_alpha[key]

    def cluster_hull_area(self, label: int, Xi: np.ndarray) -> float:
        label = int(label)
        if label not in self._cluster_hull_area:
            Z = self.cluster_2d(label, Xi)
            self._cluster_hull_area[label] = _convex_hull_area(Z) if Z is not None else 0.0
        return self._cluster_hull_area[label]

    # ---------------------------------------- raster + skeleton per cluster
    def cluster_raster(
        self,
        label: int,
        Xi: np.ndarray,
        raster_params,
        bounds,
    ) -> Optional[np.ndarray]:
        """Cached raster of ``Xi`` for given (raster_params, bounds)."""
        label = int(label)
        key = (label, raster_params, bounds)
        if key not in self._cluster_raster:
            from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import (
                rasterize_points,
            )
            self._cluster_raster[key] = rasterize_points(Xi, raster_params, bounds=bounds)
        return self._cluster_raster[key]

    def cluster_skeleton(
        self,
        label: int,
        Xi: np.ndarray,
        raster_params,
        bounds,
        *,
        close_ksize: int = 3,
        close_iter: int = 1,
        thinning_max_iter: int = 80,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Cached ``(bin_mask, skeleton)`` keyed by (label, raster_params, close, thinning).

        Returns ``None`` if the raster is empty/invalid.
        """
        label = int(label)
        key = (label, raster_params, (close_ksize, close_iter), int(thinning_max_iter))
        if key not in self._cluster_skeleton:
            img = self.cluster_raster(label, Xi, raster_params, bounds)
            if img is None or int(np.max(img)) == 0 or cv2 is None:
                self._cluster_skeleton[key] = None
            else:
                bin_img = (img > 0).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
                bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
                skel = zhang_suen_thinning(bin_img, max_iter=int(thinning_max_iter))
                self._cluster_skeleton[key] = (bin_img, skel)
        return self._cluster_skeleton[key]

    # ---------------------------------------------- filtered global distances
    def filtered_full_distance(self) -> Optional[np.ndarray]:
        """Distance matrix on ``X_full[noise_mask]`` sliced from the global cache.

        Built from the per-search ``distance_for(X_full)`` matrix, so calling
        this from multiple configurations only triggers one underlying
        ``pairwise_distances`` call per search. Returns ``None`` when there
        is no global context or the matrix is above the memory guard.
        """
        return self._sliced_global_distance(
            getattr(self.global_context, "X_full", None),
            cache_key="full",
        )

    def filtered_decision_distance(self) -> Optional[np.ndarray]:
        """Distance matrix on ``X_decision[noise_mask]`` sliced from the global cache."""
        return self._sliced_global_distance(
            getattr(self.global_context, "X_decision", None),
            cache_key="decision",
        )

    def _sliced_global_distance(
        self, X_ref: Optional[np.ndarray], *, cache_key: str,
    ) -> Optional[np.ndarray]:
        if X_ref is None or self.global_context is None:
            return None
        # Local memo so we don't redo the np.ix_ slice for every metric on the
        # same partition.
        if cache_key in self._noise_filtered_distance:
            return self._noise_filtered_distance[cache_key]
        if not hasattr(self.global_context, "distance_for"):
            self._noise_filtered_distance[cache_key] = None
            return None
        D_full = self.global_context.distance_for(X_ref)
        if D_full is None:
            self._noise_filtered_distance[cache_key] = None
            return None
        idx = np.where(self.noise_mask)[0]
        D = np.ascontiguousarray(D_full[np.ix_(idx, idx)])
        self._noise_filtered_distance[cache_key] = D
        return D

    # ---------------------------------------------- pairwise distances (view)
    def pairwise_distance_view(self, X: np.ndarray, *, key: Optional[Any] = None) -> Optional[np.ndarray]:
        """Pairwise distance matrix on ``X``.

        Prefers the per-search global cache (``GlobalMetricContext``) so the
        matrix is built only once per search rather than per configuration.
        Falls back to a per-partition cache only when no global context is
        attached or when ``X`` isn't one of the cached arrays.
        """
        # Per-search global cache: one matrix per (X identity).
        if self.global_context is not None and hasattr(self.global_context, "distance_for"):
            D = self.global_context.distance_for(X)
            if D is not None:
                return D
            # ``None`` means "above the memory guard"; honour that rather than
            # trying the smaller per-partition cache, which would just hit the
            # same memory cap.
            return None

        # Legacy fallback (no global context): cache per partition.
        cache_key = key if key is not None else id(X)
        if cache_key in self._pairwise_view:
            return self._pairwise_view[cache_key]
        if X.shape[0] > self.max_distance_cache_n:
            self._pairwise_view[cache_key] = None
            return None
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X)
        if D.dtype != np.float64:
            D = D.astype(np.float64, copy=False)
        self._pairwise_view[cache_key] = D
        return D

    def noise_filtered_distance(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Pairwise distance on ``X[noise_mask]``.

        Sliced from the per-search global matrix when one is cached for ``X``
        (cheap, O(n²) memory copy but no extra distance work). Falls back to
        a direct per-partition computation when no shared matrix exists.
        """
        key = id(X)
        if key in self._noise_filtered_distance:
            return self._noise_filtered_distance[key]

        # Slice from the per-search global cache when available.
        if self.global_context is not None and hasattr(self.global_context, "distance_for"):
            unfiltered = self.global_context.distance_for(X)
            if unfiltered is not None:
                idx = np.where(self.noise_mask)[0]
                D = np.ascontiguousarray(unfiltered[np.ix_(idx, idx)])
                self._noise_filtered_distance[key] = D
                return D

        # Direct compute fallback.
        mask = self.noise_mask
        Xf = X[mask]
        if Xf.shape[0] > self.max_distance_cache_n:
            self._noise_filtered_distance[key] = None
            return None
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(Xf)
        if D.dtype != np.float64:
            D = D.astype(np.float64, copy=False)
        self._noise_filtered_distance[key] = D
        return D
