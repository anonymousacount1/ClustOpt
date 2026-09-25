# models/ClustOpt/cluster_validity_indices/metrics/image_base/skeleton_connectivity.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import (
    RasterParams,
    rasterize_points,
    Bounds2D,
    compute_global_bounds,
)
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext

# Vectorised thinning lives in the shared context module so the same fast
# implementation powers both skeleton_connectivity and thickness_uniformity
# without code duplication.
from models.ClustOpt.contexts.partition_metric_context import zhang_suen_thinning


def _skeleton_graph_stats(skel: np.ndarray) -> tuple[int, int, int, float]:
    """
    Returns: components, endpoints, junctions, coverage
    coverage = skeleton_pixels / foreground_pixels (proxy)
    """
    sk = (skel > 0).astype(np.uint8)
    n_skel = int(np.count_nonzero(sk))
    if n_skel == 0:
        return 0, 0, 0, 0.0

    # connected components on skeleton
    num_labels, _ = cv2.connectedComponents(sk, connectivity=8)
    components = int(max(0, num_labels - 1))

    # Degree per skeleton pixel = (sum over 3x3 neighbourhood) - 1.
    # Use zero-padded convolution to match the original implementation, which
    # clamped slices at the image border (i.e. treated off-image neighbours as 0).
    from scipy.ndimage import convolve
    _kernel = np.ones((3, 3), dtype=np.int32)
    nb_sum = convolve(sk.astype(np.int32), _kernel, mode="constant", cval=0)
    deg = nb_sum - 1
    deg_on_skel = deg[sk == 1]
    endpoints = int(np.count_nonzero(deg_on_skel == 1))
    junctions = int(np.count_nonzero(deg_on_skel >= 3))

    # coverage proxy: skeleton length normalized by bbox area
    ys, xs = np.nonzero(sk)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    area = float((x_max - x_min + 1) * (y_max - y_min + 1))
    coverage = float(n_skel / max(1.0, area))

    return components, endpoints, junctions, coverage


class SkeletonConnectivityMetric(ImagePerClusterMetric):
    """
    Skeleton Connectivity (per-cluster)

    High when:
    - skeleton is mostly one connected component
    - few endpoints (~2 for a single arc/curve), low junctions
    - decent skeleton coverage (not empty / not tiny)

    score = exp(-a*(c-1)) * exp(-b*max(0,endpoints-2)) * exp(-c*junctions) * sat(coverage)
    """

    name = "skeleton_connectivity"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 80,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 2,
        thinning_max_iter: int = 80,
        a_comp: float = 0.40,
        b_end: float = 0.005,
        c_junc: float = 0.010,
        close_ksize: int = 3,
        close_iter: int = 1,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.thinning_max_iter = int(thinning_max_iter)
        self.a_comp = float(a_comp)
        self.b_end = float(b_end)
        self.c_junc = float(c_junc)
        self.close_ksize = int(close_ksize)
        self.close_iter = int(close_iter)

    # --------------------------------------------------------- shared scoring
    def _score_from_skeleton(self, skel: np.ndarray) -> float:
        comps, ends, juncs, cov = _skeleton_graph_stats(skel)
        if comps == 0:
            return 0.0
        comp_term = float(np.exp(-self.a_comp * max(0, comps - 1)))
        end_term = float(np.exp(-self.b_end * max(0, ends - 2)))
        j_term = float(np.exp(-self.c_junc * max(0, juncs)))
        cov_term = float(1.0 - np.exp(-8.0 * max(0.0, cov)))
        score = comp_term * end_term * j_term * cov_term
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------- legacy non-context fallback
    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or int(np.max(img)) == 0:
            return 0.0
        bin_img = (img > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_ksize, self.close_ksize))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter)
        skel = zhang_suen_thinning(bin_img, max_iter=self.thinning_max_iter)
        return self._score_from_skeleton(skel)

    # --------------------------------------------- context-aware fast path
    def _cluster_score_ctx(
        self, label: int, Xi: np.ndarray, bounds: Bounds2D, partition_context,
    ) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        cached = partition_context.cluster_skeleton(
            label, Xi, self.raster_params, bounds,
            close_ksize=self.close_ksize,
            close_iter=self.close_iter,
            thinning_max_iter=self.thinning_max_iter,
        )
        if cached is None:
            return 0.0
        _bin, skel = cached
        return self._score_from_skeleton(skel)

    # ---------------------------------------------- context-aware contract
    def evaluate_ctx(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        context: Optional[Any] = None,
        partition_context: Optional[Any] = None,
    ) -> float:
        if partition_context is None:
            return self.evaluate(X, labels)

        X = np.asarray(X); labels = np.asarray(labels)
        if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.shape[0] or X.shape[1] < 2:
            return float("-inf")
        uniq = np.unique(labels)
        if uniq.size < 1:
            return float("-inf")

        # Reuse the shared global bounds when available; otherwise compute locally.
        if context is not None:
            bounds = context.bounds(space="full")
        else:
            bounds = compute_global_bounds(X)

        total = 0.0
        total_w = 0.0
        for lab in uniq:
            Xi = X[labels == lab]
            if Xi.shape[0] < int(self.min_cluster_points):
                continue
            s = self._cluster_score_ctx(int(lab), Xi, bounds, partition_context)
            if not np.isfinite(s) or s == float("-inf"):
                continue
            w = float(Xi.shape[0])
            total += w * float(s)
            total_w += w
        return float("-inf") if total_w <= 0 else float(total / total_w)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.3)
