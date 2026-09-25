# models/ClustOpt/cluster_validity_indices/metrics/image_base/thickness_uniformity.py
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
from models.ClustOpt.contexts.partition_metric_context import zhang_suen_thinning


class ThicknessUniformityMetric(ImagePerClusterMetric):
    """
    Thickness Uniformity (per-cluster)

    Steps:
    1) rasterize -> binary mask
    2) close to fill small gaps
    3) distance transform on mask -> radius per pixel
    4) skeletonize mask -> sample DT on skeleton -> thickness = 2*radius
    5) uniformity = 1/(1+CV(thickness))

    High when ribbon thickness is consistent along the shape.
    """

    name = "thickness_uniformity"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 120,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 2,
        thinning_max_iter: int = 80,
        min_skeleton_pixels: int = 30,
        close_ksize: int = 5,
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
        self.min_skeleton_pixels = int(min_skeleton_pixels)
        self.close_ksize = int(close_ksize)
        self.close_iter = int(close_iter)

    # ---------------------------------------------- shared scoring routine
    def _score_from_mask_and_skel(self, mask: np.ndarray, skel: np.ndarray) -> float:
        dt = cv2.distanceTransform(
            (mask > 0).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3,
        )
        if float(np.max(dt)) <= 0.0:
            return 0.0
        sk = (skel > 0)
        if int(np.count_nonzero(sk)) < self.min_skeleton_pixels:
            return 0.0
        thickness = 2.0 * dt[sk]
        thickness = thickness[np.isfinite(thickness)]
        if thickness.size < self.min_skeleton_pixels:
            return 0.0
        mean = float(np.mean(thickness))
        std = float(np.std(thickness))
        if mean <= 1e-6:
            return 0.0
        cv = std / (mean + 1e-9)
        score = float(1.0 / (1.0 + cv))
        return float(np.clip(score, 0.0, 1.0))

    # --------------------------------------- legacy non-context fallback
    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or int(np.max(img)) == 0:
            return 0.0
        mask = (img > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_ksize, self.close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter)
        skel = zhang_suen_thinning(mask, max_iter=self.thinning_max_iter)
        return self._score_from_mask_and_skel(mask, skel)

    # --------------------------------------- context-aware fast path
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
        bin_mask, skel = cached
        return self._score_from_mask_and_skel(bin_mask, skel)

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
        return float(v ** 0.7)
