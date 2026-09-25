from __future__ import annotations

import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import (
    RasterParams,
    rasterize_points,
    Bounds2D,
)
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


def _largest_contour(bin_img: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) <= 1.0:
        return None
    return cnt


class CircularityCompactnessMetric(ImagePerClusterMetric):
    """
    Circularity / Compactness (per-cluster).
    circularity = 4*pi*A / P^2 in (0,1] (ideal circle -> 1)
    """

    name = "circularity_compactness"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 60,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        eps: float = 1e-9,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.eps = float(eps)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.ndim != 2 or Xi.shape[1] != 2:
            return float("-inf")
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or np.max(img) == 0:
            return 0.0

        bin_img = (img > 0).astype(np.uint8) * 255
        cnt = _largest_contour(bin_img)
        if cnt is None or len(cnt) < 3:
            return 0.0

        A = float(cv2.contourArea(cnt))
        P = float(cv2.arcLength(cnt, True))
        if A <= 0.0 or P <= 0.0:
            return 0.0

        circ = (4.0 * np.pi * A) / (P * P + self.eps)
        return float(np.clip(circ, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.5)
