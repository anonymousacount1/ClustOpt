# models/ClustOpt/cluster_validity_indices/metrics/image_base/euler_holes_count.py
from __future__ import annotations
import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import RasterParams, rasterize_points, Bounds2D
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


def _count_holes(mask255: np.ndarray) -> int:
    """
    Count holes using contour hierarchy (CCOMP).
    Holes are child contours.
    """
    cnts, hier = cv2.findContours(mask255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return 0
    hier = hier[0]
    holes = 0
    for i in range(len(cnts)):
        parent = hier[i][3]
        if parent != -1:
            holes += 1
    return int(holes)


class EulerHolesCountMetric(ImagePerClusterMetric):
    """
    Euler / Holes Count (per-cluster)

    We detect holes in the filled mask.
    Score favors having 1-2 meaningful holes (ring/crescent), penalizes many holes (noise).
    If you prefer "more holes => higher", we can flip easily later.
    """

    name = "euler_holes_count"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 150,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 2,
        close_iters: int = 1,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.close_iters = int(close_iters)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or int(np.max(img)) == 0:
            return 0.0

        mask = (img > 0).astype(np.uint8) * 255
        # Heavier morphological closing first to remove rasterisation pinholes
        # so the hole count is dominated by real topological holes, not noise.
        if self.close_iters > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=max(2, self.close_iters))

        # Drop very small holes (likely noise) by filtering on area.
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or len(cnts) == 0:
            return 0.0
        hier = hier[0]
        min_hole_area = max(8.0, 0.0008 * mask.size)
        holes = 0
        for i in range(len(cnts)):
            parent = hier[i][3]
            if parent == -1:
                continue
            if cv2.contourArea(cnts[i]) >= min_hole_area:
                holes += 1
        if holes <= 0:
            return 0.0

        # Reward presence of a clear hole. Gently penalise an abundance of holes
        # (suggesting noise) but never collapse the score to zero.
        up = float(1.0 - np.exp(-1.5 * holes))
        penalty = 1.0 / (1.0 + 0.15 * max(0, holes - 2))
        score = up * penalty
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.5)
