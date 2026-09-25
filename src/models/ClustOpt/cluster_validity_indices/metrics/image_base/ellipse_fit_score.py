from __future__ import annotations

import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import (
    RasterParams,
    rasterize_points,
    canny_edges,
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


def _ellipse_support_score(edges: np.ndarray, ellipse, tol_px: int = 2, n_samples: int = 240) -> float:
    """
    דוגמים נקודות על האליפסה ובודקים hit על edges (עם dilation בטולרנס).
    """
    H, W = edges.shape[:2]
    if tol_px > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
        E = cv2.dilate(edges, ker, iterations=1)
    else:
        E = edges

    (cx, cy), (a, b), angle = ellipse  # a,b are full axis lengths
    if a <= 1.0 or b <= 1.0:
        return 0.0

    # ellipse2Poly expects half-axes:
    pts = cv2.ellipse2Poly(
        (int(round(cx)), int(round(cy))),
        (int(round(a / 2.0)), int(round(b / 2.0))),
        int(round(angle)),
        0,
        360,
        max(1, int(round(360 / max(60, n_samples)))),
    )

    hits = 0
    total = 0
    for x, y in pts:
        if 0 <= x < W and 0 <= y < H:
            total += 1
            if E[int(y), int(x)] > 0:
                hits += 1
    if total == 0:
        return 0.0
    return float(hits) / float(total)


class EllipseFitScoreMetric(ImagePerClusterMetric):
    """
    Ellipse Fit Score (per-cluster).
    score in [0,1] based on edge support for best-fit ellipse.
    """

    name = "ellipse_fit_score"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 80,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        canny_low: int = 60,
        canny_high: int = 160,
        support_tol: int = 2,
        support_samples: int = 240,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.support_tol = int(max(0, support_tol))
        self.support_samples = int(max(60, support_samples))

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.ndim != 2 or Xi.shape[1] != 2:
            return float("-inf")
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or np.max(img) == 0:
            return 0.0

        # contour from filled mask (binary)
        bin_img = (img > 0).astype(np.uint8) * 255
        cnt = _largest_contour(bin_img)
        if cnt is None or len(cnt) < 5:
            return 0.0

        try:
            ellipse = cv2.fitEllipse(cnt)  # ((cx,cy),(a,b),angle)
        except Exception:
            return 0.0

        edges = canny_edges(img, low=self.canny_low, high=self.canny_high)
        if int(np.count_nonzero(edges)) < 30:
            return 0.0

        sup = _ellipse_support_score(edges, ellipse, tol_px=self.support_tol, n_samples=self.support_samples)
        return float(np.clip(sup, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        return float(raw)
