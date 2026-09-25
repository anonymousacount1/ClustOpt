# models/ClustOpt/cluster_validity_indices/metrics/image_base/peak_to_background_hough.py
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


class PeakToBackgroundHoughMetric(ImagePerClusterMetric):
    """
    Peak-to-Background (Hough) — per-cluster.

    רעיון:
    - עושים HoughLinesP על edges של כל קלאסטר.
    - מודדים "דומיננטיות" של הקו החזק ביותר לעומת שאר הקווים:
        score = peak_len / (peak_len + mean_other_len + eps)
      כך קלאסטר עם קו/מבנה לינארי חזק יקבל ציון גבוה,
      וקלאסטר עם הרבה קווים חלשים/רעש יקבל נמוך.
    """

    name = "peak_to_background_hough"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 30,
        grid_size: int = 256,
        raster_mode: str = "binary",     # "binary" or "density"
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_thresh: int = 40,
        min_line_len_frac: float = 0.12,
        max_line_gap_frac: float = 0.03,
        eps: float = 1e-9,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,         # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.hough_thresh = int(hough_thresh)
        self.min_line_len_frac = float(min_line_len_frac)
        self.max_line_gap_frac = float(max_line_gap_frac)
        self.eps = float(eps)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < int(self.min_cluster_points):
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or np.max(img) == 0:
            return 0.0

        edges = canny_edges(img, low=self.canny_low, high=self.canny_high)
        H, W = edges.shape[:2]
        if int(np.count_nonzero(edges)) < 10:
            return 0.0

        min_len = int(max(5, round(self.min_line_len_frac * min(H, W))))
        max_gap = int(max(1, round(self.max_line_gap_frac * min(H, W))))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(self.hough_thresh),
            minLineLength=min_len,
            maxLineGap=max_gap,
        )
        if lines is None or len(lines) == 0:
            return 0.0

        lens = []
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [int(v) for v in l]
            lens.append(float(np.hypot(x2 - x1, y2 - y1)))

        if not lens:
            return 0.0

        lens = np.asarray(lens, dtype=float)
        peak = float(np.max(lens))
        if lens.size == 1:
            return 1.0

        others = lens[lens < peak - 1e-6]
        bg = float(np.mean(others)) if others.size > 0 else float(np.mean(lens))
        # Contrast-style ratio (peak - bg) / (peak + bg). Spreads naturally
        # over [0, 1] -- 0 when many lines are equally strong, 1 when one
        # dominant line clearly stands out above the rest.
        score = (peak - bg) / (peak + bg + self.eps)
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.7)
