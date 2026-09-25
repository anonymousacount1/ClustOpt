# models/ClustOpt/cluster_validity_indices/metrics/image_base/orientation_histogram_entropy.py
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


def _entropy01(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    H = -float(np.sum(p * np.log(p + eps)))
    Hmax = float(np.log(max(2, int(p.size))))
    return float(np.clip(H / (Hmax + eps), 0.0, 1.0))


class OrientationHistogramEntropyMetric(ImagePerClusterMetric):
    """
    Orientation Histogram Entropy — per-cluster.

    מחשבים גרדיאנטים (Sobel), בונים היסטוגרמת זוויות (משוקללת לפי magnitude),
    entropy נמוך => כיוון דומיננטי ברור => score גבוה:
        score = 1 - entropy_norm
    """

    name = "orientation_histogram_entropy"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 40,
        grid_size: int = 256,
        raster_mode: str = "binary",     # "binary" or "density"
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        n_bins: int = 18,
        mag_thr_quantile: float = 0.60,
        eps: float = 1e-12,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,          # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.n_bins = int(max(6, n_bins))
        self.mag_thr_quantile = float(np.clip(mag_thr_quantile, 0.0, 0.95))
        self.eps = float(eps)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < int(self.min_cluster_points):
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or np.max(img) == 0:
            return 0.0

        img_f = img.astype(np.float32) / 255.0
        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

        mag = cv2.magnitude(gx, gy).reshape(-1)
        if mag.size == 0:
            return 0.0

        thr = float(np.quantile(mag, self.mag_thr_quantile))
        mask = mag > max(thr, self.eps)
        if int(np.count_nonzero(mask)) < 20:
            return 0.0

        ang = cv2.phase(gx, gy, angleInDegrees=False).reshape(-1)[mask]  # [0,2pi)
        w = mag[mask].astype(float)

        # orientation-invariant: theta and theta+pi same => use 2*theta trick.
        # Vector strength R = sqrt(<cos(2*theta)>^2 + <sin(2*theta)>^2) is a
        # numerically stable measure of how concentrated the orientation
        # distribution is. R = 1 -> one dominant orientation; R = 0 -> uniform.
        wsum = float(np.sum(w))
        if wsum <= self.eps:
            return 0.0
        wn = w / wsum
        c = float(np.sum(wn * np.cos(2.0 * ang)))
        s = float(np.sum(wn * np.sin(2.0 * ang)))
        R = float(np.clip(np.sqrt(c * c + s * s), 0.0, 1.0))
        return float(R)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.4)
