# models/ClustOpt/cluster_validity_indices/metrics/image_base/edge_coherence_structure_tensor.py
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


class EdgeCoherenceStructureTensorMetric(ImagePerClusterMetric):
    """
    Edge Coherence (Structure Tensor) — per-cluster.

    מבוסס על Coherence של הטנזור המקומי:
        coh = (λ1-λ2)/(λ1+λ2) ∈ [0,1]
    ממוצע על פיקסלים עם גרדיאנט חזק.
    """

    name = "edge_coherence"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 50,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        smooth_sigma: float = 1.2,
        mag_thr_quantile: float = 0.60,
        eps: float = 1e-9,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,           # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.smooth_sigma = float(max(0.0, smooth_sigma))
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

        mag = cv2.magnitude(gx, gy)
        m = mag.reshape(-1)
        if m.size == 0:
            return 0.0

        thr = float(np.quantile(m, self.mag_thr_quantile))
        mask = mag > max(thr, self.eps)
        if int(np.count_nonzero(mask)) < 30:
            return 0.0

        Jxx = gx * gx
        Jyy = gy * gy
        Jxy = gx * gy

        if self.smooth_sigma > 0.0:
            k = int(max(3, 2 * round(3 * self.smooth_sigma) + 1))
            Jxx = cv2.GaussianBlur(Jxx, (k, k), self.smooth_sigma)
            Jyy = cv2.GaussianBlur(Jyy, (k, k), self.smooth_sigma)
            Jxy = cv2.GaussianBlur(Jxy, (k, k), self.smooth_sigma)

        # Aggregate structure tensor over the masked (high-gradient) region
        # so that consistently-oriented textures produce a clearly aligned
        # global tensor while chaotic textures cancel out.
        Jxx_sum = float(np.sum(Jxx[mask]))
        Jyy_sum = float(np.sum(Jyy[mask]))
        Jxy_sum = float(np.sum(Jxy[mask]))

        tr = Jxx_sum + Jyy_sum
        det = Jxx_sum * Jyy_sum - Jxy_sum * Jxy_sum
        disc = max(tr * tr - 4.0 * det, 0.0)
        s = float(np.sqrt(disc))

        lam1 = 0.5 * (tr + s)
        lam2 = 0.5 * (tr - s)
        denom = max(lam1 + lam2, self.eps)
        score = (lam1 - lam2) / denom
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.5)
