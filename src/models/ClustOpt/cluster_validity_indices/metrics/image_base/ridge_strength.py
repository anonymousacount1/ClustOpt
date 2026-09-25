# models/ClustOpt/cluster_validity_indices/metrics/image_base/ridge_strength.py
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


def _vesselness_frangi_like(img: np.ndarray, sigmas: list[float], beta: float = 0.5, c: float = 15.0) -> np.ndarray:
    """
    Simplified Frangi-like ridge response:
    - compute Hessian at multiple scales
    - eigenvalues l1,l2 (|l1|<=|l2|)
    - prefer ridges: l2 negative, |l2| large, low blobness (|l1|/|l2| small)
    returns response in [0,1] (approximately).
    """
    img_f = img.astype(np.float32)
    img_f = img_f / float(max(1.0, np.max(img_f)))

    H, W = img_f.shape[:2]
    best = np.zeros((H, W), dtype=np.float32)

    for s in sigmas:
        s = float(max(0.8, s))
        blur = cv2.GaussianBlur(img_f, (0, 0), s)

        dxx = cv2.Sobel(blur, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(blur, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(blur, cv2.CV_32F, 1, 1, ksize=3)

        # scale normalization (approx)
        dxx *= (s * s)
        dyy *= (s * s)
        dxy *= (s * s)

        # eigenvalues of 2x2 Hessian
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        tmp = np.maximum(0.0, (trace * trace) / 4.0 - det)
        delta = np.sqrt(tmp)

        l1 = trace / 2.0 - delta
        l2 = trace / 2.0 + delta

        # ensure |l1| <= |l2|
        swap = np.abs(l1) > np.abs(l2)
        l1s = l1.copy()
        l2s = l2.copy()
        l1[swap] = l2s[swap]
        l2[swap] = l1s[swap]

        # blobness + structureness
        eps = 1e-9
        rb = (np.abs(l1) / (np.abs(l2) + eps)) ** 2
        s2 = l1 * l1 + l2 * l2

        # prefer dark ridges on bright background: l2 < 0 (tunable)
        cond = l2 < 0
        v = np.zeros_like(best)
        v[cond] = np.exp(-rb[cond] / (2.0 * beta * beta)) * (1.0 - np.exp(-s2[cond] / (2.0 * c * c)))

        best = np.maximum(best, v)

    best = np.clip(best, 0.0, 1.0)
    return best


class RidgeStrengthMetric(ImagePerClusterMetric):
    """
    Ridge Strength (per-cluster)

    - rasterize -> grayscale/density
    - compute multi-scale frangi-like vesselness
    - score = mean of top-q percentile responses (robust to background)

    High when a cluster contains strong ridge-like structures (elongated).
    """

    name = "ridge_strength"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 80,
        grid_size: int = 256,
        raster_mode: str = "density",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        sigmas: list[float] | None = None,
        top_quantile: float = 0.95,
        beta: float = 0.5,
        c: float = 15.0,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.sigmas = sigmas or [1.0, 1.6, 2.4, 3.4]
        self.top_quantile = float(top_quantile)
        self.beta = float(beta)
        self.c = float(c)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or float(np.max(img)) <= 0.0:
            return 0.0

        resp = _vesselness_frangi_like(img, sigmas=self.sigmas, beta=self.beta, c=self.c)
        if float(np.max(resp)) <= 0.0:
            return 0.0

        # Sum the strong-vesselness fraction of the image: this favours
        # extended tubular structures (where many pixels have high
        # vesselness) over blobs (where only a tiny region has any
        # significant response).
        thr = 0.25 * float(np.max(resp))
        strong_count = int(np.count_nonzero(resp >= thr))
        # Normalise by the rasterised foreground area so a small but tubular
        # cluster is not penalised.
        fg = int(np.count_nonzero(img > 0))
        if fg < 30:
            return 0.0
        ratio = float(strong_count) / float(fg)
        # Saturate so ~0.08 strong-fraction (typical of a real ridge) maps to ~0.8.
        score = float(1.0 - np.exp(-ratio / 0.045))
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.6)
