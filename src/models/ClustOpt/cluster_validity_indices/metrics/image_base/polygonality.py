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


def _fill_contour_mask(shape_hw: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1)
    return mask


def _fill_poly_mask(shape_hw: tuple[int, int], poly: np.ndarray) -> np.ndarray:
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = (a > 0)
    b = (b > 0)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / (union + eps)


class PolygonalityMetric(ImagePerClusterMetric):
    """
    Polygonality (per-cluster).
    1) approxPolyDP -> polygon approximation
    2) IoU(contour_fill, polygon_fill)
    3) simplicity term based on #vertices (few vertices => higher)
    score = IoU * simplicity  in [0,1]
    """

    name = "polygonality"
    space = "full"
    higher_is_better = True

    def __init__(self, min_cluster_points: int = 120, grid_size: int = 256, raster_mode: str = "binary",
                 blur_ksize: int = 3, dilate_iters: int = 1, approx_eps_frac: float = 0.02, max_vertices: int = 12,
                 eps: float = 1e-9):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.approx_eps_frac = float(max(1e-4, approx_eps_frac))
        self.max_vertices = int(max(4, max_vertices))
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

        per = float(cv2.arcLength(cnt, True))
        if per <= 0.0:
            return 0.0

        eps = self.approx_eps_frac * per
        poly = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
        nV = int(len(poly))
        if nV < 3:
            return 0.0

        H, W = bin_img.shape[:2]
        m_cnt = _fill_contour_mask((H, W), cnt)
        m_poly = _fill_poly_mask((H, W), poly)

        fit = _iou(m_cnt, m_poly, eps=self.eps)

        # simplicity: 3 vertices best, more vertices reduces score
        # map nV=3 -> 1, nV=max_vertices -> ~0
        nV_cap = min(nV, self.max_vertices)
        simp = 1.0 - (nV_cap - 3) / max(1, (self.max_vertices - 3))
        simp = float(np.clip(simp, 0.0, 1.0))

        score = fit * simp
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        return float(raw)
