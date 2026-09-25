# models/ClustOpt/cluster_validity_indices/metrics/image_base/glcm_haralick.py
from __future__ import annotations
import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import RasterParams, rasterize_points, Bounds2D
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


def _quantize(img: np.ndarray, levels: int) -> np.ndarray:
    img_f = img.astype(np.float32)
    mx = float(np.max(img_f))
    if mx <= 0:
        return np.zeros_like(img, dtype=np.uint8)
    img_f = img_f / mx
    q = np.floor(img_f * (levels - 1) + 1e-6).astype(np.uint8)
    return q


def _glcm_features(q: np.ndarray, levels: int, dx: int, dy: int) -> tuple[float, float, float]:
    """
    Returns (contrast, homogeneity, energy) computed from symmetric normalized GLCM.
    """
    H, W = q.shape
    x0 = max(0, -dx)
    x1 = min(W, W - dx)
    y0 = max(0, -dy)
    y1 = min(H, H - dy)

    a = q[y0:y1, x0:x1].astype(np.int32)
    b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.int32)

    if a.size < 50:
        return 0.0, 0.0, 0.0

    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (a.ravel(), b.ravel()), 1.0)
    glcm = glcm + glcm.T
    s = float(glcm.sum())
    if s <= 0:
        return 0.0, 0.0, 0.0
    P = glcm / s

    i = np.arange(levels, dtype=np.float64)
    I, J = np.meshgrid(i, i, indexing="ij")
    diff2 = (I - J) ** 2

    contrast = float(np.sum(P * diff2))
    homogeneity = float(np.sum(P / (1.0 + np.abs(I - J))))
    energy = float(np.sum(P * P))
    return contrast, homogeneity, energy


class GLCMHaralickMetric(ImagePerClusterMetric):
    """
    GLCM/Haralick texture coherence (per-cluster).

    Compute GLCM on quantized density image for a few offsets and combine:
    score high when: homogeneity high + energy high + contrast low.
    """

    name = "glcm_haralick"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 120,
        grid_size: int = 256,
        raster_mode: str = "density",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        levels: int = 16,
        offsets: list[tuple[int, int]] | None = None,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.levels = int(max(8, levels))
        self.offsets = offsets or [(1, 0), (0, 1), (2, 0), (0, 2)]

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or float(np.max(img)) <= 0.0:
            return 0.0

        q = _quantize(img, self.levels)

        feats = []
        for dx, dy in self.offsets:
            c, h, e = _glcm_features(q, self.levels, dx, dy)
            feats.append((c, h, e))
        if not feats:
            return 0.0

        contrast = float(np.mean([f[0] for f in feats]))
        hom = float(np.mean([f[1] for f in feats]))
        energy = float(np.mean([f[2] for f in feats]))

        # normalize-ish:
        # contrast in [0, (levels-1)^2], map to [0,1] with saturating
        c_norm = float(1.0 - np.exp(-contrast / (0.35 * (self.levels - 1) ** 2 + 1e-9)))
        # hom in (0,1], energy in (0,1]
        score = (1.0 - c_norm) * float(np.clip(hom, 0.0, 1.0)) * float(np.clip(np.sqrt(energy), 0.0, 1.0))
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.5)
