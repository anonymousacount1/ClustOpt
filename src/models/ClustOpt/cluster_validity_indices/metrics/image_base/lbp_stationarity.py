# models/ClustOpt/cluster_validity_indices/metrics/image_base/lbp_stationarity.py
from __future__ import annotations
import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import RasterParams, rasterize_points, Bounds2D
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


def _lbp8(img_u8: np.ndarray) -> np.ndarray:
    """
    Basic 8-neighbor LBP. Returns codes in [0,255].
    """
    img = img_u8
    H, W = img.shape
    c = img[1:H-1, 1:W-1].astype(np.int16)
    code = np.zeros((H-2, W-2), dtype=np.uint8)

    nbrs = [
        img[0:H-2, 0:W-2], img[0:H-2, 1:W-1], img[0:H-2, 2:W],
        img[1:H-1, 2:W],   img[2:H,   2:W],   img[2:H,   1:W-1],
        img[2:H,   0:W-2], img[1:H-1, 0:W-2],
    ]
    for k, nb in enumerate(nbrs):
        code |= ((nb.astype(np.int16) >= c) << (7 - k)).astype(np.uint8)
    return code


def _hist256(x: np.ndarray) -> np.ndarray:
    h = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    s = float(h.sum())
    return h / s if s > 0 else h


def _chi2(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    d = p - q
    return float(0.5 * np.sum((d * d) / (p + q + eps)))


class LBPStationarityMetric(ImagePerClusterMetric):
    """
    LBP Stationarity (per-cluster)

    - Compute LBP codes.
    - Split image into tiles; compute LBP hist per tile.
    - Stationarity high when tile histograms are similar (low mean chi-square).
    """

    name = "lbp_stationarity"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 150,
        grid_size: int = 256,
        raster_mode: str = "density",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        tiles: int = 4,
        min_foreground_frac: float = 0.02,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.tiles = int(max(2, tiles))
        self.min_foreground_frac = float(min_foreground_frac)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or float(np.max(img)) <= 0.0:
            return 0.0

        # normalize to uint8
        img_f = img.astype(np.float32)
        img_u8 = (255.0 * (img_f / float(np.max(img_f)))).astype(np.uint8)

        fg = (img_u8 > 0)
        if float(np.mean(fg)) < self.min_foreground_frac:
            return 0.0

        lbp = _lbp8(img_u8)  # (H-2,W-2)
        H, W = lbp.shape
        t = self.tiles
        hs = []
        for iy in range(t):
            for ix in range(t):
                y0 = int(round(iy * H / t))
                y1 = int(round((iy + 1) * H / t))
                x0 = int(round(ix * W / t))
                x1 = int(round((ix + 1) * W / t))
                tile = lbp[y0:y1, x0:x1]
                if tile.size < 100:
                    continue
                hs.append(_hist256(tile))
        if len(hs) < 2:
            return 0.0

        # mean pairwise chi2 vs global mean
        h_mean = np.mean(np.stack(hs, axis=0), axis=0)
        d = float(np.mean([_chi2(h, h_mean) for h in hs]))
        score = float(1.0 / (1.0 + 6.0 * d))  # scale to [0,1]
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.7)
