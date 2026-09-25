# models/ClustOpt/cluster_validity_indices/metrics/image_base/fractal_dimension.py
from __future__ import annotations
import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import RasterParams, rasterize_points, canny_edges, Bounds2D
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


def _boxcount(Z: np.ndarray, k: int) -> int:
    H, W = Z.shape
    Hk = (H // k) * k
    Wk = (W // k) * k
    if Hk <= 0 or Wk <= 0:
        return 0
    Zc = Z[:Hk, :Wk]
    # reshape into blocks and count non-empty
    blocks = Zc.reshape(Hk // k, k, Wk // k, k)
    nonempty = np.any(blocks, axis=(1, 3))
    return int(np.count_nonzero(nonempty))


def _fractal_dimension(binary: np.ndarray, sizes: list[int]) -> float:
    # binary: bool
    counts = []
    ss = []
    for k in sizes:
        c = _boxcount(binary, k)
        if c > 0:
            counts.append(c)
            ss.append(k)
    if len(counts) < 2:
        return 0.0
    x = np.log(1.0 / np.array(ss, dtype=np.float64))
    y = np.log(np.array(counts, dtype=np.float64))
    # linear fit
    a, _b = np.polyfit(x, y, 1)
    return float(a)


class FractalDimensionMetric(ImagePerClusterMetric):
    """
    Fractal Dimension (per-cluster)

    We compute on edge map (Canny) to reflect structural complexity.
    Score high when dimension is closer to 1 (smooth curve/line),
    low when closer to 2 (cloudy/filled/noisy).
    """

    name = "fractal_dimension"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 120,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 2,
        canny_low: int = 50,
        canny_high: int = 150,
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

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or int(np.max(img)) == 0:
            return 0.0

        edges = canny_edges(img, low=self.canny_low, high=self.canny_high)
        E = (edges > 0)
        if int(np.count_nonzero(E)) < 40:
            return 0.0

        # box sizes (powers of 2)
        H, W = E.shape
        m = int(min(H, W))
        sizes = []
        k = 2
        while k <= m // 2:
            sizes.append(k)
            k *= 2
        if len(sizes) < 2:
            return 0.0

        D = _fractal_dimension(E, sizes=sizes)  # ~[1,2]
        if D <= 0:
            return 0.0

        # map: D=1 -> 1, D=2 -> 0
        score = float(np.clip(2.0 - D, 0.0, 1.0))
        return score

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.6)
