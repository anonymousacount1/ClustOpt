# models/ClustOpt/cluster_validity_indices/metrics/image_base/symmetry_score.py
from __future__ import annotations
import numpy as np

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import RasterParams, rasterize_points, Bounds2D
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


def _sym(img: np.ndarray, flipped: np.ndarray) -> float:
    a = img.astype(np.float32)
    b = flipped.astype(np.float32)
    ma = float(np.max(a))
    mb = float(np.max(b))
    if ma > 0: a = a / ma
    if mb > 0: b = b / mb
    denom = float(np.mean(a) + 1e-9)
    diff = float(np.mean(np.abs(a - b)))
    # invert and normalize
    s = 1.0 - diff / (denom + 1e-9)
    return float(np.clip(s, 0.0, 1.0))


class SymmetryScoreMetric(ImagePerClusterMetric):
    """
    Symmetry Score (per-cluster)

    score = max(horizontal symmetry, vertical symmetry).
    """

    name = "symmetry_score"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 120,
        grid_size: int = 256,
        raster_mode: str = "density",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or float(np.max(img)) <= 0.0:
            return 0.0

        sh = _sym(img, np.flipud(img))
        sv = _sym(img, np.fliplr(img))
        return float(max(sh, sv))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.45)
