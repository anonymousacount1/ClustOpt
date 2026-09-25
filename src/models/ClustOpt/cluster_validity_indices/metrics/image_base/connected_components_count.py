# models/ClustOpt/cluster_validity_indices/metrics/image_base/connected_components_count.py
from __future__ import annotations
import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import RasterParams, rasterize_points, Bounds2D
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


class ConnectedComponentsCountMetric(ImagePerClusterMetric):
    """
    Connected Components Count (per-cluster)

    Score high when there is ~1 connected component in the filled mask.
    score = exp(-a*(components-1)) * coverage_term
    """

    name = "connected_components_count"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 120,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 2,
        a_comp: float = 0.85,
        min_foreground: int = 60,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.a_comp = float(a_comp)
        self.min_foreground = int(min_foreground)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")
        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or int(np.max(img)) == 0:
            return 0.0

        # Threshold against a fraction of the peak intensity so soft Gaussian
        # tails (which speckle into many tiny pieces after blur+dilate) don't
        # blow up the count.
        peak = float(np.max(img))
        thr = max(1.0, 0.30 * peak)
        mask = (img >= thr).astype(np.uint8)

        # Aggressively close gaps so a dense blob with scattered tail pixels
        # is treated as a single component.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        fg = int(np.count_nonzero(mask))
        if fg < self.min_foreground:
            return 0.0

        # Count only components that are at least 1% of the foreground -- this
        # filters out tiny rasterisation specks while still seeing real
        # disjoint pieces of a fragmented cluster.
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        if num_labels <= 1:
            return 0.0
        min_comp_area = max(20, int(0.01 * fg))
        comps = 0
        for lab in range(1, num_labels):
            if int(np.count_nonzero(labels == lab)) >= min_comp_area:
                comps += 1
        if comps <= 0:
            return 0.0

        # Score purely on component count.
        score = float(np.exp(-self.a_comp * max(0, comps - 1)))
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.4)
