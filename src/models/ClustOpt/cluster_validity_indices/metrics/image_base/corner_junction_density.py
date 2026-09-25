# metrics/image_base/corner_junction_density.py
from __future__ import annotations

import numpy as np

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import (
    RasterParams,
    rasterize_points,
    canny_edges,
    Bounds2D,
)
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext

try:
    import cv2
except Exception:
    cv2 = None


class CornerJunctionDensityMetric(ImagePerClusterMetric):
    """
    Corner / Junction Density — per-cluster image metric.

    What it captures:
    - Polygons (rectangles/triangles), "steps" (ladder-like), and intersections generate corners/junctions.
    - Smooth curves (arcs/ellipses) and uniform clouds produce fewer stable corners.

    Implementation:
    1) Rasterize cluster points (binary by default) + optional dilation for connectivity.
    2) Compute edges (Canny) and then corners using Shi-Tomasi (goodFeaturesToTrack).
    3) Convert corner count to [0,1] via saturating function: 1 - exp(-count / c).

    Output:
    - raw score in [0,1] (higher => more corners/junctions).
    """

    name = "corner_junction_density"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 80,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        canny_low: int = 50,
        canny_high: int = 150,
        max_corners: int = 120,
        quality_level: float = 0.01,
        min_distance: int = 6,
        block_size: int = 3,
        use_harris: bool = False,
        # saturation constant: ~score 0.63 at count=c
        sat_c: float = 10.0,
        eps: float = 1e-12,
    ):
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for image metrics. Install opencv-python.")

        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,          # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)

        self.max_corners = int(max_corners)
        self.quality_level = float(quality_level)
        self.min_distance = int(min_distance)
        self.block_size = int(block_size)
        self.use_harris = bool(use_harris)

        self.sat_c = float(max(1e-6, sat_c))
        self.eps = float(eps)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or np.max(img) == 0:
            return 0.0

        edges = canny_edges(img, low=self.canny_low, high=self.canny_high)
        if int(np.count_nonzero(edges)) < 25:
            return 0.0

        src = edges.astype(np.uint8)

        corners = cv2.goodFeaturesToTrack(
            src,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            useHarrisDetector=self.use_harris,
        )

        count = 0 if corners is None else int(corners.shape[0])

        # Density per edge length (corners per linear edge pixel). This way a
        # large filled blob (lots of jagged boundary corners) doesn't dominate
        # over a sparse but corner-heavy structure.
        edge_len = float(np.count_nonzero(edges))
        if edge_len <= self.eps:
            return 0.0
        density = float(count) / edge_len  # ~corners / unit edge length

        # Saturate density: ~0.03 corners per edge pixel already counts as
        # "rich in corners" for typical 256x256 rasters.
        score = 1.0 - np.exp(-density / 0.015)
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.6)
