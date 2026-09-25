# metrics/image/hough_line_strength.py
from __future__ import annotations
import numpy as np
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import RasterParams, rasterize_points, canny_edges, Bounds2D

try:
    import cv2
except Exception:
    cv2 = None


class HoughLineStrengthMetric(ImagePerClusterMetric):
    name = "hough_line_strength"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 30,
        raster: RasterParams = RasterParams(grid_size=256, mode="binary", blur_ksize=3, dilate_iters=1),
        canny_low: int = 50,
        canny_high: int = 150,
        rho: float = 1.0,
        theta: float = np.pi / 180.0,
        threshold: int = 50,
        min_line_length: int = 30,
        max_line_gap: int = 5,
        eps: float = 1e-12,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster = raster
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.rho = float(rho)
        self.theta = float(theta)
        self.threshold = int(threshold)
        self.min_line_length = int(min_line_length)
        self.max_line_gap = int(max_line_gap)
        self.eps = float(eps)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for image metrics. Install opencv-python.")

        img = rasterize_points(Xi, self.raster, bounds=bounds)
        edges = canny_edges(img, low=self.canny_low, high=self.canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if lines is None:
            return 0.0

        # score by total detected line length normalized by number of edge pixels
        edge_count = float(np.count_nonzero(edges))
        if edge_count <= 0:
            return 0.0

        total_len = 0.0
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            total_len += float(np.hypot(x2 - x1, y2 - y1))

        # Score = how much of the detected edge length is explained by Hough
        # line segments. For a shape with many true straight lines this is
        # close to 1; for a soft blob the boundary is too curved for Hough
        # so the total line length stays well below the edge length.
        ratio = float(total_len / (edge_count + self.eps))
        score = float(np.clip(ratio, 0.0, 1.0))
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.6)
