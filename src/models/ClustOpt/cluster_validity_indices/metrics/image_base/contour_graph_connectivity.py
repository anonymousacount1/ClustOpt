# models/ClustOpt/cluster_validity_indices/metrics/image_base/contour_graph_connectivity.py
from __future__ import annotations

import numpy as np
import cv2

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import (
    RasterParams,
    rasterize_points,
    canny_edges,
    Bounds2D,
)
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext


class ContourGraphConnectivityMetric(ImagePerClusterMetric):
    """
    Contour Graph Connectivity (per-cluster)

    Build edge map (Canny), treat edge pixels as a graph (8-neighborhood).
    Score high when:
    - few connected components in edge graph
    - largest component dominates (mass ratio)
    - total edge mass exists (not empty)

    score = exp(-a*(components-1)) * (largest_mass/total_mass)
    """

    name = "contour_graph_connectivity"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 80,
        grid_size: int = 256,
        raster_mode: str = "density",
        blur_ksize: int = 3,
        dilate_iters: int = 1,
        canny_low: int = 50,
        canny_high: int = 150,
        a_comp: float = 0.05,
        min_edges: int = 40,
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
        self.a_comp = float(a_comp)
        self.min_edges = int(min_edges)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or int(np.max(img)) == 0:
            return 0.0

        edges = canny_edges(img, low=self.canny_low, high=self.canny_high)
        n_edges = int(np.count_nonzero(edges))
        if n_edges < self.min_edges:
            return 0.0

        # Close small gaps in the edge map so that one underlying contour
        # doesn't appear as dozens of tiny edge segments.
        sk = (edges > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sk = cv2.morphologyEx(sk, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels = cv2.connectedComponents(sk, connectivity=8)
        comps = int(max(0, num_labels - 1))
        if comps <= 0:
            return 0.0

        # Component masses; filter out tiny segments (rasterisation noise).
        masses = np.array(
            [int(np.count_nonzero(labels == lab)) for lab in range(1, num_labels)],
            dtype=np.float32,
        )
        total = float(np.sum(masses))
        min_keep = max(8, int(0.005 * sk.size))
        sig_masses = masses[masses >= min_keep]
        if sig_masses.size == 0:
            return 0.0
        n_sig = int(sig_masses.size)
        largest = float(np.max(sig_masses))
        dom = float(largest / max(1.0, total))

        # Reward dominance of the largest component plus penalty for many
        # significant disjoint pieces. The penalty is gentle so a clean shape
        # with ~1 large + a few small pieces still scores high.
        comp_term = float(np.exp(-self.a_comp * max(0, n_sig - 1)))
        score = comp_term * dom
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.5)
