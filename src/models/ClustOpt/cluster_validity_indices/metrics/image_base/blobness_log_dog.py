# metrics/image_base/blobness_log_dog.py
from __future__ import annotations

import numpy as np

from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_metric_base import ImagePerClusterMetric
from models.ClustOpt.cluster_validity_indices.metrics.image_base.utils.image_utils import (
    RasterParams,
    rasterize_points,
    Bounds2D,
)
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext

try:
    import cv2
except Exception:
    cv2 = None


class BlobnessLogDogMetric(ImagePerClusterMetric):
    """
    Blobness (LoG/DoG) — per-cluster image metric.

    Intuition:
    - "Blobs" (lobes, compact filled regions, dense cloud-islands) generate strong
      Laplacian-of-Gaussian / Difference-of-Gaussians responses at their centers.
    - Thin structures (lines/arcs) tend to yield weaker / less spatially concentrated responses.

    Implementation (deterministic, no history):
    1) Rasterize cluster points (default: density mode).
    2) Multi-scale DoG response: max over scales of |G(s1) - G(s2)|.
    3) Score is the fraction of response-energy concentrated in top-q% pixels,
       normalized by occupied pixels (to avoid "big cluster always higher").

    Output:
    - raw score already in [0,1] (higher is "more blob-like").
    """

    name = "blobness_log_dog"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 80,
        grid_size: int = 256,
        raster_mode: str = "density",
        blur_ksize: int = 0,
        dilate_iters: int = 0,
        # DoG scales (in pixels) — tuned to be resolution-robust
        sigmas: tuple[float, ...] = (1.2, 2.0, 3.2),
        dog_ratio: float = 1.6,
        # top quantile for energy concentration
        top_q: float = 0.03,
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
        self.sigmas = tuple(float(s) for s in sigmas)
        self.dog_ratio = float(dog_ratio)
        self.top_q = float(top_q)
        self.eps = float(eps)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img_u8 = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img_u8 is None:
            return float("-inf")

        # occupied pixels (after rasterization) — used for normalization
        occ = int(np.count_nonzero(img_u8))
        if occ < 20:
            return 0.0

        img = img_u8.astype(np.float32) / 255.0

        # Multi-scale DoG response (absolute)
        resp_max = None
        for s in self.sigmas:
            s1 = float(s)
            s2 = float(s) * self.dog_ratio
            g1 = cv2.GaussianBlur(img, (0, 0), sigmaX=s1, sigmaY=s1)
            g2 = cv2.GaussianBlur(img, (0, 0), sigmaX=s2, sigmaY=s2)
            dog = np.abs(g1 - g2)
            resp_max = dog if resp_max is None else np.maximum(resp_max, dog)

        if resp_max is None:
            return 0.0

        # Focus only on area that actually has content (avoid background dominating)
        mask = img_u8 > 0
        r = resp_max[mask]
        if r.size < 20:
            return 0.0

        r = np.clip(r, 0.0, None)

        total = float(np.sum(r))
        if total <= self.eps:
            return 0.0

        # Energy concentration in top-q pixels relative to the median pixel.
        # Using top-fraction / total has a built-in floor of `top_q` even for
        # perfectly uniform images, which makes the metric insensitive to
        # blob-like structure. Comparing the top-mean to the median is much
        # more sensitive: smooth images give peak/median ~ 1-2, while images
        # with clear blobs easily reach 10+.
        q = float(np.clip(self.top_q, 0.001, 0.25))
        k = int(max(1, round(q * r.size)))
        topk = np.partition(r, -k)[-k:]
        top_mean = float(np.mean(topk))
        med = float(np.median(r))
        denom = max(med, self.eps)
        peakiness = top_mean / denom  # 1 = no peaks, large = blobby

        # Map peakiness into [0,1] with a saturating curve calibrated so a
        # 5x ratio reaches ~0.8.
        conc = float(1.0 - np.exp(-(peakiness - 1.0) / 3.0))
        conc = float(max(0.0, conc))

        # Adjust for occupancy: if extremely sparse, avoid accidental spikes
        occ_factor = 1.0 - np.exp(-occ / 200.0)

        score = conc * occ_factor
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # The raw concentration ratio is bounded below by top_q itself (uniform
        # case), so even a clearly blob-like cluster rarely exceeds ~0.2. Map
        # the useful working range into a more discriminative [0,1] band.
        v = float(max(0.0, min(1.0, raw)))
        # piecewise log-like squash via sqrt so 0.05 -> 0.22, 0.15 -> 0.39,
        # 0.30 -> 0.55, 0.60 -> 0.77.
        return float(v ** 0.5)
