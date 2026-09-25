# models/ClustOpt/cluster_validity_indices/metrics/image_base/morphological_band_count.py
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


def _dominant_angle_from_foreground(mask: np.ndarray) -> float:
    """
    Estimate dominant orientation of foreground pixels via PCA on (x,y).
    Returns angle in degrees in [-90,90] of principal axis.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return 0.0
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0, keepdims=True)
    X = pts - mean
    cov = (X.T @ X) / max(1.0, float(len(pts) - 1))
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    # map to [-90, 90]
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return ang


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _smooth_1d(x: np.ndarray, k: int = 9) -> np.ndarray:
    k = int(max(3, k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode="same")


def _find_peaks_1d(x: np.ndarray, min_dist: int = 8, min_prom: float = 0.05) -> list[int]:
    """
    Very simple peak finder on a normalized 1D signal.
    """
    if x.size < 3:
        return []
    min_dist = int(max(1, min_dist))
    peaks: list[int] = []
    for i in range(1, x.size - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            peaks.append(i)

    if not peaks:
        return []

    # compute prominence vs local baseline
    keep: list[int] = []
    for p in peaks:
        left = max(0, p - min_dist)
        right = min(x.size - 1, p + min_dist)
        baseline = float(min(np.min(x[left:p + 1]), np.min(x[p:right + 1])))
        prom = float(x[p] - baseline)
        if prom >= float(min_prom):
            keep.append(p)

    # enforce min_dist
    keep.sort(key=lambda i: x[i], reverse=True)
    selected: list[int] = []
    for p in keep:
        if all(abs(p - q) >= min_dist for q in selected):
            selected.append(p)
    return sorted(selected)


class MorphologicalBandCountMetric(ImagePerClusterMetric):
    """
    Morphological Band Count (per-cluster)

    Steps:
    1) rasterize cluster points -> binary/density image
    2) estimate dominant axis (PCA) and rotate to align bands horizontally
    3) apply morphological opening with a horizontal line kernel (enhances bands)
    4) build 1D profile by summing rows, smooth, detect peaks
    5) score favors:
       - many peaks (bands) BUT
       - approximately regular spacing (low CV of inter-peak distances)
       - strong peaks (profile energy)

    Returns score in [0,1].
    """

    name = "morphological_band_count"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 80,
        grid_size: int = 256,
        raster_mode: str = "binary",
        blur_ksize: int = 3,
        dilate_iters: int = 2,
        line_kernel_frac: float = 0.12,     # length relative to grid width
        profile_smooth_k: int = 11,
        peak_min_dist: int = 10,
        peak_min_prom: float = 0.06,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.raster_params = RasterParams(
            grid_size=int(grid_size),
            mode=raster_mode,  # type: ignore[arg-type]
            blur_ksize=int(blur_ksize),
            dilate_iters=int(dilate_iters),
        )
        self.line_kernel_frac = float(line_kernel_frac)
        self.profile_smooth_k = int(profile_smooth_k)
        self.peak_min_dist = int(peak_min_dist)
        self.peak_min_prom = float(peak_min_prom)

    def _cluster_score(self, Xi: np.ndarray, bounds: Bounds2D) -> float:
        if Xi.shape[0] < self.min_cluster_points:
            return float("-inf")

        img = rasterize_points(Xi, self.raster_params, bounds=bounds)
        if img is None or int(np.max(img)) == 0:
            return 0.0

        # binarize foreground
        mask = (img > 0).astype(np.uint8) * 255

        # rotate to align principal direction with x-axis
        ang = _dominant_angle_from_foreground(mask)
        mask_r = _rotate_image(mask, -ang)

        h, w = mask_r.shape[:2]
        klen = int(max(5, round(self.line_kernel_frac * w)))
        if klen % 2 == 0:
            klen += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        opened = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)

        # 1D row profile (sum over columns) -> bands produce peaks
        prof = opened.astype(np.float32).sum(axis=1)
        if float(np.max(prof)) <= 0.0:
            return 0.0
        prof = prof / float(np.max(prof))

        prof_s = _smooth_1d(prof, k=self.profile_smooth_k)
        peaks = _find_peaks_1d(prof_s, min_dist=self.peak_min_dist, min_prom=self.peak_min_prom)

        n = len(peaks)
        if n <= 1:
            return 0.0

        # spacing regularity
        d = np.diff(np.array(peaks, dtype=np.float32))
        if d.size == 0 or float(np.mean(d)) <= 1e-6:
            return 0.0
        cv = float(np.std(d) / (np.mean(d) + 1e-9))
        regular = float(1.0 / (1.0 + cv))  # in (0,1]

        # band strength proxy
        energy = float(np.mean(prof_s[peaks]))

        # favor more bands but saturate
        n_term = float(1.0 - np.exp(-0.35 * n))

        score = n_term * regular * energy
        return float(np.clip(score, 0.0, 1.0))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.6)
# Morphological Band Count Metric

